"""
easymode's 2D-engine models (``.scnm``), run in this process.

Most easymode features are published as Ais ``.scnm`` files rather than the ``.h5`` weights
``easymode.core.distribution.load_model`` builds. easymode's own CLI runs them by shelling out
to ``ais segment``; Ais pins TensorFlow <= 2.11 and Python <= 3.10 and pulls in a GUI stack,
so it cannot share an environment with copick. A ``.scnm`` is an uncompressed tar of a full
Keras model (``<title>_weights.h5``) and its metadata (``<title>_metadata.json``), and the
model loads in Keras 3 once the ``groups`` argument Keras 2 wrote into its transposed
convolutions is dropped. So this module loads it here and ports Ais's inference:
``_preprocess_tomo``, ``_segmentation_thread`` and ``_infer_slab`` of
``Ais/core/cli_fn.py`` (ais-cryoet 1.2.38, Mart G. F. So-Last, GPL-3.0 like this package),
with the defaults easymode's ``ais segment`` call leaves in place (whole Z range, batch 1,
512 px XY tiles with a 64 px discarded border, no post-blur).

The result is the same 0-1 probability map ``segment_tomogram_from_array`` returns for an
``.h5`` model, so thresholding and everything downstream is shared.
"""

import glob
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

SCNM_SUFFIX = ".scnm"
#: Ais's test-time augmentation: four 90-degree rotations, then the same four flipped.
MAX_TTA = 8
_TTA_ROTATIONS = (0, 1, 2, 3, 0, 1, 2, 3)
_TTA_FLIPS = (0, 0, 0, 0, 1, 1, 1, 1)
#: Ais's slab-model XY tiling defaults (``INFERENCE_TILE_3D``, ``INFERENCE_OVERLAP_PX_3D``).
SLAB_TILE = 512
SLAB_OVERLAP = 64
NORM_GLOBAL_MAD = "global_mad"


@dataclass(frozen=True)
class ScnmModel:
    """A loaded ``.scnm`` model and the metadata its inference needs."""

    model: Any
    title: str
    apix: float
    depth: int
    normalization: Any
    z_jitter: int

    @property
    def is_slab(self) -> bool:
        """A slab model emits a full Z-slab (rank-5 output) and is tiled in Z as well as XY."""
        return len(self.model.output_shape) == 5

    @property
    def dimensionality(self) -> float:
        if len(self.model.input_shape) == 5:
            return 3
        return 2.5 if self.depth > 1 else 2


def is_scnm(path) -> bool:
    return str(path).lower().endswith(SCNM_SUFFIX)


def _without_groups(base):
    """``base`` accepting (and requiring to be 1) the ``groups`` argument Keras 2 serialized."""

    class Layer(base):
        def __init__(self, *args, groups=1, **kwargs):
            if groups not in (1, None):
                raise ValueError(f"{base.__name__} with groups={groups} has no Keras 3 equivalent")
            super().__init__(*args, **kwargs)

    Layer.__name__ = base.__name__
    return Layer


def load_scnm(path) -> ScnmModel:
    """Load an Ais ``.scnm`` model for inference, with a batch- and size-free input."""
    import keras

    custom_objects = {
        "Conv2DTranspose": _without_groups(keras.layers.Conv2DTranspose),
        "Conv3DTranspose": _without_groups(keras.layers.Conv3DTranspose),
    }
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path, "r") as archive:
            archive.extractall(tmp, filter="data")
        weights = glob.glob(os.path.join(tmp, "*_weights.h5"))
        metadata = glob.glob(os.path.join(tmp, "*_metadata.json"))
        if not weights or not metadata:
            raise ValueError(f"{path}: not an Ais model (no *_weights.h5 and *_metadata.json inside)")
        with open(metadata[0]) as fh:
            meta = json.load(fh)
        saved = keras.models.load_model(weights[0], custom_objects=custom_objects, compile=False)

    depth = int(meta.get("model_depth", 1) or 1)
    if len(saved.input_shape) == 4:
        shape = (None, None, depth)
    elif len(saved.input_shape) == 5:
        shape = (None, None, depth, 1)
    else:
        raise ValueError(f"{path}: unsupported model input {saved.input_shape}")
    inp = keras.Input(shape=shape)
    model = keras.models.clone_model(saved, input_tensors=[inp] if isinstance(saved.input, list) else inp)
    model.set_weights(saved.get_weights())
    return ScnmModel(
        model=model,
        title=meta.get("title", os.path.splitext(os.path.basename(str(path)))[0]),
        apix=float(meta["apix"]),
        depth=depth,
        normalization=meta.get("normalization"),
        z_jitter=int(meta.get("z_jitter", 0) or 0),
    )


def _global_stats(volume, n_slices=32):
    """Ais ``normalization.global_stats`` (global_mad): mean and 1.4826 x MAD over the central
    XY region of up to 32 evenly spaced Z-slices."""
    z, k, w = volume.shape
    zidx = np.linspace(0, z - 1, min(n_slices, z)).astype(int)
    mk = min(int(0.2 * k), 64)
    ml = min(int(0.2 * w), 64)
    region = np.asarray(volume[zidx][:, mk : (k - mk) if mk else k, ml : (w - ml) if ml else w], dtype=np.float32)
    center = float(np.mean(region))
    med = np.median(region)
    return center, float(1.4826 * np.median(np.abs(region - med))) + 1e-7


def _scale_xy(volume, factor):
    from skimage.transform import resize

    if abs(1.0 - factor) < 0.05:
        return volume
    shape = np.round([volume.shape[0], volume.shape[1] * factor, volume.shape[2] * factor]).astype(int)
    return resize(volume, shape, anti_aliasing=True)


def _pad_xy(volume):
    _, k, w = volume.shape
    pad_k = ((32 - (k % 32)) % 32) + 64
    pad_l = ((32 - (w % 32)) % 32) + 64
    pk = (pad_k // 2, pad_k - pad_k // 2)
    pl = (pad_l // 2, pad_l - pad_l // 2)
    return np.pad(volume, ((0, 0), pk, pl), mode="reflect"), (*pk, *pl)


def _unpad_xy(volume, padding):
    pt, pb, pl, pr = padding
    return volume[:, pt : None if pb == 0 else -pb, pl : None if pr == 0 else -pr]


def _preprocess(volume, data_apix, scnm: ScnmModel):
    if scnm.normalization == NORM_GLOBAL_MAD and volume.dtype == np.int8:
        volume = volume.view(np.uint8)  # Ais reads int8 maps as uint8 under global_mad
    volume = np.array(volume, dtype=np.float32)
    global_norm = scnm.normalization == NORM_GLOBAL_MAD
    if global_norm:  # whole-volume statistic at native resolution, before rescaling
        center, scale = _global_stats(volume)
        volume -= center
        volume /= scale
    volume = _scale_xy(volume, float(data_apix) / scnm.apix)
    if not global_norm:  # legacy models: per-slice
        for k in range(volume.shape[0]):
            sl = volume[k]
            volume[k] = (sl - sl.mean()) / (sl.std() + 1e-6)
    return _pad_xy(volume)


def _call(model, batch):
    return model([batch] if isinstance(model.input, list) else batch, training=False).numpy()


def _slice_input(vi, j, depth, dimensionality):
    n = vi.shape[0]
    if dimensionality == 2 or depth <= 1:
        return vi[np.clip(j, 0, n - 1)][..., np.newaxis]
    half = depth // 2
    slab = vi[np.clip(np.arange(j - half, j - half + depth), 0, n - 1)]
    if dimensionality == 3:
        return np.transpose(slab, (1, 2, 0))[..., np.newaxis]
    return np.transpose(slab, (1, 2, 0))


def _infer_slices(scnm: ScnmModel, vi, batch_size):
    si = np.zeros_like(vi, dtype=np.float32)
    n = vi.shape[0]
    for start in range(0, n, batch_size):
        js = list(range(start, min(start + batch_size, n)))
        out = _call(scnm.model, np.stack([_slice_input(vi, j, scnm.depth, scnm.dimensionality) for j in js]))
        for i, j in enumerate(js):
            si[j] = np.squeeze(out[i])
    return si


def _tile_starts(n, tile, stride):
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, stride))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def _infer_slab(scnm: ScnmModel, vi, tile=SLAB_TILE, overlap=SLAB_OVERLAP):
    """Ais ``_infer_slab`` over the whole Z range: depth-``depth`` slabs at 50% Z overlap, each
    slice weighted 1 within ``z_jitter / 2`` of the slab centre (the trained range) and 0
    outside it; in XY, overlapping tiles of which only the centre (``overlap`` px in) counts."""
    depth = scnm.depth
    nz, ny, nx = vi.shape
    stride = max(1, depth // 2)
    if nz <= depth:
        starts = [0]
    else:
        hi = nz - depth
        starts = list(range(0, hi + 1, stride))
        if starts[-1] != hi:
            starts.append(hi)

    jitter_half = scnm.z_jitter // 2
    if jitter_half > 0:
        w = (np.abs(np.arange(depth) - depth // 2) <= jitter_half).astype(np.float32)
    else:
        w = np.ones(depth, dtype=np.float32)

    t = max(32, int(tile) // 32 * 32)
    margin = min(int(overlap), t // 3)
    context = max(64, margin)

    def pad_for_tiles(n):
        step = t - 2 * margin
        padded = max(t, n + 2 * context)
        padded += (step - (padded - t) % step) % step
        return context, padded - n - context

    (ply, phy), (plx, phx) = pad_for_tiles(ny), pad_for_tiles(nx)
    py, px = ny + ply + phy, nx + plx + phx

    mask = np.ones((t, t), dtype=np.float32)
    if margin > 0:
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0

    acc = np.zeros_like(vi, dtype=np.float32)
    cnt_z = np.zeros(nz, dtype=np.float32)
    cnt_xy = np.zeros((ny, nx), dtype=np.float32)
    seen_z, seen_xy = set(), set()
    for z0 in starts:
        slab = vi[z0 : z0 + depth]
        if slab.shape[0] < depth:  # a volume thinner than one slab
            slab = np.pad(slab, ((0, depth - slab.shape[0]), (0, 0), (0, 0)), mode="reflect")
        slab = np.pad(slab, ((0, 0), (ply, phy), (plx, phx)), mode="reflect")
        z1 = min(z0 + depth, nz)
        d = z1 - z0
        for y0 in _tile_starts(py, t, t - 2 * margin):
            for x0 in _tile_starts(px, t, t - 2 * margin):
                oy, ox = y0 - ply, x0 - plx
                sy0, sy1 = max(0, oy), min(ny, oy + t)
                sx0, sx1 = max(0, ox), min(nx, ox + t)
                if sy0 >= sy1 or sx0 >= sx1:
                    continue
                inp = np.transpose(slab[:, y0 : y0 + t, x0 : x0 + t], (1, 2, 0))[np.newaxis, ..., np.newaxis]
                out = np.squeeze(_call(scnm.model, inp), axis=(0, -1))  # (tile, tile, depth)
                ty0, tx0 = sy0 - oy, sx0 - ox
                m = mask[ty0 : ty0 + (sy1 - sy0), tx0 : tx0 + (sx1 - sx0)]
                o = np.transpose(out, (2, 0, 1))[:d, ty0 : ty0 + (sy1 - sy0), tx0 : tx0 + (sx1 - sx0)]
                acc[z0:z1, sy0:sy1, sx0:sx1] += w[:d, None, None] * m[None, :, :] * o
                if (y0, x0) not in seen_xy:
                    cnt_xy[sy0:sy1, sx0:sx1] += m
                    seen_xy.add((y0, x0))
        if z0 not in seen_z:
            cnt_z[z0:z1] += w[:d]
            seen_z.add(z0)

    si = np.zeros_like(vi, dtype=np.float32)
    for z in range(nz):
        if cnt_z[z] > 0:
            dz = cnt_z[z] * cnt_xy
            np.divide(acc[z], dz, out=si[z], where=dz > 0)
    return si


def segment_volume_scnm(scnm: ScnmModel, volume: np.ndarray, data_apix: float, tta: int = 1, batch_size: int = 1):
    """Segment a (Z, Y, X) tomogram at ``data_apix`` A/px with an ``.scnm`` model.

    Returns a float32 probability map in [0, 1] of the input's shape. ``tta`` above
    :data:`MAX_TTA` is an error here; the caller decides whether to clamp and say so.
    """
    from scipy.ndimage import zoom

    if not 1 <= tta <= MAX_TTA:
        raise ValueError(f"tta must be between 1 and {MAX_TTA} for an .scnm model, got {tta}")
    original_shape = volume.shape
    prepared, padding = _preprocess(volume, data_apix, scnm)
    pt, pb, pl, pr = padding
    seg = np.zeros((prepared.shape[0], prepared.shape[1] - pt - pb, prepared.shape[2] - pl - pr), dtype=np.float32)
    for k in range(tta):
        vi = np.rot90(prepared, k=_TTA_ROTATIONS[k], axes=(1, 2))
        if _TTA_FLIPS[k]:
            vi = np.flip(vi, axis=2)
        vi = np.ascontiguousarray(vi)
        si = _infer_slab(scnm, vi) if scnm.is_slab else _infer_slices(scnm, vi, batch_size)
        if _TTA_FLIPS[k]:
            si = np.flip(si, axis=2)
        si = np.rot90(si, k=-_TTA_ROTATIONS[k], axes=(1, 2))
        seg += _unpad_xy(si, padding)
    seg = np.clip(seg / tta, 0.0, 1.0)
    if seg.shape != tuple(original_shape):
        seg = zoom(
            seg,
            (1.0, original_shape[1] / seg.shape[1], original_shape[2] / seg.shape[2]),
            order=1,
            prefilter=False,
        )
    return seg.astype(np.float32)
