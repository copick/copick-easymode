"""
XLA-compiled inference (``--xla``): a model whose forward pass runs as one XLA-compiled function.

XLA fuses the memory-bound work between a network's convolutions (normalization, activations, residual adds), which
dominates a 3D UNet's time on large tiles. On real tomograms at TTA 4 this took the easymode microtubule ``.h5`` network
2.9x faster and the actin ``.scnm`` slab model 1.3x faster, with masks that agree with the plain model's to an IoU of
0.9998 or better. It is opt-in: compiling can fail, which falls back to the plain model, but a missing or too old
``ptxas`` can also abort the process inside TensorFlow, which no fallback can catch.

It is the option easymode 1.2.6 has as ``easymode segment --xla`` and ais-cryoet 1.2.40 as ``ais segment --xla``. This
package keeps its own wrapper rather than easymode's ``XLAModel``: the ``.scnm`` inference reads the network's shapes
through it, each compiled shape is logged, and a fallback's reason goes into the report. Ais's ``_XLAModel`` is never
reached, since ``.scnm`` models run here rather than through ``ais segment``.
"""

import logging
import time
from typing import Optional


def _shape_label(shapes) -> str:
    return ", ".join(f"({', '.join('?' if d is None else str(d) for d in shape)})" for shape in shapes)


class XLAModel:
    """A Keras model whose forward pass runs as one XLA-compiled function.

    Called like the model, ``m(x, training=False)``, so it drops in wherever the model is called: easymode's tiling of
    an ``.h5`` network and the ``.scnm`` inference alike. Every other attribute (``input``, ``input_shape``,
    ``output_shape``, ...) is the wrapped model's. Inference only: the compiled function always runs with
    ``training=False``.

    The first call for each input shape compiles (tens of seconds for a large network) and is logged with its time;
    later calls of that shape reuse it. If a call raises, e.g. because XLA cannot find libdevice, one warning is
    logged, the reason is kept in :attr:`fallback`, and this call and every later one run the plain model.
    """

    def __init__(self, model, logger=None, name: str = "model"):
        import tensorflow as tf

        self.model = model
        self.name = name
        #: Why the compiled function was given up (``"<error type>: <first line>"``), or None while it is in use.
        self.fallback: Optional[str] = None
        self._log = logger if logger is not None else logging.getLogger(__name__)
        self._tf = tf
        self._compiled = tf.function(lambda x: model(x, training=False), jit_compile=True)
        self._shapes = set()

    def __getattr__(self, name):
        # Reached only for what the wrapper itself lacks. "model" missing means __init__ has not run (unpickling).
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def __call__(self, x, training=False):
        if self.fallback is None:
            tf = self._tf
            inputs = tf.nest.map_structure(tf.convert_to_tensor, x)
            shapes = tuple(tuple(t.shape) for t in tf.nest.flatten(inputs))
            started = time.perf_counter()
            try:
                output = self._compiled(inputs)
            except Exception as exc:
                reason = (str(exc).strip().splitlines() or [""])[0][:200]
                self.fallback = f"{type(exc).__name__}: {reason}"
                self._log.warning(f"XLA failed for {self.name} ({self.fallback}); continuing without XLA")
            else:
                if shapes not in self._shapes:
                    self._shapes.add(shapes)
                    self._log.info(
                        f"XLA compiled {self.name} for input {_shape_label(shapes)}: first call "
                        f"{time.perf_counter() - started:.1f} s",
                    )
                return output
        return self.model(x, training=False)
