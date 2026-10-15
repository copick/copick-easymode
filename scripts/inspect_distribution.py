"""Fail builds whose artifacts do not expose the package contract."""

import argparse
import tarfile
import zipfile
from pathlib import Path


def inspect_distributions(dist_dir: Path) -> tuple[Path, Path]:
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"Expected one wheel and one source distribution, found {wheels!r} and {sdists!r}")

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        entry_point_names = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_point_names) != 1:
            raise ValueError("Wheel must contain exactly one METADATA and one entry_points.txt file")

        entry_points = archive.read(entry_point_names[0]).decode()
        if (
            "[copick.inference.commands]" not in entry_points
            or "easymode = copick_easymode.cli.inference:easymode" not in entry_points
        ):
            raise ValueError("Wheel does not register the copick inference easymode command")

    sdist = sdists[0]
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        if not any(name.endswith("/uv.lock") for name in names):
            raise ValueError("Source distribution does not contain uv.lock")

    return wheel, sdist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    wheel, sdist = inspect_distributions(args.dist_dir)
    print(f"Validated {wheel.name} and {sdist.name}")


if __name__ == "__main__":
    main()
