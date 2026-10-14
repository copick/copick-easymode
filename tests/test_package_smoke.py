"""Installed-package smoke tests for the copick command plugin.

Each check runs in a fresh interpreter: other test modules import Keras (and with it TensorFlow) while they are
collected, so "importing the package does not import TensorFlow" can only be observed in a clean process.
"""

import subprocess
import sys
import textwrap

HEAVY = """
heavy = sorted({name.split(".")[0] for name in sys.modules} & {"tensorflow", "keras", "easymode"})
assert not heavy, f"imported {heavy}"
"""


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code) + HEAVY],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_all_implementation_modules_import_without_tensorflow_or_easymode():
    out = _run(
        """
        import importlib
        import pkgutil
        import sys

        import copick_easymode

        modules = [module.name for module in pkgutil.walk_packages(copick_easymode.__path__, "copick_easymode.")]
        assert "copick_easymode.core.easymode_env" in modules, modules
        for module in modules:
            importlib.import_module(module)
        print(len(modules))
        """,
    )
    assert int(out.split()[-1]) > 0


def test_copick_command_entry_point_loads_and_renders_help_offline():
    out = _run(
        """
        import importlib.metadata
        import socket
        import sys

        from click.testing import CliRunner


        def unexpected_network(*args, **kwargs):
            raise AssertionError("command help must not access the network")


        socket.create_connection = unexpected_network
        entry_points = [
            entry_point
            for entry_point in importlib.metadata.entry_points(group="copick.inference.commands")
            if entry_point.dist.name == "copick-easymode"
        ]
        assert len(entry_points) == 1, entry_points
        result = CliRunner().invoke(entry_points[0].load(), ["--help"])
        assert result.exit_code == 0, result.output
        print(result.output)
        """,
    )
    assert "--overwrite / --no-overwrite" in out
    assert "--model-dir" in out and "--report" in out
