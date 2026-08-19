"""Installed-package smoke tests for the copick command plugin."""

import importlib
import importlib.metadata
import pkgutil
import sys

from click.testing import CliRunner

import copick_easymode


def test_all_implementation_modules_import_without_tensorflow():
    modules = [module.name for module in pkgutil.walk_packages(copick_easymode.__path__, "copick_easymode.")]

    assert modules
    assert "tensorflow" not in sys.modules
    for module in modules:
        importlib.import_module(module)
    assert "tensorflow" not in sys.modules


def test_copick_command_entry_point_loads_and_renders_help_offline(monkeypatch):
    def unexpected_network(*args, **kwargs):
        raise AssertionError("command help must not access the network")

    monkeypatch.setattr("socket.create_connection", unexpected_network)
    entry_points = [
        entry_point
        for entry_point in importlib.metadata.entry_points(group="copick.inference.commands")
        if entry_point.dist.name == "copick-easymode"
    ]

    assert len(entry_points) == 1
    assert "tensorflow" not in sys.modules
    result = CliRunner().invoke(entry_points[0].load(), ["--help"])
    assert result.exit_code == 0, result.output
    assert "--overwrite / --no-overwrite" in result.output
    assert "tensorflow" not in sys.modules
