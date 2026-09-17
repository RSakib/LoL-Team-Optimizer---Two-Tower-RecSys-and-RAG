"""Verify Gradio cannot accept requests before successful initialization."""
import runpy
import sys
from types import SimpleNamespace

import pytest

from scripts.package_space import ROOT


@pytest.mark.parametrize("startup_fails", [False, True])
def test_space_warms_before_launch_and_stops_on_failure(monkeypatch, startup_fails):
    from src import config, deployment, service
    events = []
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.setattr(config, "SCOUT_DEVICE", "cpu")
    monkeypatch.setattr(config, "UI_BACKEND", "local")
    monkeypatch.setattr(deployment, "validate_bundle", lambda root: events.append("validate"))

    def warm():
        events.append("warm")
        if startup_fails:
            raise RuntimeError("Missing real deployment evidence")

    class Demo:
        def queue(self, **kwargs):
            events.append("queue")
            return self

        def launch(self, **kwargs):
            events.append("launch")

    monkeypatch.setattr(service, "warm_runtime", warm)
    monkeypatch.setitem(sys.modules, "src.ui.app", SimpleNamespace(demo=Demo()))
    if startup_fails:
        with pytest.raises(RuntimeError, match="Missing real"):
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        assert events == ["validate", "warm"]
    else:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        assert events == ["validate", "warm", "queue", "launch"]
