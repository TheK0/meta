from __future__ import annotations

from pathlib import Path

from fsmol_cliff.adapters import AdapterSpec
from fsmol_cliff.fsmol_bridge import load_callable_from_spec


def test_load_callable_from_spec_imports_function_from_script(tmp_path: Path) -> None:
    script = tmp_path / "custom_adapter.py"
    script.write_text(
        "def run_adapter(value):\n"
        "    return {'echo': value}\n"
    )
    spec = AdapterSpec(
        name="custom",
        script_path="custom_adapter.py",
        callable_name="run_adapter",
        requires_checkpoint=False,
        family="test",
    )

    loaded = load_callable_from_spec(spec, root=tmp_path)

    assert loaded("ok") == {"echo": "ok"}
