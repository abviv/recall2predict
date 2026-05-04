from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text()


def test_trajectory_selector_imports_mlps_from_layers_in_my_way():
    source = _read("src/models/modules/trajectory_selector_softattn_tf.py")

    assert "from layers_in_my_way.modules.mlp import MlpS" in source
    assert "from src.models.modules.flash_attention import MlpS" not in source


def test_ac_model_imports_mlps_from_layers_in_my_way():
    source = _read("src/models/ac_model_R2P_gqa.py")

    assert "from layers_in_my_way.modules.mlp import MlpS" in source
    assert "from src.models.modules.flash_attention import MlpS" not in source
