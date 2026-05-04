import ast
from pathlib import Path


MODEL_CONFIGS = (
    Path("configs/model/ac_baware_av2_base.yaml"),
    Path("configs/model/ac_R2P_av2.yaml"),
    Path("configs/model/ac_R2P_womd.yaml"),
)

REMOVED_METRIC_KEYS = {
    "simple_metric",
    "offroad_metric",
    "kinematic_metric",
}

REMOVED_SELECTOR_DEBUG_TOKENS = (
    "return_debug",
    "return_selector_debug",
    "selector_debug_bank_topk",
    "selector_debug",
    "debug_bank_topk",
)


def test_main_drops_direct_diversity_metric_dependency():
    source = Path("src/main.py").read_text()

    assert "TrajectoryDiversity" not in source
    assert "self.diversity_metric" not in source


def test_future_motion_signature_drops_optional_metric_configs():
    module = ast.parse(Path("src/main.py").read_text())
    future_motion = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FutureMotion"
    )
    init_fn = next(
        node
        for node in future_motion.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    params = {arg.arg for arg in init_fn.args.args}

    for key in REMOVED_METRIC_KEYS:
        assert key not in params


def test_future_motion_init_saves_hparams_for_later_hparams_access():
    module = ast.parse(Path("src/main.py").read_text())
    future_motion = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FutureMotion"
    )
    init_fn = next(
        node
        for node in future_motion.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    save_calls = [
        node
        for node in ast.walk(init_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "save_hyperparameters"
    ]

    assert save_calls, "FutureMotion.__init__ must populate self.hparams"


def test_selector_debug_plumbing_removed_from_selector_stack():
    tracked_files = (
        Path("src/models/modules/trajectory_selector_softattn_tf.py"),
        Path("src/models/ac_model_R2P_gqa.py"),
        Path("src/main.py"),
    )

    for tracked_file in tracked_files:
        source = tracked_file.read_text()
        for token in REMOVED_SELECTOR_DEBUG_TOKENS:
            assert token not in source, f"{tracked_file} still references {token}"


def test_model_configs_drop_optional_metric_blocks():
    for config_path in MODEL_CONFIGS:
        config_text = config_path.read_text()
        for key in REMOVED_METRIC_KEYS:
            assert f"\n{key}:" not in config_text, f"{config_path} still defines {key}"
