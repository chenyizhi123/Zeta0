import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/data/build_downstream_exclusion_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_downstream_exclusion_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_downstream_versions_share_raw_episode_lineage(tmp_path):
    for version in ("sonic_refpose_v3_1", "twist2_refpose_v3_1"):
        meta = tmp_path / "HSI_task" / version / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    rows = MODULE.build_exclusion_rows(tmp_path, seed=101)
    assert len(rows) == 10
    assert all(len(row["paths"]) == 2 for row in rows)
    assert all(row["lineage_ids"][0].startswith("HumanoidArena_raw:HSI_task") for row in rows)
    assert {row["split"] for row in rows} == {"FT_train", "FT_val", "FT_test"}
