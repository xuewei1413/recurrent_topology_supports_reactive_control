"""Audit the sole unchanged-condition discrepancy seen in the local pilot."""
import importlib.util
import json

from local_replication.run_current import OUT, build_controller, np, gym, write_json, digest
from _mediation.mc_injector import run_cell, derive_seed


def main():
    task, seed, d, rep, alpha = "Acrobot", 4012, 1, 1, 0.5
    episodes = list(range(86400, 86420))
    old_source = OUT / "source_before_fix/mc_injector.py"
    spec = importlib.util.spec_from_file_location("before_fix_injector", old_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved_path = OUT / "current/single/Acrobot/rescue_Acrobot_s4012_d1_r1_self_exact_a0p5.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    runs = {}
    for name, function in (("current_repeat1", run_cell), ("current_repeat2", run_cell),
                           ("before_fix_in_local_environment", module.run_cell)):
        env = gym.make("Acrobot-v1")
        try:
            rows = function(build_controller(task, seed, d, rep), build_controller(task, seed),
                            env, episodes, alpha, "self_exact",
                            derive_seed("run", task, seed, d, rep, "self_exact", alpha, "rescue"), keep_trace=True)
        finally:
            env.close()
        np.testing.assert_array_equal([row["return"] for row in rows], [row["return"] for row in saved["rows"]])
        runs[name] = rows
    write_json(OUT / "current/single/partial_exact_discrepancy_audit.json",
               {"all_local_returns_identical": True, "source_before_fix_sha256": digest(old_source.read_bytes()),
                "saved_cell": saved_path.relative_to(OUT).as_posix(), "runs": runs,
                "interpretation": "The injector fixes do not explain this partial-exact discrepancy. "
                "Historical environment and full executed source are unavailable; cause is not attributed."})
    print("Both current repeats and the before-fix injector match the saved local 20-episode run.")


if __name__ == "__main__":
    main()
