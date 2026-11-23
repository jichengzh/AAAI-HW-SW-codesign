#!/usr/bin/env python3
"""Actual-Orin startup diagnostic; no requested-frame inference or driving."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "data-root", "manifest", "artifact-root", "native-repo-root"):
        parser.add_argument("--" + name, required=True)
    options = parser.parse_args()
    root = Path(options.runtime_root)
    sys.path.insert(0, str(root))
    from scripts.coptv2x_hil_orin_worker import OrinHilWorker, build_parser, HIL_ARM_IDS
    from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION, parse_hil_trigger
    document = json.loads(Path(options.manifest).read_text())
    first = document["samples"][0]
    for arm in HIL_ARM_IDS:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            arguments = ["--data-root", options.data_root, "--manifest", options.manifest,
                         "--artifact-root", options.artifact_root, "--native-repo-root", options.native_repo_root,
                         "--result-port", str(listener.getsockname()[1]), "--result-count", "2",
                         "--config-dir", str(root / "simulation/leaderboard/team_code/agent_config/coptv2x_hil"),
                         "--serial-optimized", "--startup-arm-id", arm, "--runtime-warmup-count", "1"]
            start = time.monotonic_ns()
            with redirect_stdout(sys.stderr):
                worker = OrinHilWorker(build_parser().parse_args(arguments))
            initialized_at = time.monotonic_ns()
            connected, _ = listener.accept()
            try:
                initialized = worker.backend
                assert initialized is not None and initialized.inference_count == 0
                assert worker.processed == 0 and worker.pipeline is None and worker.admission is None
                previous_pipeline = previous_admission = None
                for run in ("startup-diagnostic-A", "startup-diagnostic-B"):
                    trigger = parse_hil_trigger({
                        "schema_version": HIL_TRIGGER_SCHEMA_VERSION, "run_id": run,
                        "route_id": document["route_id"], "scenario_id": document["scenario_id"],
                        "sample_id": first["sample_id"], "source_frame": first["source_frame"],
                        "source_sim_time_ns": 0, "sequence_no": 0, "arm_id": arm,
                    })
                    worker._activate_episode(trigger, (run, trigger.route_id, trigger.scenario_id, arm))
                    assert worker.backend is initialized
                    assert worker.pipeline is not previous_pipeline and worker.admission is not previous_admission
                    previous_pipeline, previous_admission = worker.pipeline, worker.admission
                connected.settimeout(0.05)
                try:
                    unexpected = connected.recv(1)
                except socket.timeout:
                    unexpected = None
                assert unexpected is None
                print(json.dumps({"event": "actual_orin_startup_diagnostic", "arm_id": arm,
                    "passed": True, "constructor_elapsed_ns": initialized_at - start,
                    "diagnostic_elapsed_ns": time.monotonic_ns() - start,
                    "runtime_warmup_count": 1, "requested_inference_count": initialized.inference_count,
                    "requested_results_published": worker.processed, "same_process_model_reused": True,
                    "fresh_episode_admission": True, "is_carla": False, "is_ros": False,
                    "scope": "synthetic_startup_warmup_and_same_source_run_transition_only"}), flush=True)
            finally:
                connected.close()
                worker.close()


if __name__ == "__main__":
    main()
