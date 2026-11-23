#!/usr/bin/env python3
"""Serve trigger-driven frozen CoptV2X inference from Orin-local inputs."""

from __future__ import print_function

import argparse
import gc
import json
import socket
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from coptv2x_hil_artifacts import DEFAULT_CONFIG_DIR, HIL_ARM_IDS, load_hil_artifact_registry
from coptv2x_hil_backend import CoptV2xHilBackend
from coptv2x_hil_trigger import HilTriggerAdmission, parse_hil_trigger
from ditl_edge_protocol import ProtocolLimits, send_wire_frame
from ditl_rsu_pipeline import CoptV2xHilPipeline
from ditl_rsu_source import IndexedLocalRsuSource
from scripts.ditl_rsu_edge_publisher import (
    _connect_local_result_handoff,
    _prepare_result_frame,
)


MAX_TRIGGER_LINE = 16 * 1024


def parse_worker_trigger_line(line):
    """Decode exactly one newline-delimited, schema-validated trigger."""

    if not isinstance(line, bytes) or not line.endswith(b"\n") or len(line) > MAX_TRIGGER_LINE:
        raise ValueError("trigger line is malformed or too large")
    try:
        value = json.loads(line[:-1].decode("utf-8"))
        return parse_hil_trigger(value)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError("trigger payload violates the HIL trigger contract") from error


def _read_bounded_line(connection, maximum):
    chunks = bytearray()
    while len(chunks) <= maximum:
        value = connection.recv(1)
        if not value:
            raise EOFError("trigger client closed before newline")
        chunks.extend(value)
        if value == b"\n":
            return bytes(chunks)
    raise ValueError("trigger line exceeds configured maximum")


class OrinHilWorker(object):
    def __init__(self, args):
        self.args = args
        config_dir = Path(args.config_dir).resolve()
        registry = load_hil_artifact_registry(config_dir)
        first = registry[HIL_ARM_IDS[0]]
        input_contract = {
            name: {"shape": tuple(spec.shape), "dtype": spec.dtype}
            for name, spec in ((tensor.name, tensor) for tensor in first.input_contract)
        }
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        self.source = IndexedLocalRsuSource(
            args.data_root,
            manifest,
            input_contract,
            max_samples=args.max_samples,
            max_total_tensor_bytes=args.max_total_tensor_mib * 1024 * 1024,
            memory_map=True,
            lazy_load=True,
        )
        self.backend = None
        self.pipeline = None
        self.admission = None
        self.active_key = None
        self.processed = 0
        self.connection = _connect_local_result_handoff(
            args.result_host, args.result_port, args.connect_timeout_s
        )

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self._release_arm()

    def _release_arm(self):
        self.pipeline = None
        self.admission = None
        self.backend = None
        self.active_key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def process_trigger(self, trigger):
        if self.processed >= self.args.result_count:
            raise ValueError("trigger count exceeds frozen result_count")
        key = (trigger.run_id, trigger.route_id, trigger.scenario_id, trigger.arm_id)
        if self.active_key != key:
            self._release_arm()
            self.backend = CoptV2xHilBackend(
                trigger.arm_id,
                artifact_root=self.args.artifact_root,
                config_dir=self.args.config_dir,
                native_repo_root=self.args.native_repo_root,
            )
            self.backend.load(trigger.arm_id)
            self.pipeline = CoptV2xHilPipeline(
                self.backend,
                run_id=trigger.run_id,
                route_id=trigger.route_id,
                scenario_id=trigger.scenario_id,
                compression="none",
                limits=ProtocolLimits(
                    max_payload_bytes=self.args.max_payload_mib * 1024 * 1024,
                    max_raw_payload_bytes=self.args.max_payload_mib * 1024 * 1024,
                ),
            )
            self.admission = HilTriggerAdmission(
                trigger.run_id,
                trigger.route_id,
                trigger.scenario_id,
                trigger.arm_id,
            )
            self.active_key = key
        reserved = self.admission.reserve(trigger.to_metadata())
        try:
            frame = self.source.frame_for_trigger(reserved)
            self.admission.admit_trigger(reserved)
        except Exception:
            self.admission.cancel(reserved)
            raise
        result = self.pipeline.process(frame)
        header, payload = _prepare_result_frame(result, warmup=False)
        send_wire_frame(self.connection, header, payload)
        self.processed += 1
        return result.header["request_id"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--native-repo-root", required=True)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=56612)
    parser.add_argument("--result-host", default="127.0.0.1")
    parser.add_argument("--result-port", type=int, default=55021)
    parser.add_argument("--result-count", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-total-tensor-mib", type=int, default=262144)
    parser.add_argument("--max-payload-mib", type=int, default=512)
    parser.add_argument("--connect-timeout-s", type=float, default=180.0)
    parser.add_argument("--socket-timeout-s", type=float, default=180.0)
    parser.add_argument("--ready-file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker_host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("worker listener must stay on loopback")
    if args.result_count <= 0 or not 1 <= args.worker_port <= 65535:
        raise SystemExit("result count and worker port must be positive/valid")
    if args.max_total_tensor_mib <= 0 or args.max_payload_mib <= 0:
        raise SystemExit("memory and payload bounds must be positive")
    worker = OrinHilWorker(args)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.worker_host, args.worker_port))
    server.listen(1)
    server.settimeout(1.0)
    if args.ready_file:
        Path(args.ready_file).write_text(
            json.dumps({"ready": True, "expected_results": args.result_count}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"event": "orin_worker_ready", "expected_results": args.result_count}), flush=True)
    try:
        while worker.processed < args.result_count:
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            with client:
                client.settimeout(args.socket_timeout_s)
                request_id = None
                try:
                    trigger = parse_worker_trigger_line(_read_bounded_line(client, MAX_TRIGGER_LINE))
                    request_id = worker.process_trigger(trigger)
                    client.sendall(("DONE %s\n" % request_id).encode("ascii"))
                except Exception as error:
                    message = ("ERROR %s\n" % str(error).replace("\n", " "))[:2048]
                    try:
                        client.sendall(message.encode("utf-8", errors="replace"))
                    except OSError:
                        pass
                    raise
        print(json.dumps({"event": "orin_worker_complete", "results": worker.processed}), flush=True)
        return 0
    finally:
        server.close()
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
