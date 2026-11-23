#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import socket
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSEDLOOP = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "closedloop"
if str(CLOSEDLOOP) not in sys.path:
    sys.path.insert(0, str(CLOSEDLOOP))

from ditl_edge_protocol import ProtocolLimits
from ditl_edge_runtime import ReplayBackend, handle_connection


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Serve replay or TensorRT RSU perception over the DitL wire contract."
    )
    parser.add_argument("--backend", choices=("replay", "tensorrt"), required=True)
    parser.add_argument("--engine", help="Fixed TensorRT engine path for the service")
    parser.add_argument("--model-id")
    parser.add_argument("--replay-scale", type=float, default=1.0)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55020)
    parser.add_argument("--allow-client", action="append", default=[])
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--socket-timeout-s", type=float, default=120.0)
    parser.add_argument("--response-compression", choices=("none", "zlib"), default="none")
    parser.add_argument("--max-payload-mib", type=int, default=512)
    parser.add_argument("--ready-file")
    return parser.parse_args(argv)


def build_backend(args):
    if args.backend == "replay":
        return ReplayBackend(scale=args.replay_scale, model_id=args.model_id or "replay-rsu-v1")
    if not args.engine:
        raise ValueError("--engine is required for the tensorrt backend")
    from ditl_trt_backend import TensorRtBackend

    return TensorRtBackend(args.engine, model_id=args.model_id)


def write_ready_file(path, payload):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.max_requests < 0:
        raise SystemExit("--max-requests must be non-negative")
    if args.socket_timeout_s <= 0:
        raise SystemExit("--socket-timeout-s must be positive")
    if args.max_payload_mib <= 0:
        raise SystemExit("--max-payload-mib must be positive")
    backend = build_backend(args)
    limits = ProtocolLimits(
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_raw_payload_bytes=args.max_payload_mib * 1024 * 1024,
    )
    allow_clients = frozenset(args.allow_client)
    served = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind, args.port))
        listener.listen(8)
        ready = {
            "event": "ready",
            "backend": args.backend,
            "bind": args.bind,
            "port": listener.getsockname()[1],
            "model_id": backend.model_id,
        }
        write_ready_file(args.ready_file, ready)
        print(json.dumps(ready, sort_keys=True), flush=True)
        while args.max_requests == 0 or served < args.max_requests:
            connection, address = listener.accept()
            with connection:
                connection.settimeout(args.socket_timeout_s)
                if allow_clients and address[0] not in allow_clients:
                    print(
                        json.dumps(
                            {"event": "rejected_client", "peer": address[0]},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                response = handle_connection(
                    connection,
                    address,
                    backend,
                    response_compression=args.response_compression,
                    limits=limits,
                )
                served += 1
                print(
                    json.dumps(
                        {
                            "event": "request_finished",
                            "request_id": response.get("request_id"),
                            "kind": response.get("kind"),
                            "served": served,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
