from __future__ import print_function

from dataclasses import FrozenInstanceError
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ditl_audit import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    AuditContext,
    DitlAuditError,
    DitlAuditRecord,
    DitlAuditWriter,
    build_audit_record,
)
from ditl_event_scheduler import (  # noqa: E402
    DeliveryEvent,
    DeliveryOutcome,
    LogicalMessage,
    ReplayCounters,
    ReplayTick,
    StreamSelection,
    SubmissionOutcome,
)
from ditl_impairment_policy import (  # noqa: E402
    DeliveryPlan,
    PolicyDecision,
    RNG_VERSION,
)


EXPECTED_FIELDS = {
    "schema_version", "run_id", "route_id", "stream_id",
    "consume_frame", "consume_sim_time_ns", "submitted_message_id",
    "submitted_source_frame", "trace_sample_id", "cycle_index",
    "used_message_id", "used_source_frame", "source_sim_time_ns",
    "ready_time_ns", "base_delay_ns", "effective_delay_ns", "aoi_ns",
    "zoh_hold_ns", "is_zoh", "is_startup_ego_only", "is_expired",
    "arrival_deadline_miss", "fusion_deadline_miss", "drop_reason",
    "reorder_status", "duplicate_status", "policy_seed",
    "policy_config_hash", "rng_version", "queue_depth",
    "accepted_count", "dropped_count", "expired_count",
}


def context():
    return AuditContext(
        run_id="run-1",
        route_id="route-1",
        policy_seed=20260923,
        policy_config_hash="a" * 64,
        rng_version=RNG_VERSION,
    )


def submission(drop=False, reorder_ns=0, duplicate=False, sequence=7,
               effective_delay_ns=25):
    message = LogicalMessage(
        logical_message_id="message-%d" % sequence,
        stream_id="rsu-1",
        source_id="source-rsu-1",
        source_frame=sequence,
        source_sim_time_ns=100,
        sequence_id=sequence,
        payload={},
        trace_sample_id="sample-3",
        base_delay_ns=20,
        cycle_index=2,
    )
    plans = () if drop else (
        DeliveryPlan(0, effective_delay_ns, reorder_ns),
        DeliveryPlan(1, effective_delay_ns, reorder_ns) if duplicate else None,
    )
    plans = tuple(item for item in plans if item is not None)
    decision = PolicyDecision(
        dropped=drop,
        drop_reason="bernoulli" if drop else None,
        delay_clamped=False,
        loss_uniform=0.25,
        duplicate_uniform=0.25,
        burst_state="disabled",
        deliveries=plans,
    )
    events = tuple(
        DeliveryEvent(
            delivery_id="rsu-1:message-%d:%d" % (sequence, plan.copy_index),
            message=message,
            copy_index=plan.copy_index,
            effective_delay_ns=plan.effective_delay_ns,
            ready_time_ns=(message.source_sim_time_ns +
                           plan.effective_delay_ns + plan.reorder_delay_ns),
            expiry_time_ns=200,
            reorder_delay_ns=plan.reorder_delay_ns,
        )
        for plan in plans
    )
    return SubmissionOutcome(message, decision, events)


def tick(kind="ordinary", outcome_status=None, reorder_ns=0,
         arrival_miss=False):
    selected = StreamSelection(
        stream_id="rsu-1",
        used_message_id="message-7" if kind in ("ordinary", "zoh") else None,
        used_sequence_id=7 if kind in ("ordinary", "zoh") else None,
        payload={} if kind in ("ordinary", "zoh") else None,
        source_frame=7 if kind in ("ordinary", "zoh") else None,
        source_sim_time_ns=100 if kind in ("ordinary", "zoh") else None,
        ready_time_ns=125 if kind in ("ordinary", "zoh") else None,
        aoi_ns=50 if kind in ("ordinary", "zoh") else None,
        zoh_hold_ns=25 if kind == "zoh" else (0 if kind == "ordinary" else None),
        is_zoh=kind == "zoh",
        is_startup_ego_only=kind == "startup",
        is_expired=kind == "expired",
        fusion_deadline_miss=kind == "zoh",
    )
    delivery_outcomes = ()
    if outcome_status is not None:
        event = submission(reorder_ns=reorder_ns, duplicate=True).enqueued[1]
        delivery_outcomes = (
            DeliveryOutcome(event, outcome_status, arrival_miss),
        )
    counters = ReplayCounters(
        1 if kind in ("ordinary", "zoh") else 0,
        1 if kind == "drop" else 0,
        1 if kind == "expired" else 0,
    )
    return ReplayTick(9, 150, {"rsu-1": selected}, delivery_outcomes, 3,
                      counters)


def record(frame=9, kind="ordinary", submitted=None, **tick_kwargs):
    made = build_audit_record(
        tick(kind=kind, **tick_kwargs),
        submission() if submitted is None else submitted,
        "rsu-1",
        context(),
    )
    if frame == 9:
        return made
    values = dict(made.values)
    values["consume_frame"] = frame
    return DitlAuditRecord(values)


class AuditRecordTest(unittest.TestCase):
    def test_ordinary_record_has_exact_complete_schema_and_values(self):
        made = record(outcome_status="accepted", arrival_miss=True)
        self.assertEqual(EXPECTED_FIELDS, set(made.values))
        expected = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "run_id": "run-1", "route_id": "route-1",
            "stream_id": "rsu-1", "consume_frame": 9,
            "consume_sim_time_ns": 150,
            "submitted_message_id": "message-7",
            "submitted_source_frame": 7, "trace_sample_id": "sample-3",
            "cycle_index": 2, "used_message_id": "message-7",
            "used_source_frame": 7, "source_sim_time_ns": 100,
            "ready_time_ns": 125, "base_delay_ns": 20,
            "effective_delay_ns": 25, "aoi_ns": 50, "zoh_hold_ns": 0,
            "is_zoh": False, "is_startup_ego_only": False,
            "is_expired": False, "arrival_deadline_miss": True,
            "fusion_deadline_miss": False, "drop_reason": None,
            "reorder_status": "none", "duplicate_status": "none",
            "policy_seed": 20260923, "policy_config_hash": "a" * 64,
            "rng_version": RNG_VERSION, "queue_depth": 3,
            "accepted_count": 1, "dropped_count": 0,
            "expired_count": 0,
        }
        self.assertEqual(expected, dict(made.values))

    def test_startup_zoh_drop_and_expiry_keep_complete_schema(self):
        cases = (
            ("startup", submission(), None, True, False, None,
             "none", "none"),
            ("zoh", submission(), "message-7", False, True, None,
             "none", "none"),
            ("drop", submission(drop=True), None, False, False, "bernoulli",
             "not_applicable", "not_applicable"),
            ("expired", submission(), None, False, False, None,
             "none", "none"),
        )
        for (kind, submitted, used, startup, zoh, drop_reason,
             reorder_status, duplicate_status) in cases:
            with self.subTest(kind=kind):
                made = record(kind=kind, submitted=submitted)
                self.assertEqual(EXPECTED_FIELDS, set(made.values))
                self.assertEqual(used, made.values["used_message_id"])
                self.assertEqual(startup,
                                 made.values["is_startup_ego_only"])
                self.assertEqual(zoh, made.values["is_zoh"])
                self.assertEqual(drop_reason, made.values["drop_reason"])
                self.assertFalse(made.values["arrival_deadline_miss"])
                self.assertEqual(reorder_status,
                                 made.values["reorder_status"])
                self.assertEqual(duplicate_status,
                                 made.values["duplicate_status"])
        self.assertIsNone(record(kind="drop", submitted=submission(drop=True)).values[
            "effective_delay_ns"])

    def test_reordered_duplicate_and_late_ticks_are_explicit(self):
        scheduled = record(submitted=submission(reorder_ns=10,
                                                 duplicate=True))
        self.assertEqual("scheduled", scheduled.values["reorder_status"])
        self.assertEqual("scheduled", scheduled.values["duplicate_status"])
        delivered = record(
            submitted=submission(), outcome_status="accepted", reorder_ns=10)
        self.assertEqual("delivered", delivered.values["reorder_status"])
        discarded = record(
            kind="zoh", outcome_status="duplicate_discarded", reorder_ns=10)
        self.assertEqual("delivered", discarded.values["reorder_status"])
        self.assertEqual("discarded", discarded.values["duplicate_status"])
        late = record(kind="zoh", outcome_status="late_superseded")
        self.assertEqual("late_superseded", late.values["reorder_status"])

    def test_ordinary_prior_deadline_miss_is_reported_only_on_release_tick(self):
        current = submission(sequence=8, effective_delay_ns=100)
        prior = submission(sequence=7, effective_delay_ns=25)
        prior_outcome = DeliveryOutcome(
            prior.enqueued[0], "accepted", True,
        )
        selected = tick().streams["rsu-1"]
        released = ReplayTick(
            9, 150, {"rsu-1": selected}, (prior_outcome,), 3,
            ReplayCounters(1, 0, 0),
        )
        quiet = ReplayTick(
            10, 151, {"rsu-1": selected}, (), 3,
            ReplayCounters(1, 0, 0),
        )

        self.assertTrue(build_audit_record(
            released, current, "rsu-1", context()
        ).values["arrival_deadline_miss"])
        self.assertFalse(build_audit_record(
            quiet, current, "rsu-1", context()
        ).values["arrival_deadline_miss"])

    def test_tick_aggregates_prior_delivery_outcomes_with_current_submission(self):
        current = submission(effective_delay_ns=100)
        prior = submission(reorder_ns=10, duplicate=True, sequence=6)
        prior_outcome = DeliveryOutcome(
            prior.enqueued[1], "duplicate_discarded", True,
        )
        used = StreamSelection(
            stream_id="rsu-1", used_message_id="message-6",
            used_sequence_id=6, payload={}, source_frame=6,
            source_sim_time_ns=100, ready_time_ns=135, aoi_ns=50,
            zoh_hold_ns=9, is_zoh=True, is_startup_ego_only=False,
            is_expired=False, fusion_deadline_miss=False,
        )
        replay_tick = ReplayTick(
            9, 150, {"rsu-1": used}, (prior_outcome,), 3,
            ReplayCounters(1, 0, 0),
        )

        made = build_audit_record(replay_tick, current, "rsu-1", context())

        self.assertEqual("message-7", made.values["submitted_message_id"])
        self.assertEqual(100, made.values["effective_delay_ns"])
        self.assertEqual("message-6", made.values["used_message_id"])
        self.assertEqual(6, made.values["used_source_frame"])
        self.assertEqual(100, made.values["source_sim_time_ns"])
        self.assertTrue(made.values["is_zoh"])
        self.assertTrue(made.values["arrival_deadline_miss"])
        self.assertEqual("delivered", made.values["reorder_status"])
        self.assertEqual("discarded", made.values["duplicate_status"])

        no_delivery_tick = ReplayTick(
            10, 151, {"rsu-1": used}, (), 3, ReplayCounters(1, 0, 0),
        )
        no_delivery = build_audit_record(
            no_delivery_tick, current, "rsu-1", context()
        )
        self.assertFalse(no_delivery.values["arrival_deadline_miss"])
        self.assertEqual("none", no_delivery.values["reorder_status"])
        self.assertEqual("none", no_delivery.values["duplicate_status"])

        dropped_current = build_audit_record(
            replay_tick, submission(drop=True, sequence=8),
            "rsu-1", context(),
        )
        self.assertEqual("message-8", dropped_current.values[
            "submitted_message_id"])
        self.assertEqual("bernoulli", dropped_current.values["drop_reason"])
        self.assertIsNone(dropped_current.values["effective_delay_ns"])
        self.assertTrue(dropped_current.values["arrival_deadline_miss"])
        self.assertEqual("delivered", dropped_current.values["reorder_status"])
        self.assertEqual("discarded", dropped_current.values[
            "duplicate_status"])

    def test_records_and_context_are_frozen_and_defensively_copied(self):
        values = dict(record().values)
        made = DitlAuditRecord(values)
        values["run_id"] = "changed"
        self.assertEqual("run-1", made.values["run_id"])
        with self.assertRaises(TypeError):
            made.values["run_id"] = "changed"
        for model in (made, context()):
            with self.assertRaises(FrozenInstanceError):
                model.run_id = "changed"

    def test_invalid_context_record_and_build_inputs_are_rejected(self):
        bad_contexts = (
            dict(run_id="", route_id="route", policy_seed=1,
                 policy_config_hash="a" * 64, rng_version=RNG_VERSION),
            dict(run_id="run", route_id="route", policy_seed=True,
                 policy_config_hash="a" * 64, rng_version=RNG_VERSION),
            dict(run_id="run", route_id="route", policy_seed=1,
                 policy_config_hash="xyz", rng_version=RNG_VERSION),
        )
        for values in bad_contexts:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    AuditContext(**values)
        values = dict(record().values)
        for name, value in (
                ("queue_depth", True),
                ("consume_frame", -1),
                ("run_id", ""),
                ("arrival_deadline_miss", 1),
                ("aoi_ns", float("nan"))):
            changed = dict(values)
            changed[name] = value
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    DitlAuditRecord(changed)
        changed = dict(values)
        changed["extra"] = 1
        with self.assertRaises(ValueError):
            DitlAuditRecord(changed)
        with self.assertRaises(ValueError):
            build_audit_record(tick(), submission(), "missing", context())


class AuditWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "ditl_event_audit.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write_records(self, path, records, bound=1):
        writer = DitlAuditWriter(path, max_buffer_records=bound)
        for item in records:
            writer.append(item)
        writer.close()

    def test_identical_records_produce_canonical_byte_identical_jsonl(self):
        records = [record(frame=1), record(frame=2, kind="zoh")]
        path_a = self.root / "a.jsonl"
        path_b = self.root / "b.jsonl"
        self.write_records(path_a, records)
        self.write_records(path_b, records)
        self.assertEqual(path_a.read_bytes(), path_b.read_bytes())
        self.assertTrue(path_a.read_bytes().endswith(b"\n"))
        lines = path_a.read_text("ascii").splitlines()
        self.assertEqual(2, len(lines))
        self.assertEqual(dict(records[0].values), json.loads(lines[0]))
        expected = json.dumps(
            dict(records[0].values), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n"
        self.assertEqual(expected.encode("ascii"),
                         path_a.read_bytes().splitlines(True)[0])

    def test_buffer_is_bounded_and_close_flushes_and_fsyncs(self):
        writer = DitlAuditWriter(self.output, max_buffer_records=2)
        descriptor_flags = fcntl.fcntl(writer._fd, fcntl.F_GETFD)
        status_flags = fcntl.fcntl(writer._fd, fcntl.F_GETFL)
        self.assertTrue(descriptor_flags & fcntl.FD_CLOEXEC)
        self.assertTrue(status_flags & os.O_APPEND)
        with mock.patch("ditl_audit.os.fsync", wraps=os.fsync) as fsync:
            writer.append(record(frame=1))
            self.assertEqual(b"", self.output.read_bytes())
            self.assertEqual(1, len(writer._buffer))
            writer.append(record(frame=2))
            self.assertEqual(0, len(writer._buffer))
            writer.append(record(frame=3))
            writer.close()
            writer.close()
        self.assertEqual(3, len(self.output.read_text("ascii").splitlines()))
        fsync.assert_called_once()

    def test_existing_complete_file_is_appended_without_overwrite(self):
        self.output.write_bytes(b'{"prior":true}\n')
        self.write_records(self.output, [record()])
        content = self.output.read_bytes()
        self.assertTrue(content.startswith(b'{"prior":true}\n'))
        self.assertEqual(2, len(content.splitlines()))

    def test_exclusive_writer_atomically_rejects_existing_file_unchanged(self):
        prior = b'{"foreign":true}\n'
        self.output.write_bytes(prior)
        with self.assertRaisesRegex(DitlAuditError, str(self.output)):
            DitlAuditWriter(self.output, exclusive=True)
        self.assertEqual(prior, self.output.read_bytes())

        fresh = self.root / "fresh.jsonl"
        writer = DitlAuditWriter(fresh, exclusive=True)
        writer.append(record())
        writer.close()
        self.assertEqual(1, len(fresh.read_text("ascii").splitlines()))

        raced = self.root / "raced.jsonl"
        real_open = os.open
        injected = []

        def race_open(path, flags, mode=0o777, *, dir_fd=None):
            if (path == raced.name and flags & os.O_CREAT and
                    not injected):
                injected.append(True)
                foreign_fd = real_open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600, dir_fd=dir_fd,
                )
                try:
                    os.write(foreign_fd, prior)
                finally:
                    os.close(foreign_fd)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with mock.patch("ditl_audit.os.open", side_effect=race_open):
            with self.assertRaisesRegex(DitlAuditError, str(raced)):
                DitlAuditWriter(raced, exclusive=True)
        self.assertEqual(prior, raced.read_bytes())

    def test_existing_partial_file_is_rejected_unchanged(self):
        self.output.write_bytes(b'{"partial":true}')
        before = self.output.read_bytes()
        with self.assertRaisesRegex(DitlAuditError, str(self.output)):
            DitlAuditWriter(self.output)
        self.assertEqual(before, self.output.read_bytes())

    def test_new_file_uses_owner_only_mode_and_latency_csv_is_rejected(self):
        writer = DitlAuditWriter(self.output)
        writer.close()
        self.assertEqual(0o600, stat.S_IMODE(self.output.stat().st_mode))
        latency = self.root / "latency_align_audit.csv"
        latency.write_text("keep\n", encoding="ascii")
        with self.assertRaisesRegex(DitlAuditError, str(latency)):
            DitlAuditWriter(latency)
        self.assertEqual("keep\n", latency.read_text("ascii"))

    def test_final_and_parent_symlinks_and_nonregular_outputs_are_rejected(self):
        target = self.root / "target.jsonl"
        target.write_bytes(b"")
        final_link = self.root / "final.jsonl"
        final_link.symlink_to(target)
        with self.assertRaisesRegex(DitlAuditError, str(final_link)):
            DitlAuditWriter(final_link)

        real_parent = self.root / "real"
        real_parent.mkdir()
        parent_link = self.root / "linked"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(DitlAuditError, str(parent_link)):
            DitlAuditWriter(parent_link / "audit.jsonl")
        self.assertFalse((real_parent / "audit.jsonl").exists())

        directory = self.root / "directory.jsonl"
        directory.mkdir()
        with self.assertRaisesRegex(DitlAuditError, str(directory)):
            DitlAuditWriter(directory)

    def test_missing_parent_relative_escape_and_invalid_bound_are_rejected(self):
        cases = (
            self.root / "missing" / "audit.jsonl",
            Path("..") / "audit.jsonl",
        )
        for path in cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(DitlAuditError, "audit.jsonl"):
                    DitlAuditWriter(path)
        for bound in (True, 0, -1, 1.5):
            with self.subTest(bound=bound):
                with self.assertRaises((TypeError, ValueError)):
                    DitlAuditWriter(self.output, max_buffer_records=bound)

    def test_short_writes_are_retried_until_the_line_is_complete(self):
        writer = DitlAuditWriter(self.output)
        real_write = os.write
        calls = []

        def short_write(fd, data):
            calls.append(len(data))
            return real_write(fd, data[:3])

        with mock.patch("ditl_audit.os.write", side_effect=short_write):
            writer.append(record())
        writer.close()
        self.assertGreater(len(calls), 1)
        self.assertEqual(1, len(self.output.read_text("ascii").splitlines()))

    def test_write_error_is_surfaced_with_path_and_failure_is_stable(self):
        writer = DitlAuditWriter(self.output, max_buffer_records=1)
        with mock.patch.object(writer, "_write_all",
                               side_effect=OSError("disk full")):
            with self.assertRaisesRegex(DitlAuditError, str(self.output)) as first:
                writer.append(record())
        with self.assertRaisesRegex(DitlAuditError, "disk full") as second:
            writer.append(record(frame=2))
        self.assertEqual(str(first.exception), str(second.exception))
        with self.assertRaisesRegex(DitlAuditError, "disk full"):
            writer.close()
        with self.assertRaisesRegex(DitlAuditError, "disk full"):
            writer.close()

    def test_zero_progress_write_and_fsync_failures_close_descriptors(self):
        writer = DitlAuditWriter(self.output)
        with mock.patch("ditl_audit.os.write", return_value=0):
            with self.assertRaisesRegex(DitlAuditError, str(self.output)):
                writer.append(record())
        with self.assertRaises(DitlAuditError):
            writer.close()
        self.assertEqual(-1, writer._fd)

        second = self.root / "fsync.jsonl"
        writer = DitlAuditWriter(second, max_buffer_records=2)
        writer.append(record())
        with mock.patch("ditl_audit.os.fsync", side_effect=OSError("I/O")):
            with self.assertRaisesRegex(DitlAuditError, str(second)):
                writer.close()
        self.assertEqual(-1, writer._fd)
        with self.assertRaisesRegex(DitlAuditError, "I/O"):
            writer.close()

    def test_append_rejects_wrong_record_and_closed_writer(self):
        writer = DitlAuditWriter(self.output)
        with self.assertRaises(TypeError):
            writer.append({})
        writer.close()
        with self.assertRaisesRegex(DitlAuditError, str(self.output)):
            writer.append(record())


if __name__ == "__main__":
    unittest.main()
