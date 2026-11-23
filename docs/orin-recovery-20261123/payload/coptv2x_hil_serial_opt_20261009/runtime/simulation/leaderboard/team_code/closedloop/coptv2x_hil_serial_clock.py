"""One non-overlapping HIL cycle with a simulation-time availability gate."""
import copy


class SerialHilClock:
    def __init__(self, client, simulation_config=None):
        if client.result_buffer.timing_mode != "serial_fixed_comm":
            raise ValueError("SerialHilClock requires serial_fixed_comm timing")
        config = {} if simulation_config is None else simulation_config
        conflicting = [key for key in (
            "rsu_fifo_delay_ms", "latency_inject_ms", "comm_latency",
            "latency_trace", "ditl_replay",
        ) if key in config]
        if config.get("latency_aware", False):
            conflicting.append("latency_aware")
        if conflicting:
            raise ValueError("serial_fixed_comm conflicts with %s" % conflicting)
        self.client = client
        self._pending = None

    @property
    def pending(self):
        return self._pending is not None

    def begin(self, trigger, collaboration_data, rsu_data):
        if self.pending:
            raise RuntimeError("serial HIL cycle already pending")
        # Source metadata must remain paired with this result at its later
        # fusion tick, while ego data is collected fresh by the normal agent.
        collaboration = copy.deepcopy(collaboration_data)
        rsu = copy.deepcopy(rsu_data)
        decision = self.client.publish_trigger(trigger)
        if not decision.accepted:
            raise RuntimeError("HIL trigger rejected: %s" % decision.reason)
        self.client.wait_for_result(trigger)
        self._pending = (trigger, collaboration, rsu)

    def consume_if_ready(self, current_frame, current_sim_time_ns):
        if not self.pending:
            return None
        trigger, collaboration, rsu = self._pending
        selected = self.client.select_for_fusion(current_frame, current_sim_time_ns, trigger)
        if selected.result is None:
            return None
        self._pending = None
        return selected, collaboration, rsu
