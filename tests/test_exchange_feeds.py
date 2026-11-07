"""
Tests for exchange_feeds.run_with_reconnect, the reconnect loop every WebSocket feed (trades,
liquidations and level-2 depth) runs inside. No sockets: the WebSocketApp is a stand-in whose
run_forever() just advances a fake clock by how long that "connection" lasted.
"""
from types import SimpleNamespace

import exchange_feeds


class _Stop:
    """A stop_event stand-in that records every wait (the backoff) and stops after N connections."""

    def __init__(self, connections):
        self.remaining = connections
        self.waits = []

    def is_set(self):
        return self.remaining <= 0

    def wait(self, seconds):
        self.waits.append(seconds)
        return self.is_set()


class _Sink:
    def __init__(self):
        self.states = []

    def set_status(self, name, state):
        self.states.append(state)


def _run(monkeypatch, lifetimes):
    clock = {"t": 0.0}
    slept = []
    monkeypatch.setattr(exchange_feeds, "time", SimpleNamespace(
        time=lambda: clock["t"], sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s))))
    stop = _Stop(len(lifetimes))
    queue = list(lifetimes)

    def create_app():
        def run_forever(**kwargs):
            clock["t"] += queue.pop(0)
            stop.remaining -= 1
        return SimpleNamespace(run_forever=run_forever)

    exchange_feeds.run_with_reconnect("x", create_app, stop, _Sink(), {})
    return stop.waits + slept


def test_backoff_resets_after_a_connection_that_stayed_up(monkeypatch):
    """
    The backoff doubled on every disconnect for the life of the process and never came back down,
    so after a handful of ordinary drops (exchanges routinely cycle connections) every reconnect
    cost the full 30s of missing data -- long enough to drop a venue from the book aggregate.
    A connection that held for a while proves the path is healthy; the next drop starts at 1s.
    """
    # three instant failures, then an hour up, then two more instant failures
    waits = _run(monkeypatch, [0.1, 0.1, 0.1, 3600.0, 0.1, 0.1, 0.1])
    assert waits == [1, 2, 4, 1, 2, 4]


def test_backoff_still_grows_and_caps_while_connections_keep_failing(monkeypatch):
    assert _run(monkeypatch, [0.1] * 8) == [1, 2, 4, 8, 16, 30, 30]


def test_backoff_waits_on_the_stop_event_so_stop_is_not_delayed_up_to_30s(monkeypatch):
    # time.sleep(backoff) ignored stop(); waiting on the event returns as soon as it's set.
    clock = {"t": 0.0}
    monkeypatch.setattr(exchange_feeds, "time", SimpleNamespace(
        time=lambda: clock["t"], sleep=lambda s: (_ for _ in ()).throw(AssertionError("time.sleep used"))))
    stop = _Stop(2)
    exchange_feeds.run_with_reconnect(
        "x", lambda: SimpleNamespace(run_forever=lambda **k: setattr(stop, "remaining", stop.remaining - 1)),
        stop, _Sink(), {})
    assert stop.waits == [1]
