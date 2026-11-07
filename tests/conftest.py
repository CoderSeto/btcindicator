"""Makes the project modules importable from the tests directory, and keeps the suite offline."""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LiveNetworkInTest(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_live_http(monkeypatch):
    """
    Any real HTTP request from a test fails at once. Tests mock the calls they
    mean to make (monkeypatching `requests.get` or a module's fetch function
    overrides this); anything else reaching the network is a test bug -- and in
    a sandbox whose proxy refuses the host, a silent slow fallback rather than
    an error. The dashboard's ticker read (data_fetch.latest_price) is the
    case that prompted this: unstubbed, every page test called Coinbase.
    """
    def refuse(self, method, url, *a, **k):
        raise LiveNetworkInTest(f"test tried a live {method} {url}")
    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
