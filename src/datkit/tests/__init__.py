"""Tests, shipped inside the package so they can run in the target environment.

The cleanroom has pytest but no source checkout, so the suite travels in the
wheel and runs there with `pytest --pyargs datkit`. That verifies the upload
against the real pandas build, the real memory ceiling and the real data —
none of which CI can reproduce.
"""
