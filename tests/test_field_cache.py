"""Tests for FieldCache LRU eviction, interpolation, and guess logic."""
from __future__ import annotations

import numpy as np
import pytest

from src.simulator.field_cache import FieldCache


def _make_value(phi_val: float = 0.0):
    phi = np.array([phi_val, phi_val + 1.0])
    E = np.array([-1.0, 1.0])
    Pe = np.array([0.1, 0.2])
    Ph = np.array([0.3, 0.4])
    return phi, E, Pe, Ph, 0.0, 1.0


def test_put_and_get():
    cache = FieldCache(maxlen=5)
    val = _make_value(10.0)
    cache.put(1.0, val)
    assert len(cache) == 1
    result = cache.get(1.0)
    assert result is not None
    np.testing.assert_array_equal(result[0], val[0])


def test_get_missing():
    cache = FieldCache()
    assert cache.get(999.0) is None


def test_lru_eviction():
    cache = FieldCache(maxlen=3)
    for i in range(5):
        cache.put(float(i), _make_value(float(i)))
    assert len(cache) == 3
    assert cache.get(0.0) is None
    assert cache.get(1.0) is None
    assert cache.get(2.0) is not None
    assert cache.get(3.0) is not None
    assert cache.get(4.0) is not None


def test_lru_access_refreshes():
    cache = FieldCache(maxlen=3)
    for i in range(3):
        cache.put(float(i), _make_value(float(i)))
    cache.get(0.0)
    cache.put(3.0, _make_value(3.0))
    assert cache.get(0.0) is not None
    assert cache.get(1.0) is None


def test_interpolate_guess_close():
    cache = FieldCache()
    cache.put(10.0, _make_value(10.0))
    cache.put(20.0, _make_value(20.0))
    guess = cache.interpolate_guess(15.0)
    assert guess is not None
    np.testing.assert_array_equal(guess, cache.get(10.0)[0])


def test_interpolate_guess_too_far():
    cache = FieldCache()
    cache.put(0.0, _make_value())
    assert cache.interpolate_guess(100.0) is None


def test_interpolate_guess_empty():
    cache = FieldCache()
    assert cache.interpolate_guess(5.0) is None


def test_clear():
    cache = FieldCache()
    cache.put(1.0, _make_value())
    cache.clear()
    assert len(cache) == 0
    assert cache.get(1.0) is None


def test_vkey_rounding():
    cache = FieldCache()
    cache.put(1.0000001, _make_value(42.0))
    result = cache.get(1.0000002)
    assert result is not None
    np.testing.assert_array_equal(result[0], [42.0, 43.0])
