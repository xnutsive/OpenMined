import numpy as np
import pytest
from syft.syft import FloatTensor, SyftController


def pytest_namespace():
    return {
        'sc': None
    }


@pytest.yield_fixture(autouse=True)
def setup_controller():
    pytest.sc = SyftController()


def test_abs():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    b = a.abs()

    np.testing.assert_array_equal(expected, b.to_numpy())
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_abs_():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    a.abs_()

    # a does change when inlined
    np.testing.assert_array_equal(expected, a.to_numpy())


def test_trace():
    data = np.random.randn(17, 17).astype('float')
    expected = data.trace()

    a = pytest.sc.FloatTensor(data)
    actual = a.trace()
    np.testing.assert_almost_equal(actual, expected, decimal=7)

    a = a.gpu()
    actual = a.trace()
    np.testing.assert_almost_equal(actual, expected, decimal=5)
