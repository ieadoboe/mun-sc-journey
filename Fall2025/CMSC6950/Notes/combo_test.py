import pytest


@pytest.mark.parametrize("x", [0, pytest.param(1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("y", [2, 3])
def test_sum(x, y):
    assert x+y == y

@pytest.fixture
def my_input():
    return 0

@pytest.mark.parametrize("y", [2, 3])
def test_sum2(my_input, y):
    assert my_input + y == y