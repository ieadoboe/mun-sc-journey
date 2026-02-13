import pytest
import array_stats


@pytest.mark.parametrize("data_list, expected",
                         [([-2, -1, 0, 1, 2], 0),
                          ([1, 2, 3, 4], 2.5),
                          (range(1, 1000), 500),
                          ([0.5, 1.2, 3.8, 4.5], 2.5)])
def test_mean(data_list, expected):
    observed = array_stats.mean(data_list)
    assert observed == expected
