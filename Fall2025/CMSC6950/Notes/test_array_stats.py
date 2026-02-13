import array_stats


def test_zero():
    data_list = [-2, -1, 0, 1, 2]
    observed = array_stats.mean(data_list)
    expected = 0
    assert observed == expected


def test_float_value():
    data_list = [1, 2, 3, 4]
    observed = array_stats.mean(data_list)
    expected = 2.5
    assert observed == expected


def test_big():
    data_list = range(1, 1000)
    observed = array_stats.mean(data_list)
    expected = 500
    assert observed == expected


def test_float_array():
    data_list = [0.5, 1.2, 3.8, 4.5]
    observed = array_stats.mean(data_list)
    expected = 2.5
    assert observed == expected
