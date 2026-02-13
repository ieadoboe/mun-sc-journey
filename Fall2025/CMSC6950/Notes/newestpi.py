import argparse
import numpy as np


def np_points(n):
    return np.random.uniform(size=n) + np.random.uniform(size=n)*1j


def np_inside(points):
    return np.abs(points) <= 1


def estimate_pi(n):
    '''
    Use Monte-Carlo with n points to estimate pi
    '''
    count_inside = 0
    points = np_points(n)
    in_out = np_inside(points)
    count_inside = np.sum(in_out)
    return 4 * count_inside / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python estpi.py',
                                     description='Monte-Carlo estimation of pi')  # noqa: E501
    parser.add_argument("--n", type=int, default=10,
                        help='number of points to use in estimate, default is 10')  # noqa: E501
    args = parser.parse_args()
    print(f"With {args.n} points, estimate of pi is {estimate_pi(args.n)}")
