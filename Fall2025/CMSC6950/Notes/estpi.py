from random import uniform
import argparse


def estimate_pi(n):
    '''
    Use Monte-Carlo with n points to estimate pi
    '''
    count_inside = 0
    for i in range(n):
        point = random_complex_point()
        if inside(point):
            count_inside += 1
    return 4 * count_inside / n


def inside(point):
    '''
    Is the complex number given by point inside the unit circle or not?
    '''
    return abs(point) <= 1


def random_complex_point():
    '''
    Generate a random complex number in the unit square in the complex domain
    '''
    return complex(uniform(0, 1), uniform(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python estpi.py',
                                     description='Monte-Carlo estimation of pi')  # noqa: E501
    parser.add_argument("--n", type=int, default=10,
                        help='number of points to use in estimate, default is 10')  # noqa: E501
    args = parser.parse_args()
    print(f"With {args.n} points, estimate of pi is {estimate_pi(args.n)}")
