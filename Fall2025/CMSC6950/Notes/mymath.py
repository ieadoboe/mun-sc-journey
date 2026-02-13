import math as m


def sinc2d(x, y):
    if x == 0 and y == 0:
        return 1.0
    elif x == 0:
        return m.sin(y)/y
    elif y == 0:
        return m.sin(x)/x
    else:
        return (m.sin(x)/x)*(m.sin(y)/y)


def addone(x):
    return x+1


def double(x):
    return 2*x


def addone_then_double(x):
    return double(addone(x))
