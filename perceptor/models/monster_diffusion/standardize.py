def encode(x):
    return x.mul(2).sub(1)


def decode(x):
    return x.add(1).div(2)
