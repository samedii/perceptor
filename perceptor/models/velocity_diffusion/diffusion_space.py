

def encode(images):
    return images.mul(2).sub(1)


def decode(x):
    return x.add(1).div(2)
