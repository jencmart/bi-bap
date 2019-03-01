import cppimport
import numpy as np

code = cppimport.imp("code")

if __name__ == '__main__':
    xs = np.random.rand(10)
    print(xs)
    print(code.square(xs))

    ys = range(10)
    print(code.square(ys))
