import math

def iexp(n):
    return complex(math.cos(n), math.sin(n))

def dft(xs):
    
    n = len(xs)
    return [sum((xs[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]

X = dft([1,2,3,4])