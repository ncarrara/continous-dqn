from multiprocessing import Pool
import numpy as np
def f(x):
    y= 0
    for i in x:
        y+=i
    return y

with Pool(None) as p:
    y = p.map(f,[np.linspace(0,n,100) for n in range(10000)])

print(y)
