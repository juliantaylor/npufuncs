import npfma
import numpy as np

a = np.arange(5.)
b = np.arange(5.)
c = np.arange(5.)

print npfma.fma(a, b, c)

a = np.arange(50000.)
b = np.arange(50000.)
c = np.arange(50000.)
print npfma.fms(a, b, c)


