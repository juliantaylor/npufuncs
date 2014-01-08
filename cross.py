from numpy import array, asarray
import numpy as np
from npufunc import fma, fms
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa, axisb, axisc=(axis,)*3
    a = asarray(a).swapaxes(axisa, 0)
    b = asarray(b).swapaxes(axisb, 0)
    msg = "incompatible dimensions for cross product\n"\
          "(dimension must be 2 or 3)"
    if (a.shape[0] not in [2, 3]) or (b.shape[0] not in [2, 3]): 
        raise ValueError(msg)
    if a.shape[0] == 2:
        if (b.shape[0] == 2):
            cp = a[0]*b[1] - a[1]*b[0]
            if cp.ndim == 0:
                return cp
            else:
                return cp.swapaxes(0, axisc)
        else:
            x = a[1]*b[2]
            y = -a[0]*b[2]
            z = a[0]*b[1] - a[1]*b[0]
            cp = array([x, y, z])
    elif a.shape[0] == 3:
        if (b.shape[0] == 3):
            import numpy as np
            a0, a1, a2, b0, b1, b2 = np.broadcast_arrays(a[0], a[1], a[2], b[0], b[1], b[2])
            cp = np.empty((3,) + a1.shape)
            np.multiply(a1, b2, out=cp[0])
            tmp = a[2]*b[1]
            cp[0] -= tmp
            np.multiply(a2, b0, out=cp[1])
            np.multiply(a0, b2, out=tmp)
            cp[1] -= tmp
            np.multiply(a0, b1, out=cp[2])
            np.multiply(a1, b0, out=tmp)
            cp[2] -= tmp
            #cp = array([x, y, z])
        else:
            x = -a[2]*b[1]
            y = a[2]*b[0]
            z = a[0]*b[1] - a[1]*b[0]
            cp = array([x, y, z])
    if cp.ndim == 1:
        return cp
    else:
        return cp.swapaxes(0, axisc)

def cross2(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa, axisb, axisc=(axis,)*3
    a = asarray(a).swapaxes(axisa, 0)
    b = asarray(b).swapaxes(axisb, 0)
    msg = "incompatible dimensions for cross product\n"\
          "(dimension must be 2 or 3)"
    if (a.shape[0] not in [2, 3]) or (b.shape[0] not in [2, 3]): 
        raise ValueError(msg)
    if a.shape[0] == 2:
        if (b.shape[0] == 2):
            cp = a[0]*b[1] - a[1]*b[0]
            if cp.ndim == 0:
                return cp
            else:
                return cp.swapaxes(0, axisc)
        else:
            x = a[1]*b[2]
            y = -a[0]*b[2]
            z = a[0]*b[1] - a[1]*b[0]
            cp = array([x, y, z])
    elif a.shape[0] == 3:
        if (b.shape[0] == 3):
            import numpy as np
            a0, a1, a2, b0, b1, b2 = np.broadcast_arrays(a[0], a[1], a[2], b[0], b[1], b[2])
            cp = np.empty((3,) + a1.shape)
            np.negative(a[2], out=cp[0])
            cp[0] *= b[1]
            fma(a1, b2, cp[0], out=cp[0])
            np.negative(a0, out=cp[1])
            cp[1] *= b2
            fma(a2, b0, cp[1], out=cp[1])
            np.negative(a1, out=cp[2])
            cp[2] *= b0
            fma(a0, b1, cp[2], out=cp[2])
            #cp = array([x, y, z])
        else:
            x = -a[2]*b[1]
            y = a[2]*b[0]
            z = a[0]*b[1] - a[1]*b[0]
            cp = array([x, y, z])
    if cp.ndim == 1:
        return cp
    else:
        return cp.swapaxes(0, axisc)


def cross3(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa, axisb, axisc=(axis,)*3
    a = asarray(a).swapaxes(axisa, 0)
    b = asarray(b).swapaxes(axisb, 0)
    msg = "incompatible dimensions for cross product\n"\
          "(dimension must be 2 or 3)"
    if (a.shape[0] not in [2, 3]) or (b.shape[0] not in [2, 3]): 
        raise ValueError(msg)
    if a.shape[0] == 2:
        if (b.shape[0] == 2):
            cp = a[0]*b[1] - a[1]*b[0]
            if cp.ndim == 0:
                return cp
            else:
                return cp.swapaxes(0, axisc)
        else:
            x = a[1]*b[2]
            y = -a[0]*b[2]
            z = a[0]*b[1] - a[1]*b[0]
            cp = array([x, y, z])
    elif a.shape[0] == 3:
        if (b.shape[0] == 3):
            import numpy as np
            a0, a1, a2, b0, b1, b2 = np.broadcast_arrays(a[0], a[1], a[2], b[0], b[1], b[2])
            cp = np.empty((3,) + a1.shape)
            np.multiply(a[2], b[1], out=cp[0])
            fms(a1, b2, cp[0], out=cp[0])
            np.multiply(a0, b2, out=cp[1])
            fms(a2, b0, cp[1], out=cp[1])
            np.multiply(a1, b0, out=cp[2])
            fms(a0, b1, cp[2], out=cp[2])
            #cp = array([x, y, z])
        else:
            x = -a[2]*b[1]
            y = a[2]*b[0]
            z = a[0]*b[1] - a[1]*b[0]
            cp = array([x, y, z])
    if cp.ndim == 1:
        return cp
    else:
        return cp.swapaxes(0, axisc)

