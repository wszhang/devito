import socket
import numpy as np
import sympy 
from sympy import sqrt
from mpi4py import MPI

from devito import (Grid, Function, TimeFunction, Eq, Operator, div, grad, diag)
from devito import VectorFunction, TensorFunction, NODE
from examples.seismic import RickerSource, TimeAxis


def grads(func, side="left"):
    shift = 1 if side == "right" else -1
    print(func, shift)
    comps = [getattr(func, 'd%s' % d.name)(x0=d + shift * d.spacing/2)
             for d in func.dimensions if d.is_Space]
    return VectorFunction(name='grad_%s' % func.name, space_order=func.space_order,
                          components=comps, grid=func.grid, staggered=(None, None, None))
            
def divs(func, side="left"):
    shift = 1 if side == "right" else -1
    print(func, shift)
    return sum([getattr(func[i], 'd%s' % d.name)(x0=d + shift * d.spacing/2)
                for i, d in enumerate(func.space_dimensions)])


space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 20.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

shape = (1201, 1201, 601)
spacing = (10.0, 10.0, 10.0)
origin = tuple([0.0 for s in shape])
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

b = Function(name='b', grid=grid, space_order=space_order)
vel = Function(name='vel', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=grid, space_order=space_order)

b.data[:] = 1.0
vel.data[:] = 1.5
wOverQ.data[:] = 1.0

t0 = 0.0
t1 = 12.0
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel**2 / b)

# Vector for gradients
P_I = VectorFunction(name="P_I", grid=grid, space_order=space_order, staggered=(None, None, None))

# Diagonal matrix
b_ii = [[b, 0, 0],
        [0, b, 0],
        [0, 0, b]]

B = TensorFunction(name="B", grid=grid, components=b_ii, diagonal=True)

# P_I
eq_PI = Eq(P_I, B * grads(p0, side="right"))

# Time update equation for quasi-P state variable p
update_p = t.spacing**2 * vel**2 / b * divs(P_I, side="left") + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

stencil_p = Eq(p0.forward, update_p)

dt = time_axis.step
spacing_map = grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([eq_PI, stencil_p, src_term],
              subs=spacing_map, name='OpExampleIsoTensor')

filename = "timing_iso_tensor.%s.txt" % (socket.gethostname())
print("filename; ", filename)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

bx1 = 0
bx2 = 64
dbx = 2
by1 = 0
by2 = 64
dby = 2

f = open(filename, "w")

for bx in range(bx2, bx1, -dbx):
    for by in range(by2, by1, -dby):
        p0.data[:] = 0
        s = op.apply(x0_blk0_size=bx, y0_blk0_size=by)
        if rank == 0:
            gpointss = np.sum([v.gpointss for k, v in s.items()])
            # gpointss = np.max([v.gpointss for k, v in s.items()])
            print("bx,by,gpts/s; %3d %3d %10.6f" % (bx, by, gpointss))
            print("bx,by,gpts/s; %3d %3d %10.6f" % (bx, by, gpointss), file=f)
            f.flush()

f.close()
