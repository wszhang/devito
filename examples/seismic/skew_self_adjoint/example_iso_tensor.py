import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, norm)
from devito import VectorFunction, TensorFunction, NODE
from examples.seismic import RickerSource, TimeAxis

space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

shape = (1201, 1201, 601)
spacing = (10.0, 10.0, 10.0)
origin = tuple([0.0 for s in shape])
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

b = Function(name='b', grid=grid, space_order=space_order)
vel = Function(name='vel', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=vel.grid, space_order=space_order)

# b._data_with_outhalo[:] = 1.0
# vel._data_with_outhalo[:] = 1.5
# wOverQ._data_with_outhalo[:] = 1.0
b.data[:] = 1.0
vel.data[:] = 1.5
wOverQ.data[:] = 1.0

t0 = 0.0
t1 = 250.0
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
# src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src_coords[0, :] = [d * (s-1)//2 + 100 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=vel.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel**2 / b)

def grads(func):
    comps = [getattr(func, 'd%s' % d.name)(x0=d + d.spacing/2)
             for d in func.dimensions if d.is_Space]
    return VectorFunction(name='grad_%s' % func.name, space_order=func.space_order,
                          components=comps, grid=func.grid, staggered=(None, None, None))
            
def divs(func):
    return sum([getattr(func[i], 'd%s' % d.name)(x0=d - d.spacing/2)
                for i, d in enumerate(func.space_dimensions)])

P = VectorFunction(name="P", grid=grid, space_order=space_order, staggered=(None, None, None))

b_ii = [[b, 0, 0],
        [0, b, 0],
        [0, 0, b]]

B = TensorFunction(name="B", grid=grid, components=b_ii, diagonal=True)

eq_P = Eq(P, B * grads(p0))

update_p = t.spacing**2 * vel**2 / b * divs(P) + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

stencil_p = Eq(p0.forward, update_p)

dt = time_axis.step
spacing_map = vel.grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([eq_P, stencil_p, src_term],
              subs=spacing_map, name='OpExampleIsoTensor')

f = open("operator.iso_tensor.c", "w")
print(op, file=f)
f.close()

bx = 1; by = 1; # 7502
# bx = 16; by = 4; # 7742

op.apply(x0_blk0_size=bx, y0_blk0_size=by)

print("")
print("bx,by,norm; %3d %3d %12.6e" % (bx, by, norm(p0)))
