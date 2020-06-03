import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, norm)
from examples.seismic import RickerSource, TimeAxis

space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

shape = (1201, 1201, 601)
shape = (501, 501, 251)
spacing = (10.0, 10.0, 10.0)
origin = tuple([0.0 for s in shape])
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

b = Function(name='b', grid=grid, space_order=space_order)
vel = Function(name='vel', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=vel.grid, space_order=space_order)

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
src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=vel.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel**2 / b)


def g1(field):
    return field.dx(x0=x+x.spacing/2)


def g2(field):
    return field.dy(x0=y+y.spacing/2)


def g3(field):
    return field.dz(x0=z+z.spacing/2)


def g1_tilde(field):
    return field.dx(x0=x-x.spacing/2)


def g2_tilde(field):
    return field.dy(x0=y-y.spacing/2)


def g3_tilde(field):
    return field.dz(x0=z-z.spacing/2)


p_x = Function(name='p_x', grid=grid, space_order=space_order)
p_y = Function(name='p_y', grid=grid, space_order=space_order)
p_z = Function(name='p_z', grid=grid, space_order=space_order)

update_px = Eq(p_x, g1_tilde(b * g1(p0)))
update_py = Eq(p_y, g2_tilde(b * g2(p0)))
update_pz = Eq(p_z, g3_tilde(b * g3(p0)))

update_p0 = t.spacing**2 * vel**2 / b * \
    (p_x + p_y + p_z) + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

stencil_p0 = Eq(p0.forward, update_p0)

dt = time_axis.step
spacing_map = vel.grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([update_px, update_py, update_pz, stencil_p0, src_term],
              subs=spacing_map, name='OpExampleIsoFlatten')

print(op.args)
f = open("operator.iso_flatten.c", "w")
print(op, file=f)
f.close()

bx = 8
by = 8

# 7742
bx = 16
by = 4

op.apply(x0_blk0_size=bx, y0_blk0_size=by)

print("")
print("bx,by,norm; %3d %3d %12.6e" % (bx, by, norm(p0)))

print("")
print(time_axis)
print("nx,ny,nz; %5d %5d %5d" % (shape[0], shape[1], shape[2]))

f = open("data.iso_flatten.bin", "wb")
np.save(f, p0.data[1,:,:,:])
f.close()
