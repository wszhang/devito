import numpy as np
from sympy import sqrt

from devito import (Grid, Function, TimeFunction, Eq, Operator, norm)
from examples.seismic import RickerSource, TimeAxis

space_order = 8
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 250.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

shape = (1201, 1201, 601)
spacing = (10.0, 10.0, 10.0)
origin = tuple([0.0 for s in shape])
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

b = Function(name='b', grid=grid, space_order=space_order)
f = Function(name='f', grid=grid, space_order=space_order)
vel = Function(name='vel', grid=grid, space_order=space_order)
eps = Function(name='eps', grid=grid, space_order=space_order)
eta = Function(name='eta', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=grid, space_order=space_order)

b.data[:] = 1.0
f.data[:] = 0.84
vel.data[:] = 1.5
eps.data[:] = 0.2
eta.data[:] = 0.4
wOverQ.data[:] = 1.0

t0 = 0.0
t1 = 250.0
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=grid, f0=fpeak, npoint=1, time_range=time_axis)
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


# Time update equation for quasi-P state variable p
update_p_nl = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b * (1 + 2 * eps) * g1(p0)) +
     g2_tilde(b * (1 + 2 * eps) * g2(p0)) +
     g3_tilde(b * (1 - f * eta**2) * g3(p0) +
              b * f * eta * sqrt(1 - eta**2) * g3(m0))) + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

# Time update equation for quasi-S state variable m
update_m_nl = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b * (1 - f) * g1(m0)) +
     g2_tilde(b * (1 - f) * g2(m0)) +
     g3_tilde(b * (1 - f + f * eta**2) * g3(m0) +
              b * f * eta * sqrt(1 - eta**2) * g3(p0))) + \
    (2 - t.spacing * wOverQ) * m0 + \
    (t.spacing * wOverQ - 1) * m0.backward

stencil_p_nl = Eq(p0.forward, update_p_nl)
stencil_m_nl = Eq(m0.forward, update_m_nl)

dt = time_axis.step
spacing_map = grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([stencil_p_nl, stencil_m_nl, src_term],
              subs=spacing_map, name='OpExampleVti')

f = open("operator.vti.c", "w")
print(op, file=f)
f.close()

# 7502
bx = 10
by = 6

# 7742
# bx = 16
# by = 4

op.apply(x0_blk0_size=bx, y0_blk0_size=by)

print("")
print("bx,by,norm; %3d %3d %12.6e %12.6e" % (bx, by, norm(p0), norm(m0)))
