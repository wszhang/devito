import numpy as np
from sympy import sqrt, sin, cos

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
theta = Function(name='theta', grid=grid, space_order=space_order)
phi = Function(name='phi', grid=grid, space_order=space_order)

b.data[:] = 1.0
f.data[:] = 0.84
vel.data[:] = 1.5
eps.data[:] = 0.2
eta.data[:] = 0.4
theta.data[:] = np.pi / 3
phi.data[:] = np.pi / 6
wOverQ.data[:] = 1.0

t0 = 0.0
t1 = 250.0
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
# src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src_coords[0, :] = [d * (s-1)//2 + 100 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel**2 / b)


def my_d(func, d):
    return getattr(func, 'd%s' % d.name)(x0=d + d.spacing/2)


def my_d_tilde(func, d):
    return getattr(func, 'd%s' % d.name)(x0=d - d.spacing/2)


def g3(field, phi, theta):
    return (sin(theta) * cos(phi) * field.dx(x0=x+x.spacing/2) +
            sin(theta) * sin(phi) * field.dy(x0=y+y.spacing/2) +
            cos(theta) * field.dz(x0=z+z.spacing/2))


def g3_tilde(field, phi, theta):
    return ((sin(theta) * cos(phi) * field).dx(x0=x-x.spacing/2) +
            (sin(theta) * sin(phi) * field).dy(x0=y-y.spacing/2) +
            (cos(theta) * field).dz(x0=z-z.spacing/2))


# Functions for additional factorization
b1p2e = Function(name='b1p2e', grid=grid, space_order=space_order)
b1mf = Function(name='b1mf', grid=grid, space_order=space_order)
b2epfa2 = Function(name='b2epfa2', grid=grid, space_order=space_order)
bfes1ma2 = Function(name='bfes1ma2', grid=grid, space_order=space_order)
bfa2 = Function(name='bfa2', grid=grid, space_order=space_order)

# Equations for additional factorization
eq_b1p2e = Eq(b1p2e, b * (1 + 2 * eps))
eq_b1mf = Eq(b1mf, b * (1 - f))
eq_b2epfa2 = Eq(b2epfa2, b * (2 * eps + f * eta**2))
eq_bfes1ma2 = Eq(bfes1ma2, b * f * eta * sqrt(1 - eta**2))
eq_bfa2 = Eq(bfa2, b * f * eta**2)

# Time update equation for quasi-P state variable p
update_p = t.spacing**2 * vel**2 / b * \
    (my_d_tilde(b1p2e * my_d(p0, x), x) +
     my_d_tilde(b1p2e * my_d(p0, y), y) +
     my_d_tilde(b1p2e * my_d(p0, z), z) +
     g3_tilde(- b2epfa2 * g3(p0, phi, theta) +
              bfes1ma2 * g3(m0, phi, theta), phi, theta)) + \
    (2 - t.spacing * wOverQ) * p0 + (t.spacing * wOverQ - 1) * p0.backward

# Time update equation for quasi-S state variable m
update_m = t.spacing**2 * vel**2 / b * \
    (my_d_tilde(b1mf * my_d(m0, x), x) +
     my_d_tilde(b1mf * my_d(m0, y), y) +
     my_d_tilde(b1mf * my_d(m0, z), z) +
     g3_tilde(+ bfes1ma2 * g3(p0, phi, theta) + 
              bfa2 * g3(m0, phi, theta),  phi, theta)) + \
    (2 - t.spacing * wOverQ) * m0 + (t.spacing * wOverQ - 1) * m0.backward

"""
update_p = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b * (1 + 2 * eps) * g1(p0, phi, theta), phi, theta) +
     g2_tilde(b * (1 + 2 * eps) * g2(p0, phi, theta), phi, theta) +
     g3_tilde(b * (1 - f * eta**2) * g3(p0, phi, theta) +
              b * f * eta * sqrt(1 - eta**2) * g3(m0, phi, theta), phi, theta)) + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

# Time update equation for quasi-S state variable m
update_m = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b * (1 - f) * g1(m0, phi, theta), phi, theta) +
     g2_tilde(b * (1 - f) * g2(m0, phi, theta), phi, theta) +
     g3_tilde(b * (1 - f + f * eta**2) * g3(m0, phi, theta) +
              b * f * eta * sqrt(1 - eta**2) * g3(p0, phi, theta), phi, theta)) + \
    (2 - t.spacing * wOverQ) * m0 + \
    (t.spacing * wOverQ - 1) * m0.backward
"""

stencil_p = Eq(p0.forward, update_p)
stencil_m = Eq(m0.forward, update_m)

dt = time_axis.step
spacing_map = grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([eq_b1p2e, eq_b1mf, eq_b2epfa2, eq_bfes1ma2, eq_bfa2,
               stencil_p, stencil_m, src_term],
              subs=spacing_map, name='OpExampleTtiFact1',
              opt=('advanced', {'min-storage': True}))

f = open("operator.tti_fact1.c", "w")
print(op, file=f)
f.close()

bx = 14; by = 7; # 7502
# bx = 20; by = 3; # 7742

op.apply(x0_blk0_size=bx, y0_blk0_size=by)

print("")
print("bx,by,norm; %3d %3d %12.6e %12.6e" % (bx, by, norm(p0), norm(m0)))
