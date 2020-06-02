import socket
import numpy as np
from sympy import sqrt
from mpi4py import MPI

from devito import (Grid, Function, TimeFunction, Eq, Operator)
from examples.seismic import RickerSource, TimeAxis

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
t1 = 12.0
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


# Additional factorization by hand
b1mf = Function(name='b1mf', grid=grid, space_order=space_order)
b1m2e = Function(name='b1m2e', grid=grid, space_order=space_order)
b1mfe2 = Function(name='b1mfe2', grid=grid, space_order=space_order)
b1mfpfe2 = Function(name='b1mfpfe2', grid=grid, space_order=space_order)
bfes1me2 = Function(name='bfes1me2', grid=grid, space_order=space_order)

# Equations for additional factorization
eq_b1mf = Eq(b1mf, b * (1 - f))
eq_b1m2e = Eq(b1m2e, b * (1 + 2 * eps))
eq_b1mfe2 = Eq(b1mfe2, b * (1 - f * eta**2))
eq_b1mfpfe2 = Eq(b1mfpfe2, b * (1 - f + f * eta**2))
eq_bfes1me2 = Eq(bfes1me2, b * f * eta * sqrt(1 - eta**2))

# Time update equation for quasi-P state variable p
update_p = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b1m2e * g1(p0)) +
     g2_tilde(b1m2e * g2(p0)) +
     g3_tilde(b1mfe2 * g3(p0) + bfes1me2 * g3(m0))) + \
    (2 - t.spacing * wOverQ) * p0 + (t.spacing * wOverQ - 1) * p0.backward

# Time update equation for quasi-S state variable m
update_m = t.spacing**2 * vel**2 / b * \
    (g1_tilde(b1mf * g1(m0)) +
     g2_tilde(b1mf * g2(m0)) +
     g3_tilde(b1mfpfe2 * g3(m0) + bfes1me2 * g3(p0))) + \
    (2 - t.spacing * wOverQ) * m0 + (t.spacing * wOverQ - 1) * m0.backward

stencil_p = Eq(p0.forward, update_p)
stencil_m = Eq(m0.forward, update_m)

dt = time_axis.step
spacing_map = grid.spacing_map
spacing_map.update({t.spacing: dt})

op = Operator([eq_b1mf, eq_b1m2e, eq_b1mfe2, eq_b1mfpfe2, eq_bfes1me2,
               stencil_p, stencil_m, src_term],
              subs=spacing_map, name='OpExampleVti')

filename = "timing_vti.%s.txt" % (socket.gethostname())
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
        m0.data[:] = 0
        s = op.apply(x0_blk0_size=bx, y0_blk0_size=by)
        if rank == 0:
            gpointss = np.sum([v.gpointss for k, v in s.items()])
            # gpointss = np.max([v.gpointss for k, v in s.items()])
            print("bx,by,gpts/s; %3d %3d %10.6f" % (bx, by, gpointss))
            print("bx,by,gpts/s; %3d %3d %10.6f" % (bx, by, gpointss), file=f)
            f.flush()

f.close()