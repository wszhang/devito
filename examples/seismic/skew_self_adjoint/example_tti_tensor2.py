import numpy as np
import sympy
from sympy import sqrt, sin, cos, Matrix, pprint

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
eps = Function(name='eps', grid=vel.grid, space_order=space_order)
eta = Function(name='eta', grid=vel.grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=vel.grid, space_order=space_order)
theta = Function(name='theta', grid=grid, space_order=space_order)
phi = Function(name='phi', grid=grid, space_order=space_order)

b._data_with_outhalo[:] = 1.0
f._data_with_outhalo[:] = 0.84
vel._data_with_outhalo[:] = 1.5
eps._data_with_outhalo[:] = 0.2
eta._data_with_outhalo[:] = 0.4
wOverQ._data_with_outhalo[:] = 0.0
theta._data_with_outhalo[:] = np.pi / 3
phi._data_with_outhalo[:] = np.pi / 6

t0 = 0.0
t1 = 250.0
dt = 1.0
time_axis = TimeAxis(start=t0, stop=t1, step=dt)
t = grid.time_dim

p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = p0.dimensions

src_coords = np.empty((1, len(shape)), dtype=dtype)
src_coords[0, :] = [d * (s-1)//2 for d, s in zip(spacing, shape)]
src = RickerSource(name='src', grid=vel.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]
src_term = src.inject(field=p0.forward, expr=src * t.spacing**2 * vel**2 / b)

a1_comp = [[b * (1 + 2 * eps), 0], 
           [0, b * (1 - f)]]

a2_comp = [[b * (1 - f * eta**2), b * (f * eta * sqrt(1 - eta**2))], 
           [b * (f * eta * sqrt(1-eta**2)), b * (1 - f + f * eta**2)]]

pm_comp = [[p0], [m0]]

A1 = TensorFunction(name="A1", grid=grid, components=a1_comp,
                    space_order=space_order, diagonal=True)
A2 = TensorFunction(name="A2", grid=grid, components=a2_comp,
                    space_order=space_order, diagonal=True)
PM = VectorFunction(name="PM", grid=grid, components=pm_comp,
                    space_order=space_order, staggered=(None, None))

print("")
print("A1[0,0]; ", A1[0,0])
print("A1[0,1]; ", A1[0,1])
print("A1[1,0]; ", A1[1,0])
print("A1[1,1]; ", A1[1,1])
print("")
print("A2[0,0]; ", A2[0,0])
print("A2[0,1]; ", A2[0,1])
print("A2[1,0]; ", A2[1,0])
print("A2[1,1]; ", A2[1,1])
print("")
print("PM[0]; ", PM[0])
print("PM[1]; ", PM[1])

def my_d(func, d):
    comps = [getattr(func[0], 'd%s' % d.name)(x0=d + d.spacing/2),
             getattr(func[1], 'd%s' % d.name)(x0=d + d.spacing/2)]
    return VectorFunction(name='my_d', space_order=func.space_order,
                          components=comps, grid=func.grid, staggered=(None, None))
            
def my_d_tilde(func, d):
    return sum([getattr(func[0], 'd%s' % d.name)(x0=d - d.spacing/2),
                getattr(func[1], 'd%s' % d.name)(x0=d - d.spacing/2)])

def my_g3(func, phi, theta):
    comps = [sin(theta) * cos(phi) * func[0].dx(x0=x+x.spacing/2) +
             sin(theta) * sin(phi) * func[0].dy(x0=y+y.spacing/2) +
             cos(theta) * func[0].dz(x0=z+z.spacing/2),
             sin(theta) * cos(phi) * func[1].dx(x0=x+x.spacing/2) +
             sin(theta) * sin(phi) * func[1].dy(x0=y+y.spacing/2) +
             cos(theta) * func[1].dz(x0=z+z.spacing/2)]
    return VectorFunction(name='my_g3', space_order=func.space_order,
                          components=comps, grid=func.grid, staggered=(None, None))
            
def my_g3_tilde(func, phi, theta):
    return sum([(sin(theta) * cos(phi) * func[0]).dx(x0=x-x.spacing/2) +
                (sin(theta) * sin(phi) * func[0]).dy(x0=y-y.spacing/2) +
                (cos(theta) * func[0]).dz(x0=z-z.spacing/2),
                (sin(theta) * cos(phi) * func[1]).dx(x0=x-x.spacing/2) +
                (sin(theta) * sin(phi) * func[1]).dy(x0=y-y.spacing/2) +
                (cos(theta) * func[1]).dz(x0=z-z.spacing/2)])


# # Time update equation for quasi-P state variable p
eq = (my_d_tilde((A1 * my_d(PM, x)), x) + 
      my_d_tilde((A1 * my_d(PM, y)), y) + 
      my_d_tilde((A1 * my_d(PM, z)), z) + 
      my_g3_tilde(((A2 - A1) * my_g3(PM, phi, theta)), phi, theta))

print(eq)
update_p = t.spacing**2 * vel**2 / b * eq + \
    (2 - t.spacing * wOverQ) * p0 + \
    (t.spacing * wOverQ - 1) * p0.backward

# stencil_p = Eq(p0.forward, update_p)

# dt = time_axis.step
# spacing_map = vel.grid.spacing_map
# spacing_map.update({t.spacing: dt})

# op = Operator([eq_PI, eq_MI, stencil_p, stencil_m, src_term],
#               subs=spacing_map, name='OpExampleTtiTensor2')

# f = open("operator.tti_tensor2.c", "w")
# print(op, file=f)
# f.close()

# bx = 8
# by = 8
# op.apply(x0_blk0_size=bx, y0_blk0_size=by)

print("")
print("norm; %12.6e %12.6e" % (norm(p0), norm(m0)))
