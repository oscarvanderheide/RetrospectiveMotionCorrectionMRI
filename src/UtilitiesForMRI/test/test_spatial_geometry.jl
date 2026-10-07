using UtilitiesForMRI, LinearAlgebra, Test

# Cartesian domain
n = (256, 256, 256)
fov = (1.0, 2.0, 2.1)
o = (0.8, 1.0, 1.6)
X = spatial_geometry(fov, n; origin=o)

# Features
field_of_view(X)
h = spacing(X)
o = origin(X; wrt_center=true)
x, y, z = coord(X)

# k-space coordinates
kx, ky, kz = k_coord(X; mesh=true)