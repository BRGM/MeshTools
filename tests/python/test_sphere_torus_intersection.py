import numpy as np
import MeshTools as MT
from MeshTools.CGALWrappers import sphere, torus

print(sphere(np.sqrt(3),0,0))
print(torus(1.5,0.5,0))
print(torus(-1.5,0.5,0))

r = 5/3
h = np.sqrt(2)/3

for theta in np.linspace(0, 2*np.pi):
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    print(np.abs(torus(x,h,z)-sphere(x,h,z)), np.abs(torus(x,-h,z)-sphere(x,-h,z)))