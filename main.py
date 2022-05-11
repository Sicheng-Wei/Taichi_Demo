from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(-1, (0.5, 0.5, 0.4))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 1, 1))


@ti.func
def rect_voxels(x, y, z, lx, ly, lz, color):
    for i, j, k in ti.ndrange((x, x + lx), (y, y + ly), (z + lz)):
        scene.set_voxel(vec3(i, j, k), 1, color)


@ti.kernel
def initialize_voxels():
    rect_voxels(-6, -60, -24, 12, 108, 48, vec3(0, 0, 0))

    for i, k in ti.ndrange((-64, 64), (-64, 64)):
        scene.set_voxel(vec3(i, -64, k), 1, vec3(0.1, 0, 0))
        scene.set_voxel(vec3(i, -63, k), 1, vec3(0.1, 0, 0))
        scene.set_voxel(vec3(i, -62, k), 1, vec3(0.1, 0, 0))
        scene.set_voxel(vec3(i, -61, k), 1, vec3(0.1, 0, 0))


initialize_voxels()
scene.finish()
