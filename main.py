from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(-1, (0.5, 0.5, 0.4))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, 1), 0.2, (0.03, 0.03, 0.03)) # dark mode
# scene.set_directional_light((1, 1, 1), 0.2, (0.9, 0.9, 0.9)) # light mode
pale_gold, black = vec3(190/256, 177/256, 139/256), vec3(0.03, 0.03, 0.03)
dark, pure_gold, nv_green = vec3(0.02, 0.02, 0.02), vec3(193 / 256, 129 / 256, 69 / 256), vec3(119/256, 170/256, 0)

@ti.func
def rect_voxels(x, y, z, lx, ly, lz, color, vox=1):
    for i, j, k in ti.ndrange((x, x + lx), (y, y + ly), (z, z + lz)):
        scene.set_voxel(vec3(i, j, k), vox, color)

@ti.func
def single_a100(x, y, z):
    # gpu body
    rect_voxels(x, y, z, 109, 10, 40, pale_gold)
    rect_voxels(x, y + 10, z - 1, 109, 5, 42, black)
    # pcie x16
    rect_voxels(x + 20, y + 14, z - 5, 5, 1, 4, pure_gold), rect_voxels(x + 20, y + 14, z - 3, 5, 1, 2, black)
    rect_voxels(x + 26, y + 14, z - 5, 30, 1, 4, pure_gold), rect_voxels(x + 26, y + 14, z - 3, 30, 1, 2, black)
    rect_voxels(x + 57, y + 14, z - 3, 3, 1, 2, black)
    # gpu edge
    for i in ti.ndrange((0, 55)):
        rect_voxels(x + i * 2, y, z - 1, 1, 10, 1, pale_gold)
        rect_voxels(x + i * 2, y, -z, 1, 10, 1, pale_gold)
    rect_voxels(x + 60, y, z - 1, 7, 15, 2, black, 0)
    # air duct
    rect_voxels(x, y + 1, z + 1, 109, 8, 1, black, 0)
    rect_voxels(x, y + 1, z + 38, 109, 8, 1, black, 0)
    for i in ti.ndrange(1, 9):
        rect_voxels(x, y + 1, z + 4 * i + 2, 109, 8, 1, black, 0)
        rect_voxels(x, y + 1, z + 4 * i + 5, 109, 8, 1, black, 0)
    # power connect
    rect_voxels(x + 108, y + 10, z, 2, 5, 9, vec3(0.07, 0.07, 0.07))
    for i, j in ti.ndrange((0, 2), (0, 4)):
        rect_voxels(x + 108, y + 11 + 2 * i, z + 1 + 2 * j, 2, 1, 1, black, 0)

    # name
    x = x + 90
    for i in ti.ndrange((0, 7)):
        rect_voxels(x + 2 * i + 1, y + 11, -z, 1, 3, 1, pale_gold, 2)
    scene.set_voxel(vec3(x + 2, y + 12, -z), 2, pale_gold)
    scene.set_voxel(vec3(x + 8, y + 11, -z), 2, pale_gold), scene.set_voxel(vec3(x + 8, y + 13, -z), 2, pale_gold)
    scene.set_voxel(vec3(x + 12, y + 11, -z), 2, pale_gold), scene.set_voxel(vec3(x + 12, y + 13, -z), 2, pale_gold)

@ti.func
def link_bridge(x, y, z):
    rect_voxels(x, y, z, 19, 19, 2, vec3(0, 0, 0)), rect_voxels(x + 1, y + 1, z, 17, 17, 2, dark)
    rect_voxels(x + 1, y + 1, z, 17, 17, 2, dark)
    rect_voxels(x + 1, y, z, 17, 1, 1, pure_gold), rect_voxels(x + 1, y + 18, z, 17, 1, 1, pure_gold)
    rect_voxels(x, y + 1, z, 1, 17, 1, pure_gold), rect_voxels(x + 18, y + 1, z, 1, 17, 1, pure_gold)
    # voxel nvidia logo
    rect_voxels(x + 15, y + 5, z + 1, 2, 10, 1, nv_green, 2)
    rect_voxels(x + 14, y + 5, z + 1, 1, 3, 1, nv_green, 2), rect_voxels(x + 14, y + 9, z + 1, 1, 6, 1, nv_green, 2)
    rect_voxels(x + 13, y + 5, z + 1, 1, 2, 1, nv_green, 2), rect_voxels(x + 13, y + 9, z + 1, 1, 6, 1, nv_green, 2)
    rect_voxels(x + 12, y + 5, z + 1, 1, 2, 1, nv_green, 2), rect_voxels(x + 12, y + 8, z + 1, 1, 2, 1, nv_green, 2)
    rect_voxels(x + 12, y + 12, z + 1, 1, 3, 1, nv_green, 2)
    rect_voxels(x + 11, y + 5, z + 1, 1, 2, 1, nv_green, 2), rect_voxels(x + 11, y + 8, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 11, y + 12, z + 1, 1, 3, 1, nv_green, 2)
    rect_voxels(x + 10, y + 5, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 10, y + 7, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 10, y + 9, z + 1, 1, 3, 1, nv_green, 2), rect_voxels(x + 10, y + 13, z + 1, 1, 2, 1, nv_green, 2)
    rect_voxels(x + 9, y + 5, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 9, y + 7, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 9, y + 9, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 9, y + 11, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 9, y + 13, z + 1, 1, 2, 1, nv_green, 2)
    rect_voxels(x + 8, y + 5, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 8, y + 7, z + 1, 1, 3, 1, nv_green, 2)
    rect_voxels(x + 8, y + 12, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 8, y + 14, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 7, y + 5, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 7, y + 8, z + 1, 1, 3, 1, nv_green, 2)
    rect_voxels(x + 7, y + 12, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 7, y + 14, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 6, y + 6, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 6, y + 8, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 6, y + 11, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 5, y + 7, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 5, y + 9, z + 1, 1, 2, 1, nv_green, 2)
    rect_voxels(x + 5, y + 12, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 4, y + 7, z + 1, 1, 1, 1, nv_green, 2), rect_voxels(x + 4, y + 12, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 3, y + 8, z + 1, 1, 2, 1, nv_green, 2), rect_voxels(x + 3, y + 11, z + 1, 1, 1, 1, nv_green, 2)
    rect_voxels(x + 2, y + 9, z + 1, 1, 2, 1, nv_green, 2)

@ti.kernel
def initialize_voxels():
    single_a100(-55, -55, -20), single_a100(-55, -38, -20)
    link_bridge(-53, -43, 21), link_bridge(-33, -43, 21), link_bridge(-13, -43, 21)

initialize_voxels()
scene.finish()
