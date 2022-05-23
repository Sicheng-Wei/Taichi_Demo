from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(-1, (0.5, 0.5, 0.4))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, 1), 0.2, (0.03, 0.03, 0.03)) # dark mode
scene.set_directional_light((1, 1, 1), 0.2, (0.9, 0.9, 0.9)) # light mode

initialize_voxels()
scene.finish()