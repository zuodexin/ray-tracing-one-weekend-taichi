from renderer import Renderer
import taichi as ti
from vector import *
import ray
from time import time
from hittable import World, Sphere
from camera import Camera
from material import *
import math
import random


# switch to cpu if needed
ti.init(arch=ti.gpu)


if __name__ == "__main__":
    # materials
    mat_ground = Lambert([0.5, 0.5, 0.5])
    mat2 = Lambert([0.4, 0.2, 0.2])
    mat4 = Dielectric(1.5)
    mat3 = Metal([0.7, 0.6, 0.5], 0.0)
    mat4 = Gaussian6D()

    # world
    R = math.cos(math.pi / 4.0)
    world = World()
    world.add(Sphere([0.0, -1000, 0], 1000.0, mat_ground))

    static_point = Point(4.0, 0.2, 0.0)
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = Point(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())

            if (center - static_point).norm() > 0.9:
                if choose_mat < 0.5:
                    mat = Gaussian6D()
                elif choose_mat < 0.8:
                    # diffuse
                    mat = Lambert(
                        Color(random.random(), random.random(), random.random()) ** 2
                    )
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(
                        Color(random.random(), random.random(), random.random()) * 0.5
                        + 0.5,
                        random.random() * 0.5,
                    )
                else:
                    mat = Dielectric(1.5)

            world.add(Sphere(center, 0.2, mat))

    world.add(Sphere([0.0, 1.0, 0.0], 1.0, mat4))
    world.add(Sphere([-4.0, 1.0, 0.0], 1.0, mat4))
    world.add(Sphere([4.0, 1.0, 0.0], 1.0, mat4))
    world.commit()

    # camera
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    up = Vector(0.0, 1.0, 0.0)
    focus_dist = 10.0
    aperture = 0.1
    aspect_ratio = 3 / 2
    cam = Camera(vfrom, at, up, 20.0, aspect_ratio, aperture, focus_dist)

    image_width = 1200
    renderer = Renderer(
        image_width=image_width,
        image_height=image_width / aspect_ratio,
    )

    image = renderer.render(cam, world)
    ti.tools.image.imwrite(image, "out.png")
