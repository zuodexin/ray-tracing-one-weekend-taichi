import taichi as ti
import ray
from vector import *
from time import time


@ti.func
def get_background(dir):
    """Returns the background color for a given direction vector"""
    unit_direction = dir.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE


class Renderer:

    def __init__(
        self, image_width=1200, image_height=800, max_depth=16, samples_per_pixel=512
    ) -> None:
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.aspect_ratio = self.image_width / self.image_height
        self.rays = ray.Rays(self.image_width, self.image_height)
        self.pixels = ti.Vector.field(3, dtype=float)
        self.sample_count = ti.field(dtype=ti.i32)
        self.needs_sample = ti.field(dtype=ti.i32)
        ti.root.dense(ti.ij, (self.image_width, self.image_height)).place(
            self.pixels, self.sample_count, self.needs_sample
        )
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth

    def render(self, cam, world):

        start_attenuation = Vector(1.0, 1.0, 1.0)

        @ti.kernel
        def finish():
            for x, y in self.pixels:
                self.pixels[x, y] = ti.sqrt(self.pixels[x, y] / self.samples_per_pixel)

        @ti.kernel
        def wavefront_initial():
            for x, y in self.pixels:
                self.sample_count[x, y] = 0
                self.needs_sample[x, y] = 1

        @ti.kernel
        def wavefront_big() -> ti.i32:
            """Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample backgound
            return pixels that hit max samples
            """
            num_completed = 0
            for x, y in self.pixels:
                if self.sample_count[x, y] == self.samples_per_pixel:
                    continue

                # gen sample
                ray_org = Point(0.0, 0.0, 0.0)
                ray_dir = Vector(0.0, 0.0, 0.0)
                depth = self.max_depth
                pdf = start_attenuation

                if self.needs_sample[x, y] == 1:
                    self.needs_sample[x, y] = 0
                    u = (x + ti.random()) / (self.image_width - 1)
                    v = (y + ti.random()) / (self.image_height - 1)
                    ray_org, ray_dir = cam.get_ray(u, v)
                    self.rays.set(x, y, ray_org, ray_dir, depth, pdf)
                else:
                    ray_org, ray_dir, depth, pdf = self.rays.get(x, y)

                # intersect
                hit, p, n, front_facing, index = world.hit_all(ray_org, ray_dir)
                depth -= 1
                self.rays.depth[x, y] = depth
                if hit:
                    reflected, out_origin, out_direction, attenuation = (
                        world.materials.scatter(index, ray_dir, p, n, front_facing)
                    )
                    self.rays.set(
                        x, y, out_origin, out_direction, depth, pdf * attenuation
                    )
                    ray_dir = out_direction

                if not hit or depth == 0:
                    self.pixels[x, y] += pdf * get_background(ray_dir)
                    self.sample_count[x, y] += 1
                    self.needs_sample[x, y] = 1

                    if self.sample_count[x, y] == self.samples_per_pixel:
                        num_completed += 1

            return num_completed

        num_pixels = self.image_width * self.image_height

        t = time()
        print("starting big wavefront")
        wavefront_initial()
        num_completed = 0
        while num_completed < num_pixels:
            num_completed += wavefront_big()
            # print("completed", num_completed)

        finish()
        print(time() - t)
        return self.pixels.to_numpy()
