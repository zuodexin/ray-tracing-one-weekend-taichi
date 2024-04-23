import taichi as ti
from vector import *
import ray
from material import Materials
from bvh import BVH

vec3 = ti.math.vec3


@ti.func
def is_front_facing(ray_direction, normal):
    return ray_direction.dot(normal) < 0.0


class Hittable:
    def __init__(self):
        self.id = -1

    @property
    def bounding_box(self):
        raise NotImplementedError

    @ti.func
    def hit(self, ray_origin, ray_direction, t_min, closest_so_far):
        hit_anything = False
        n = Vector(0.0, 0.0, 0.0)  # normal
        pos = Point(0.0, 0.0, 0.0)  # position
        return hit_anything, n, pos


@ti.dataclass
class SphereData(Hittable):

    center: vec3
    radius: ti.f32


class Sphere(Hittable):

    def __init__(self, center, radius, material):
        super().__init__()
        self.center = center
        self.radius = radius
        self.material = material

        self.box_min = [
            self.center[0] - radius,
            self.center[1] - radius,
            self.center[2] - radius,
        ]
        self.box_max = [
            self.center[0] + radius,
            self.center[1] + radius,
            self.center[2] + radius,
        ]

    @staticmethod
    @ti.func
    def hit_sphere(center, radius, ray_origin, ray_direction, t_min, t_max):
        """Intersect a sphere of given radius and center and return
        if it hit and the least root."""
        oc = ray_origin - center
        a = ray_direction.norm_sqr()
        half_b = oc.dot(ray_direction)
        c = oc.norm_sqr() - radius**2
        discriminant = (half_b**2) - a * c

        hit = discriminant >= 0.0
        root = -1.0
        if hit:
            sqrtd = discriminant**0.5
            root = (-half_b - sqrtd) / a

            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    hit = False

        return hit, root

    @staticmethod
    @ti.func
    def hit(sphere_data, ray_origin, ray_direction, t_min, closest_so_far):
        hit_anything, t = Sphere.hit_sphere(
            sphere_data.center,
            sphere_data.radius,
            ray_origin,
            ray_direction,
            t_min,
            closest_so_far,
        )

        return hit_anything, t

    @property
    def bounding_box(self):
        return self.box_min, self.box_max

    @staticmethod
    def set_data(sphere_data, sphere):
        sphere_data.center = sphere.center
        sphere_data.radius = sphere.radius


@ti.data_oriented
class World:
    def __init__(self):
        self.hittables = []

    def add(self, sphere):
        sphere.id = len(self.hittables)
        self.hittables.append(sphere)

    def commit(self):
        """Commit should be called after all objects added.
        Will compile bvh and materials."""
        self.n = len(self.hittables)

        self.materials = Materials(self.n)
        self.bvh = BVH(self.hittables)
        self.bvh.build()

        for i in range(self.n):
            self.materials.set(i, self.hittables[i].material)

        hittables = SphereData.field(shape=(self.n,))
        for i in range(self.n):
            Sphere.set_data(hittables[i], self.hittables[i])

        self.hittables = hittables

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        """Intersects a ray against all objects."""
        hit_anything = False
        t_min = 0.0001
        closest_so_far = 9999999999.9
        hit_index = 0
        p = Point(0.0, 0.0, 0.0)
        n = Vector(0.0, 0.0, 0.0)
        front_facing = True
        i = 0
        curr = self.bvh.bvh_root

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id = self.bvh.get_full_id(curr)

            if obj_id != -1:
                # this is a leaf node, check the sphere
                hit, t = Sphere.hit(
                    self.hittables[obj_id],
                    ray_origin,
                    ray_direction,
                    t_min,
                    closest_so_far,
                )
                if hit:
                    hit_anything = True
                    closest_so_far = t
                    hit_index = obj_id
                curr = next_id
            else:
                if self.bvh.hit_aabb(
                    curr, ray_origin, ray_direction, t_min, closest_so_far
                ):
                    # add left and right children
                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id
        if hit_anything:
            p = ray.at(ray_origin, ray_direction, closest_so_far)
            n = (p - self.hittables[hit_index].center) / self.hittables[
                hit_index
            ].radius
            front_facing = is_front_facing(ray_direction, n)
            n = n if front_facing else -n
        return hit_anything, p, n, front_facing, hit_index

    @ti.func
    def scatter(self, ray_direction, p, n, front_facing, index):
        """Get the scattered direction for a ray hitting an object"""
        return self.materials.scatter(index, ray_direction, p, n, front_facing)
