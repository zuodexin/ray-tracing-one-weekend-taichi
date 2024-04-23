import taichi as ti
import numpy as np
from vector import *


@ti.data_oriented
class Gaussian6D:
    def __init__(self):
        self.mju_a = ti.math.vec3(0)
        self.mju_b = ti.math.vec3(0)

        sigma = np.cov(np.random.rand(6, 20))

        self.sigma_aa = ti.math.mat3(sigma[:3, :3])
        self.sigma_bb = ti.math.mat3(sigma[3:, 3:])
        self.sigma_ab = ti.math.mat3(sigma[:3, 3:])

        self.color = Color(1.0, 1.0, 1.0)
        self.index = 3
        self.roughness = 0.0
        self.ior = 0

    @ti.func
    def condition_prob(self, x_b):
        mju_a_given_b = self.mju_a + self.sigma_ab @ ti.math.inverse(self.sigma_bb) @ (
            x_b - self.mju_b
        )
        sigma_a_given_b = (
            self.sigma_aa
            - self.sigma_ab @ ti.math.inverse(self.sigma_bb) @ self.sigma_ab.transpose()
        )

        return mju_a_given_b, sigma_a_given_b

    @ti.func
    def sample(self, x_b):
        mju_a_given_b, sigma_a_given_b = self.condition_prob(x_b)
        return self.multivariate_gaussian(mju_a_given_b, sigma_a_given_b)

    @staticmethod
    @ti.func
    def rand(n):
        return ti.Vector([ti.randn() for _ in range(n)])

    @staticmethod
    @ti.func
    def cholesky(A):
        L = ti.math.mat3(0.0)
        L[0, 0] = ti.sqrt(A[0, 0])
        L[1, 0] = A[1, 0] / L[0, 0]
        L[1, 1] = ti.sqrt(A[1, 1] - L[1, 0] ** 2)
        L[2, 0] = A[2, 0] / L[0, 0]
        L[2, 1] = (A[2, 1] - L[2, 0] * L[1, 0]) / L[1, 1]
        L[2, 2] = ti.sqrt(A[2, 2] - L[2, 0] ** 2 - L[2, 1] ** 2)
        return L

    @ti.func
    def multivariate_gaussian(self, mju, sigma):
        L = Gaussian6D.cholesky(sigma)
        z = Gaussian6D.rand(3)
        return mju + L @ z

    @ti.func
    def scatter(self, in_direction, p, n, color):
        out_direction = self.sample(in_direction)
        attenuation = color
        reflected = ti.math.dot(out_direction, n) > 0.0
        return reflected, p, out_direction, attenuation


if __name__ == "__main__":
    ti.init(ti.cpu)

    indirection = ti.math.vec3(np.random.rand(3).astype(np.float32))
    normal = ti.math.vec3(np.random.rand(3).astype(np.float32))

    gaussion6d = Gaussian6D()

    @ti.kernel
    def test():
        reflected, p, out_direction, attenuation = gaussion6d.scatter(
            indirection, None, normal, Color(1.0, 1.0, 1.0)
        )
        print(reflected, out_direction)

    test()
