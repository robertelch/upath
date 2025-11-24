import numpy as np
import itertools
class LatticeMatrix(np.ndarray):
    def from_cartesian(self, v):
        return self @ v

    def permutation_matrix(self):
        return np.array(list(itertools.product([-1, 0, 1], repeat=self.shape[1])))

    def max_spherical_neighbour_distance(self):


    def lattice_neighbours(v1, v2, v3, a, b, c):
        """
        Compute all neighbours of the lattice point given by (a,b,c) using:
        N(a,b,c) = { (a+α1)v1 + (b+α2)v2 + (c+α3)v3 | αi ∈ {-1,0,1}, not all zero }
        """
        neighbours = []
        for alpha1, alpha2, alpha3 in itertools.product([-1, 0, 1], repeat=3):
            if (alpha1, alpha2, alpha3) != (0, 0, 0):
                neighbour_point = lattice_point(v1, v2, v3,
                                                a + alpha1, b + alpha2, c + alpha3)
                neighbours.append(neighbour_point)
        return np.array(neighbours)



# Define a 2D lattice matrix (columns are basis vectors)
lattice = np.array([[2.0, 1.0],   # x-components of basis vectors
                    [0.0, 3.0]]).view(LatticeMatrix)  # y-components of basis vectors

# Define some integer vectors in lattice coordinates
vec1 = np.array([1, 0])  # 1 * first basis vector + 0 * second basis vector
vec2 = np.array([0, 1])  # 0 * first + 1 * second
vec3 = np.array([2, 3])  # 2 * first + 3 * second

cart1 = lattice.from_cartesian(vec1)
cart2 = lattice.from_cartesian(vec2)
cart3 = lattice.from_cartesian(vec3)

print("Lattice basis:\n", lattice)
print("\nInteger vector:", vec1, "-> Cartesian:", cart1)
print("Integer vector:", vec2, "-> Cartesian:", cart2)
print("Integer vector:", vec3, "-> Cartesian:", cart3)
