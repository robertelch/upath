import numpy as np
import itertools

def lattice_point(v1, v2, v3, a, b, c):
    """Compute the lattice point corresponding to (a,b,c)."""
    return a * np.array(v1) + b * np.array(v2) + c * np.array(v3)

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

def ndarray_to_set(arr):

    # Convert to a set of unique rows (as tuples)
    set_of_tuples = {tuple(row) for row in arr}

    return (set_of_tuples)


# Example usage:
if __name__ == "__main__":
    # FCC lattice basis vectors (a = 1)
    v1 = [0, 0.5, 0.5]
    v2 = [0.5, 0, 0.5]
    v3 = [0.5, 0.5, 0]

    # Example lattice coordinates
    a, b, c = (0, 0, 0)

    # Compute lattice point
    P = lattice_point(v1, v2, v3, a, b, c)
    P2 = lattice_point(v1, v2, v3, a, 2, c)
    # Compute neighbours
    neighbours = lattice_neighbours(v1, v2, v3, a, b, c)
    neighbours2 = lattice_neighbours(v1,v2,v3,a,2,c)
    to_lin = lambda a: [np.linalg.norm(n) for n in a]
    print(to_lin(ndarray_to_set(neighbours))==to_lin(ndarray_to_set(neighbours2)))

    
    print("\nMax Distance:")
    print(max([np.linalg.norm(n) for n in neighbours]))
