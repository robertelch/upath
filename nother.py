import numpy as np

# -----------------------------
# Generator for lattice points
# -----------------------------
def lattice_generator(basis, limit):
    """
    Generate lattice points on-the-fly without storing all of them in memory.
    
    basis : np.ndarray of shape (3,3)
    limit : int
    """
    for i in range(-limit, limit+1):
        for j in range(-limit, limit+1):
            for k in range(-limit, limit+1):
                yield i*basis[0] + j*basis[1] + k*basis[2]

# -----------------------------
# Chunked processing function
# -----------------------------
def process_lattice_in_chunks(basis, limit, chunk_size=100_000):
    """
    Process lattice points in memory-efficient chunks.
    
    basis : np.ndarray (3,3)
    limit : int
    chunk_size : int, number of points to process at once
    """
    buffer = []
    for point in lattice_generator(basis, limit):
        buffer.append(point)
        if len(buffer) >= chunk_size:
            yield np.array(buffer, dtype=np.float64)
            buffer = []
    if buffer:
        yield np.array(buffer, dtype=np.float64)

# -----------------------------
# Example processing function
# -----------------------------
def example_processing(points_chunk):
    """
    Dummy processing function for demonstration.
    For example, compute norms of points.
    """
    return np.linalg.norm(points_chunk, axis=1)

# -----------------------------
# Example usage
# -----------------------------
basis = np.eye(3)  # cubic lattice
limit = 1000       # huge lattice

results = []
for chunk in process_lattice_in_chunks(basis, limit, chunk_size=1_000_000):
    print("chunky")
    norms = example_processing(chunk)
    results.append(norms)  # or do any other processing here

# Optionally, concatenate results
all_norms = np.concatenate(results)
print("Processed", all_norms.size, "points.")
