import numpy as np
from scipy.spatial import ConvexHull
from collections import defaultdict

# -----------------------------
# Define lattices (float arrays!)
# -----------------------------
SC = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
FCC = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float) 
BCC = np.array([[1,1,-1],[1,-1,1],[-1,1,1]], dtype=float) 

# -----------------------------
# Generate lattice points
# -----------------------------
def generate_lattice_points(basis, limit):
    points = []
    for i in range(-limit, limit+1):
        for j in range(-limit, limit+1):
            for k in range(-limit, limit+1):
                p = i*basis[0] + j*basis[1] + k*basis[2]
                points.append(p.astype(float))  # ensure float
    return np.array(points, dtype=float)

# -----------------------------
# Filter points inside a sphere
# -----------------------------
def filter_points(points, center, radius):
    dist2 = np.sum((points - center)**2, axis=1)
    return points[dist2 <= radius**2]

# -----------------------------
# Merge coplanar triangles into polygons
# -----------------------------
def merge_coplanar_triangles(points, hull, epsilon=1e-6):
    plane_groups = defaultdict(list)

    for simplex_index, eq in enumerate(hull.equations):
        normal = eq[:3].astype(float)  # cast to float
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue  # skip degenerate
        normal /= norm
        key = tuple(np.round(normal, 6))
        plane_groups[key].append(hull.simplices[simplex_index])

    faces = []
    for tris in plane_groups.values():
        verts = set()
        for tri in tris:
            verts.update(tri)
        verts = list(verts)
        face_pts = points[verts]
        centroid = np.mean(face_pts, axis=0)

        # Compute plane normal safely
        edge1 = (face_pts[1] - face_pts[0]).astype(float)
        edge2 = (face_pts[2] - face_pts[0]).astype(float)
        n = np.cross(edge1, edge2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            n = np.array([0.0,0.0,1.0])
        else:
            n /= norm_n

        # Create safe axes for sorting
        ref = (face_pts[0] - centroid).astype(float)
        norm_ref = np.linalg.norm(ref)
        if norm_ref < 1e-12:
            ref = np.array([1.0,0.0,0.0])
        else:
            ref /= norm_ref
        axis2 = np.cross(n, ref)
        norm_axis2 = np.linalg.norm(axis2)
        if norm_axis2 < 1e-12:
            axis2 = np.cross(n, np.array([0.0,1.0,0.0]))
            axis2 /= np.linalg.norm(axis2)
        else:
            axis2 /= norm_axis2

        def angle(p):
            v = p - centroid
            return np.arctan2(np.dot(v, axis2), np.dot(v, ref))

        verts = sorted(verts, key=lambda vi: angle(points[vi]))
        faces.append(verts)

    return faces

# -----------------------------
# Ensure outward normals
# -----------------------------
def ensure_outward_normals(points, faces):
    poly_centroid = np.mean(points, axis=0)
    corrected_faces = []
    for face in faces:
        face_pts = points[face]
        face_centroid = np.mean(face_pts, axis=0)
        edge1 = (face_pts[1] - face_pts[0]).astype(float)
        edge2 = (face_pts[2] - face_pts[0]).astype(float)
        n = np.cross(edge1, edge2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            n = np.array([0.0,0.0,1.0])
        else:
            n /= norm_n
        if np.dot(face_centroid - poly_centroid, n) < 0:
            face = list(reversed(face))
        corrected_faces.append(face)
    return corrected_faces

# -----------------------------
# Export polygonal OBJ
# -----------------------------
def export_polygonal_obj(points, faces, filename="waterman.obj"):
    with open(filename, 'w') as f:
        for v in points:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            idxs = [str(i+1) for i in face]
            f.write(f"f {' '.join(idxs)}\n")
    print(f"Exported {filename} with {len(faces)} polygonal faces.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    basis = FCC       # choose SC, FCC, BCC, or custom
    limit = 20         # lattice search range
    radius = np.sqrt(2*7)      # sphere radius
    center = np.array([0.0,0.0,0.0], dtype=float)

    lattice_points = generate_lattice_points(basis, limit)
    sphere_points = filter_points(lattice_points, center, radius+1e-8)

    hull = ConvexHull(sphere_points)
    faces = merge_coplanar_triangles(sphere_points, hull)
    faces = ensure_outward_normals(sphere_points, faces)

    export_polygonal_obj(sphere_points, faces, filename="waterman_poly.obj")
