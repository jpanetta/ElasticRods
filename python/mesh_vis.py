import numpy as np
import tri_mesh_viewer

def get_mesh_viewer(vertices, faces, normals):
    mesh = tri_mesh_viewer.RawMesh(vertices, faces, normals)
    return tri_mesh_viewer.TriMeshViewer(mesh), mesh
