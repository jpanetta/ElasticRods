################################################################################
#  auto-generated from /home/jpanetta/Research/ElasticRods_release/3rdparty/MeshFEM/python/init_template.py
################################################################################
import sys as _sys
_sys.path.insert(0, '/home/jpanetta/Research/ElasticRods_release/3rdparty/MeshFEM/python')

import sparse_matrices
import mesh
from mesh import Mesh, PeriodicCondition

import importlib
if importlib.util.find_spec('pythreejs') is not None:
    import tri_mesh_viewer
