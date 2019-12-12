import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed
import MeshFEM
import tri_mesh_viewer
import elastic_rods

# Render a linkage or a single elastic rod.
class LinkageViewer(tri_mesh_viewer.TriMeshViewer):
    @property
    def averagedMaterialFrames(self):
        return self._averagedMaterialFrames

    @averagedMaterialFrames.setter
    def averagedMaterialFrames(self, value):
        self._averagedMaterialFrames = value
        self.update()

    def getVisualizationGeometry(self):
        # Note: getVisualizationGeometry is called by TriMeshViewer's constructor,
        # so we can initialize our member variables here. (So we don't need to
        # implement a __init__ method that forwards arguments and breaks tab completion).
        if not hasattr(self, '_averagedMaterialFrames'):
            self._averagedMaterialFrames = False
            # redraw flicker isn't usually a problem for the linkage
            # deployment--especially since the index buffer isn't changing--and
            # we prefer to enable smooth interaction during deployment
            self.avoidRedrawFlicker = False
        return self.mesh.visualizationGeometry(self._averagedMaterialFrames)

class CenterlineMesh:
    def __init__(self, rodOrLinkage):
        self.rodOrLinkage = rodOrLinkage

    def visualizationGeometry(self):
        rods = []
        rodOrLinkage = self.rodOrLinkage
        if isinstance(rodOrLinkage, elastic_rods.ElasticRod):
            rods = [rodOrLinkage]
        elif isinstance(rodOrLinkage, elastic_rods.RodLinkage):
            rods = [s.rod for s in rodOrLinkage.segments()]
        else: raise Exception('Unsupported object type')

        V = np.empty((0, 3), dtype=np.float32)
        E = np.empty((0, 2), dtype=np. uint32)
        N = np.empty((0, 3), dtype=np.float32)
        for r in rods:
            idxOffset = V.shape[0]
            V = np.row_stack([V, np.array(r.deformedPoints(), dtype=np.float32)])
            E = np.row_stack([E, idxOffset + np.column_stack([np.arange(r.numVertices() - 1, dtype=np.uint32), np.arange(1, r.numVertices(), dtype=np.uint32)])])
            # Use the d2 directory averaged onto the vertices as the per-vertex normal
            padded = np.pad([np.array(f.d2, dtype=np.float32) for f in r.deformedConfiguration().materialFrame], [(1, 1), (0, 0)], mode='edge') # duplicate the first and last edge frames
            N = np.row_stack([N, 0.5 * (padded[:-1] + padded[1:])])
        return V, E, N

    # No decoding needed for per-entity fields on raw meshes.
    def visualizationField(self, data):
        return data

class CenterlineViewer(tri_mesh_viewer.LineMeshViewer):
    def __init__(self, rodOrLinkage, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None):
        super().__init__(CenterlineMesh(rodOrLinkage), width=width, height=height, scalarField=scalarField, vectorField=vectorField, superView=superView)
