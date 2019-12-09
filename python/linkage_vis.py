import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed
import MeshFEM
import tri_mesh_viewer

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
