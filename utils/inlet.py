import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import torch
import trimesh


class IndexFinder(object):
    """Find the inlet indices of a vessel sample previously used in simulation.

    Args:
        -
    """

    def __init__(self, pytorch=False):
        self.pytorch = pytorch
        self.reader = vtk.vtkXMLPolyDataReader()  # performance

    def read_file(self, file):

        # Load the mesh object
        self.reader.SetFileName(file)
        self.reader.Update()
        mesh = self.reader.GetOutput()

        # Extract the global vertex IDs
        vertices = vtk_to_numpy(mesh.GetPointData().GetArray("GlobalNodeID"))  # int32

        # Extract the polygons
        polygons = vtk_to_numpy(mesh.GetPolys().GetData())  # int64
        polygons = polygons.reshape((int(polygons.shape[0] / 4), 4))
        polygons = polygons[:, 1:]

        # Extract the vertex positions
        positions = vtk_to_numpy(mesh.GetPoints().GetData())  # float32

        return vertices, polygons, positions

    @staticmethod
    def vertex_map(source, target):
        mapping = [np.nonzero(np.all(target == point, axis=1)) for point in source]

        return np.asarray(mapping).squeeze()

    def __call__(self, surface_file):

        # Path to the inflow surface
        inlet_file = surface_file.replace("surface", "inlet")
        inlet_file = inlet_file.replace("sample", "bct")

        # Global vertex IDs and polygons
        inlet_vertices, inlet_polygons, inlet_positions = self.read_file(inlet_file)
        surface_vertices, surface_polygons, surface_positions = self.read_file(surface_file)

        # Find the vertex indices corresponding to the inlet surface
        source_vertices = [np.nonzero(surface_vertices == i) for i in inlet_vertices]

        # Translate the inlet polygon identifiers to the surface polygon numbering
        inlet_polygons = self.vertex_map(inlet_positions, surface_positions)[inlet_polygons]

        # Find the polygon indices corresponding to the inlet surface
        surface_polygons = np.sort(surface_polygons, axis=1)
        inlet_polygons = np.sort(inlet_polygons, axis=1)
        source_polygons = [np.nonzero(np.all(surface_polygons == i, axis=1)) for i in inlet_polygons]

        # Cast list of tuples to array
        source_vertices = np.asarray(source_vertices).squeeze()
        source_polygons = np.asarray(source_polygons).squeeze()

        # Pass as PyTorch tensor
        if self.pytorch:
            source_vertices = torch.from_numpy(source_vertices)
            source_polygons = torch.from_numpy(source_polygons)

        return source_vertices, source_polygons

    def area(self, surface_file):

        # Path to the inflow surface
        inlet_file = surface_file.replace("surface", "inlet")
        inlet_file = inlet_file.replace("sample", "bct")

        # Use trimesh object for area computation
        _, polygons, positions = self.read_file(inlet_file)
        area = trimesh.Trimesh(vertices=positions, faces=polygons).area

        return area
