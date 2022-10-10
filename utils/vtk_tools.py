from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import vtk
import numpy as np
import torch


# Create a "vtkPolyData" object from PyTorch data
def torch_to_vtk(points, polygons, fields=None, ftype='point'):

    # Point data
    points = numpy_to_vtk(points.detach().cpu().numpy())
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(points)

    # Polygon data
    polygons = polygons.detach().cpu().numpy().transpose()
    specifier = np.full(polygons.shape[0], 3, dtype=np.int64)
    polygons = np.column_stack((specifier, polygons)).ravel()
    vtkpolygons = vtk.vtkCellArray()
    vtkpolygons.SetCells(polygons.shape[0], numpy_to_vtkIdTypeArray(polygons))

    # Create the "vtkPolyData" object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtkpoints)
    polydata.SetPolys(vtkpolygons)
    polydata.Modified()

    # Add an arbitrary amount of new fields
    polydata = add_fields(polydata, fields, ftype=ftype)

    return polydata


def add_fields(polydata, fields, ftype='point'):

    # Iterate through "fields" and append field type "ftype"
    if fields is not None:
        for key in fields:
            array = numpy_to_vtk(fields[key].detach().cpu().numpy())
            array.SetName(key)
            if ftype == 'point':
                polydata.GetPointData().AddArray(array)
            if ftype == 'cell':
                polydata.GetCellData().AddArray(array)
            if ftype == 'scalars':
                polydata.GetPointData().SetScalars(array)
            if ftype == 'vectors':
                polydata.GetPointData().SetVectors(array)
            if ftype == 'normals':
                polydata.GetPointData().SetNormals(array)

    return polydata


# Parse polygon array from "vtkPolyData"
def parse_polygons(polygons, wind=False):
    """
    The first elements in "polygons" specify how many nodes are in
    the current polygon, followed by the corresponding point indices,
    e.g. z = [3 0 1 2 3 3 4 5 3 6 ...]. Assume all are triangles:
    """

    polygons = polygons.reshape((int(polygons.shape[0] / 4), 4))
    polygons = polygons[:, 1:]

    if wind:
        polygons[:, [1, 0]] = polygons[:, [0, 1]]  # correct face orientation

    return polygons


# Read vertex positions and polygons from "vtkPolyData"
def vtk_to_torch(polydata):
    vertices = vtk_to_numpy(polydata.GetPoints().GetData())  # float32
    polygons = parse_polygons(vtk_to_numpy(polydata.GetPolys().GetData()))  # int64

    return torch.from_numpy(vertices), torch.from_numpy(polygons.T.astype('i4'))
