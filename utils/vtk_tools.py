from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import vtk
import numpy as np


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
