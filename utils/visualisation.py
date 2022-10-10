import vtk
from utils.vtk_tools import torch_to_vtk
import torch


def new_file(points, polygons, filename, fields=None):

    # Point cloud support
    if polygons == None:
        polygons = torch.arange(points.size(0)).expand(3, -1).long()  # dummy polygons

    # Create the "vtkPolyData" object
    polydata = torch_to_vtk(points, polygons, fields)

    # Write the files
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def default_fields(sample, prediction):

    # Standard visualisation fields for the creation of a new VTP file
    padding = torch.zeros(sample.pos.shape[0])
    padding[sample.mask] = 1.
    label = sample.y

    fields = {'prediction': prediction, 'label': label, 'error': label - prediction,
              'normals': sample['normal' if 'normal' in sample else 'norm'],  # catch different naming
              'geodesics': sample.geo, 'padding': padding}

    return fields
