import vtk
from utils.vtk_tools import torch_to_vtk, add_fields
import torch


def append_file(sample, filename, fields=None):

    # Read the mesh data
    path = sample.dir[0]
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()

    # Attach an arbitrary number of fields
    polydata = add_fields(polydata, fields)

    # Save the VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Update()


def new_file(points, polygons, filename, fields=None):

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


def pooling_scales(sample):

    # Parse the cluster information
    scales = []
    for key in dir(sample):
        if 'scale' in key and 'cluster_map' in key:
            scales.extend([int(s) for s in key if s.isdigit()])
    scales.sort()

    # Construct the pooling clusters field via recursion
    pooling = torch.zeros(sample.num_nodes)
    for scale in scales:
        index = sample['scale' + str(scale) + '_sample_index']
        for s in reversed(range(scale)):
            index = sample['scale' + str(s) + '_sample_index'][index]
        pooling[index] = scale

    return pooling
