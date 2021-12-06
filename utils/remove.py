import torch
from torch_geometric.utils import remove_isolated_nodes
import numpy as np


def remove_vertices(data, vertex_mask, dummy_mask=False):

    # Edge and polygon masks
    cut = torch.nonzero(torch.eq(vertex_mask, False))
    edges, polygons = data.edge_index, data.face
    if edges is not None:
        edge_mask = torch.ones(edges.shape[1], dtype=torch.bool)
    polygon_mask = torch.ones(polygons.shape[1], dtype=torch.bool)
    for c in cut:
        if edges is not None:
            edge_mask = torch.logical_and(edge_mask,
                                          torch.all(torch.ne(edges, c), dim=0))
        polygon_mask = torch.logical_and(polygon_mask,
                                         torch.all(torch.ne(polygons, c), dim=0))

    # Remove the desired vertices from the edges
    if edges is not None:
        edges = edges[:, edge_mask]
        data.edge_index, _, _ = remove_isolated_nodes(edges)

    # Remove the now isolated vertex features
    data.geo = data.geo[vertex_mask]
    if 'norm' in data or 'normal' in data:
        data['norm' if 'norm' in data else 'normal'] = data['norm' if 'norm' in data else 'normal'][vertex_mask]
    data.pos = data.pos[vertex_mask]
    data.y = data.y[vertex_mask]

    # Remove the redundant mesh faces
    polygons = data.face[:, polygon_mask]  # in the old indices
    _, indices = np.unique(polygons, return_inverse=True)
    data.face = torch.from_numpy(indices.reshape(polygons.shape))

    # Dummy vertex mask
    if dummy_mask:
        data.mask = torch.full((data.pos.shape[0],), True)

    return data
