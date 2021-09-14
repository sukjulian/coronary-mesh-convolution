from torch_geometric.data import Data


# Overwrite "Data" class to ensure correct batching for pooling hierarchy
class MultiscaleData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 y=None, pos=None, normal=None, face=None, **kwargs):

        super(MultiscaleData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, pos=pos, normal=normal, face=face, **kwargs)

    def __inc__(self, key, value):

        # Batch edges and polygons as before
        if key == 'edge_index' or key == 'face':
            return self.num_nodes

        # Batch scales correctly
        elif 'scale' in key:
            return self[key[:6] + '_cluster_map'].max() + 1

        else:
            return 0

    def __cat_dim__(self, key, value):
        if 'edge_index' in key or 'face' in key:
            return 1
        else:
            return 0
