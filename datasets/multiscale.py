from torch_geometric.data import Data
from torch_sparse import SparseTensor


# Overwrite "Data" class to ensure correct batching for pooling hierarchy
class MultiscaleData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 y=None, pos=None, normal=None, face=None, **kwargs):

        super(MultiscaleData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, pos=pos, normal=normal, face=face, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if 'batch' in key:
            return int(value.max()) + 1

        # Batch edges and polygons as before
        elif key == 'edge_index' or key == 'face':
            return self.num_nodes

        # Batch scales correctly
        elif 'scale' in key and ('cluster_map' in key or 'edge_index' in key):
            return self[key[:6] + '_cluster_map'].max() + 1

        elif 'scale' in key and 'sample_index' in key:
            if int(key[5]) == 0:
                return self.num_nodes
            else:
                return self['scale' + str(int(key[5]) - 1) + '_sample_index'].size(dim=0)

        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'edge_index' in key or 'face' in key:
            return -1
        else:
            return 0
