from models import GEM
from transforms import InletGeodesics, RemoveFlowExtensions
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh
from gem_cnn.transform.multiscale_radius_graph import MultiscaleRadiusGraph
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.matrix_features_transform import MatrixFeaturesTransform
from .template import Experiment


def fit(device):
    # Multi-scale neural network
    ratios = [1., 0.4, 0.1]
    radii = [0.042, 0.06, 0.8]
    model = GEM(radii)

    # Training data
    dataset = 'single_arteries'
    batch_size = 4

    # Experiment tag
    tag = "gem_" + dataset

    # Precomputed graph transforms
    transforms = [
        compute_normals_edges_from_mesh,
        InletGeodesics(),
        RemoveFlowExtensions(),
        MultiscaleRadiusGraph(ratios, radii, max_neighbours=512),
        SimpleGeometry(gauge_def='random'),
        MatrixFeaturesTransform()
    ]

    # Neural network training
    experiment = Experiment(model=model, dataset=dataset, batch_size=batch_size, tag=tag, transforms=transforms,
                            epochs=160)
    experiment.run(device)
