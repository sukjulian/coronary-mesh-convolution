from models import GEMGCN
from transforms import InletGeodesics, RemoveFlowExtensions
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh
from gem_cnn.transform.multiscale_radius_graph import MultiscaleRadiusGraph
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.matrix_features_transform import MatrixFeaturesTransform
from ..utils import Experiment


def fit(artery_type, num_epochs, device, gpu):

    # Multi-scale neural network
    ratios = [1., 0.4, 0.1]
    radii = [0.042, 0.06, 0.8] if artery_type == 'single' else [0.022, 0.04, 0.1]
    model = GEMGCN(radii, in_rep=(2, 8), out_rep=(1, 1))

    batch_size = 12

    # Experiment tag
    tag = f"stead_{artery_type}"

    # Precomputed graph transforms
    transforms = [
        compute_normals_edges_from_mesh,
        InletGeodesics(),
        RemoveFlowExtensions(factor=(4., 1.)) if artery_type == 'single' else RemoveFlowExtensions(factor=(-1., 1.)),
        MultiscaleRadiusGraph(ratios, radii, max_neighbours=512),
        SimpleGeometry(gauge_def='random'),
        MatrixFeaturesTransform()
    ]

    # Neural network training
    experiment = Experiment(
        model=model,
        dataset=artery_type,
        batch_size=batch_size,
        tag=tag,
        transforms=transforms,
        epochs=num_epochs,
        parallel=gpu
    )
    experiment.run(device)
