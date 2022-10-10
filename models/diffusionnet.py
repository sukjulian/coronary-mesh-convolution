import torch
from utils.model_tools import parameter_table
import diffusion_net


class DiffusionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = diffusion_net.layers.DiffusionNet(
                C_in=7,
                C_out=3,
                N_block=4,
                C_width=32,
                last_activation=None,
                outputs_at='vertices'
        )
        
        print("{} ({} trainable parameters)".format(self.__class__.__name__, self.count_parameters))

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table(self)

    def forward(self, data):
        verts = data.pos.clone()
        faces = data.face.clone().t().contiguous()

        verts = diffusion_net.geometry.normalize_positions(verts)

        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
                verts.clone().cpu(),
                faces.clone().cpu(),
                op_cache_dir="diffusion_net-cache/",
                k_eig=512
        )

        features = torch.cat((verts, data.norm.clone(), data.geo.clone().view(-1, 1)), axis=1)

        device = data.pos.device
        outputs = self.layers(
                features,
                mass.to(device),
                L=L.to(device),
                evals=evals.to(device),
                evecs=evecs.to(device),
                gradX=gradX.to(device),
                gradY=gradY.to(device),
                faces=faces
        )

        return outputs
