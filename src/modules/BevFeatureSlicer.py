import torch.nn as nn
import torch
import torch.functional as F
from src.utils.utils import calculate_birds_eye_view_parameters

# Grid sampler
# Sample a smaller receptive-field bev from larger one
class BevFeatureSlicer(nn.Module):
    def __init__(self, grid_conf, map_grid_conf):
        super(BevFeatureSlicer, self).__init__()
        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension= calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            tmp_m, tmp_n = torch.meshgrid(
                self.norm_map_x, self.norm_map_y)  # indexing 'ij'
            tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results

            self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)  # (b, h, w, 2)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
