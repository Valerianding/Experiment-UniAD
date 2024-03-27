import numpy as np
import torch
import cv2 as cv
import mmcv
import copy
import warnings
import torch.nn as nn
import utils
import time
import pickle
from utils import ModuleList, TORCH_VERSION
from custom_base_transformer_layer import CustomBaseTransformerLayer
from builder import build_attention,build_feedforward_network,build_transformer_layer

class BEVFormerEncoder(nn.Module):
    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):
        super(BEVFormerEncoder,self).__init__()
        self.num_layers = 6
        self.layers = ModuleList()
        transformerlayer = {'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}
        for _ in range(self.num_layers):
            self.layers.append(build_transformer_layer(transformerlayer))
        
        
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    
    def point_sampling(self, reference_points, pc_range,  img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        assert lidar2img.shape == (1,6,4,4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        bev_mask = torch.nan_to_num(bev_mask)
        # else:
        #     bev_mask = bev_mask.new_tensor(
        #         np.nan_to_num(bev_mask.cpu().numpy()))
        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask
            
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                img_metas=None,
                **kwargs):
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
            
        for lid, layer in enumerate(self.layers):
            assert(len(self.layers) == 6)
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)
            
            bev_query = output
            
            if self.return_intermediate:
                intermediate.append(output)
            
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class BEVFormerLayer(CustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
    
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        with torch.no_grad():
            norm_index = 0
            attn_index = 0
            ffn_index = 0
            identity = query
            if attn_masks is None:
                attn_masks = [None for _ in range(self.num_attn)]
            elif isinstance(attn_masks, torch.Tensor):
                attn_masks = [
                    copy.deepcopy(attn_masks) for _ in range(self.num_attn)
                ]
                warnings.warn(f'Use same attn_mask in all attentions in '
                            f'{self.__class__.__name__} ')
            else:
                assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                        f'attn_masks {len(attn_masks)} must be equal ' \
                                                        f'to the number of attention in ' \
                    f'operation_order {self.num_attn}'
                    
                    
            for layer in self.operation_order:
                # temporal self attention
                if layer == 'self_attn':
    
                    query = self.attentions[attn_index](
                        query,
                        prev_bev,
                        prev_bev,
                        identity if self.pre_norm else None,
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        reference_points=ref_2d,
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        **kwargs)
                    attn_index += 1
                    identity = query

                elif layer == 'norm':
           
                    query = self.norms[norm_index](query)
                    norm_index += 1

                # spaital cross attention
                elif layer == 'cross_attn':
        
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        reference_points=ref_3d,
                        reference_points_cam=reference_points_cam,
                        mask=mask,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        **kwargs)
                    attn_index += 1
                    identity = query

                elif layer == 'ffn':
  
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                    ffn_index += 1
                    
            return query

def build_custom_encoder():
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    encoder = BEVFormerEncoder(pc_range=pc_range,num_points_in_pillar=4)
    return encoder
# Only these fields are used
#bev_query: torch.Size([1, 40000, 256]) torch.float32
#key: torch.Size([6, 30825, 1, 256]) torch.float32
#value: torch.Size([6, 30825, 1, 256]) torch.float32
#bev_pos: torch.Size([1, 40000, 256]) torch.float32
#ref_2d: torch.Size([1, 40000, 1, 2]) torch.float32
#ref_3d: torch.Size([1, 4, 40000, 3]) torch.float32
#bev_h: 200
#bev_w: 200
#bev_pos: torch.Size([40000, 1, 256]) torch.float32
#spatial_shapes: torch.Size([4, 2]) torch.int64
#level_start_index: torch.Size([4]) torch.int64
#shift: torch.Size([1, 2]) torch.float32
#img_metas: [{'filename': ['./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg', './data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg', './data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg', './data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg', './data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg', './data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg'], 'ori_shape': [(900, 1600, 3), (900, 1600, 3), (900, 1600, 3), (900, 1600, 3), (900, 1600, 3), (900, 1600, 3)], 'img_shape': [(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)], 'lidar2img': [array([[ 1.24298977e+03,  8.40649523e+02,  3.27625534e+01,
    #     -3.54351139e+02],
    #    [-1.82012609e+01,  5.36798564e+02, -1.22553754e+03,
    #     -6.44707879e+02],
    #    [-1.17025046e-02,  9.98471159e-01,  5.40221896e-02,
    #     -4.25203639e-01],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]]), array([[ 1.36494654e+03, -6.19264860e+02, -4.03391641e+01,
    #     -4.61642859e+02],

    #     -5.88246117e+02],
    #    [ 9.24052925e-01, -3.82246554e-01, -3.70989150e-03,
    #     -4.64645142e-01],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]])], 'pad_shape': [(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)], 'scale_factor': 1.0, 'flip': False, 'pcd_horizontal_flip': False, 'pcd_vertical_flip': False, 'box_mode_3d': <Box3DMode.LIDAR: 0>, 'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 'img_norm_cfg': {'mean': array([103.53 , 116.28 , 123.675], dtype=float32), 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': False}, 'sample_idx': '3e8750f331d7499e9b5123e9eb70f2e2', 'prev_idx': '', 'next_idx': '3950bd41f74548429c0f7700ff3d8269', 'pcd_scale_factor': 1.0, 'pts_filename': './data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin', 'scene_token': 'fcbccedd61424f1b85dcbf8f897f9754', 'can_bus': array([ 6.00120214e+02,  1.64749078e+03,  0.00000000e+00, -9.68669702e-01,
    #    -9.68669702e-01, -9.68669702e-01, -9.68669702e-01, -6.06941519e-01,
    #    -7.63441180e-02,  9.87149385e+00, -2.10869126e-02, -1.24397185e-02,
    #    -2.30670013e-02,  8.56405970e+00,  0.00000000e+00,  0.00000000e+00,
    #     5.78155401e+00,  3.31258644e+02])}]
    
if __name__ == "__main__":
    layer = build_attention({'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1})
    print(layer)

    layer = build_attention({'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4})
    print(layer)

    layer = build_attention({'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points':
    8, 'num_levels': 4}, 'embed_dims': 256})
    print(layer)

    layer = build_feedforward_network({'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 512, 'num_fcs': 2, 'ffn_drop': 0.1, 'act_cfg': {'type': 'ReLU', 'inplace': True}})
    print(layer)

    layer = build_transformer_layer({'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')})
    print(layer)
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    encoder = BEVFormerEncoder(pc_range=pc_range,num_points_in_pillar=4)

    encoder.load_state_dict(torch.load("/tmp/encoder.pth"))
    
    with open("/tmp/input.pickle","rb") as f:
        inputs = pickle.load(f)
    
    print(inputs)
    bev_query = inputs["query"]
    inputs.pop("query")

    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = encoder(bev_query,**inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"encoder: {(end - start) * 1000}ms")
    
        
        
        