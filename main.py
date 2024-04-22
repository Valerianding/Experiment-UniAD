# from src.utils.builder import get_transformer

# def bev_compile():
#     import pickle
#     import time
#     with open("./tests/inputs/bev_encoder.pickle","rb") as f:
#         bev_encode_input = pickle.load(f)

#     bev_queries = bev_encode_input["query"]
#     feat_flatten = bev_encode_input["key"]
#     feat_flatten = bev_encode_input["value"]
#     bev_h = bev_encode_input["bev_h"]
#     bev_w = bev_encode_input["bev_w"]
#     bev_pos = bev_encode_input["bev_pos"]
#     spatial_shapes = bev_encode_input["spatial_shapes"]
#     level_start_index = bev_encode_input["level_start_index"]
#     prev_bev = bev_encode_input["prev_bev"]
#     shift = bev_encode_input["shift"]
#     img_metas = bev_encode_input["img_metas"]
    
#     assert bev_h == 200
    
#     encoder = build_bev_encoder()
    
#     encoder.eval()
    
#     import torch._dynamo
#     torch._dynamo.reset()
#     opt_encoder = torch.compile(encoder,mode="max-autotune")
    
    
#     with torch.no_grad():
#         for _ in range(100): 
#             torch.cuda.synchronize()
#             start = time.perf_counter() 
#             output = opt_encoder(
#                 bev_queries,
#                 feat_flatten,
#                 feat_flatten,
#                 bev_h = bev_h,
#                 bev_w = bev_w,
#                 bev_pos = bev_pos,
#                 spatial_shapes = spatial_shapes,
#                 level_start_index = level_start_index,
#                 prev_bev = prev_bev,
#                 shift = shift,
#                 img_metas = img_metas,
#             )
#             torch.cuda.synchronize()
#             end = time.perf_counter()
#             print(f"encoder: {(end - start) * 1000}ms")


# def img_backbone_compile():
#     import pickle
#     import time
    
#     with open("./tests/inputs/img_backbone.pickle","rb") as file:
#         img = pickle.load(file)
        
#     resnet_config = {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': 4, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False}, 'norm_eval': True, 'style': 'caffe', 'dcn': {'type': 'DCNv2', 'deform_groups': 1}, 'stage_with_dcn': (False, False, True, True)}
#     resnet = build_backbone(resnet_config)
    
#     resnet.eval()
    
#     # import torch._dynamo
#     torch._dynamo.reset()
    
#     opt_resnet = torch.compile(resnet,mode="reduce-overhead")
    
#     with torch.no_grad():
#         for _ in range(100):
#             torch.cuda.synchronize()
#             start = time.perf_counter()
#             img_feats = opt_resnet(img)
#             torch.cuda.synchronize()
#             end = time.perf_counter()
#             print(f"resnet: {(end - start) * 1000}ms")

# def seg_deform_transformer_compile():
#     transformer = {}
    

# if __name__ == "__main__":
#     # from src.seg_head.seg_detr_head import SegDETRHead
#     # train_cfg = {'assigner': {'type': 'HungarianAssigner', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}}, 'assigner_with_mask': {'type': 'HungarianAssigner_multi_info', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}, 'mask_cost': {'type': 'DiceCost', 'weight': 2.0}}, 'sampler': {'type': 'PseudoSampler'}, 'sampler_with_mask': {'type': 'PseudoSampler_segformer'}}
#     # transformer = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
#     # kwargs = {'num_query': 300, 'num_classes': 4, 'num_things_classes': 3, 'num_stuff_classes': 1, 'in_channels': 2048, 'sync_cls_avg_factor': True, 'positional_encoding': {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 5.0}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 2.0}}
#     # head = SegDETRHead(train_cfg=train_cfg,
#     #             transformer=transformer,  
#     #             **kwargs)
#     #{'type': 'SegMaskHead', 'd_model': 256, 'nhead': 8, 'num_decoder_layers': 4}
#     #{'type': 'SegMaskHead', 'd_model': 256, 'nhead': 8, 'num_decoder_layers': 6, 'self_attn': True}
#     import pdb
#     import torch
#     from src.utils.builder import build_norm_layer, build_backbone, build_bev_encoder
#     # cfg = {'type': 'BN2d', 'requires_grad': False}
#     # name, layer = build_norm_layer(cfg,num_features=64,postfix=1)
#     # resnet_config = {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': 4, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False}, 'norm_eval': True, 'style': 'caffe', 'dcn': {'type': 'DCNv2', 'deform_groups': 1}, 'stage_with_dcn': (False, False, True, True)}
#     # resnet = build_backbone(resnet_config)
#     # pdb.set_trace()
#     # img_backbone_compile()
#     cfg = {'as_two_stage': False,
#         'bev_h': 200,
#         'bev_w': 200,
#         'canvas_size': (200, 200),
#         'in_channels': 2048,
#         'loss_bbox': {'loss_weight': 5.0, 'type': 'L1Loss'},
#         'loss_cls': {'alpha': 0.25,
#                     'gamma': 2.0,
#                     'loss_weight': 2.0,
#                     'type': 'FocalLoss',
#                     'use_sigmoid': True},
#         'loss_iou': {'loss_weight': 2.0, 'type': 'GIoULoss'},
#         'loss_mask': {'loss_weight': 2.0, 'type': 'DiceLoss'},
#         'num_classes': 4,
#         'num_query': 300,
#         'num_stuff_classes': 1,
#         'num_things_classes': 3,
#         'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
#         'positional_encoding': {'normalize': True,
#                                 'num_feats': 128,
#                                 'offset': -0.5,
#                                 'type': 'SinePositionalEncoding'},
#         'stuff_transformer_head': {'d_model': 256,
#                                     'nhead': 8,
#                                     'num_decoder_layers': 6,
#                                     'self_attn': True,
#                                     'type': 'SegMaskHead'},
#         'sync_cls_avg_factor': True,
#         'thing_transformer_head': {'d_model': 256,
#                                     'nhead': 8,
#                                     'num_decoder_layers': 4,
#                                     'type': 'SegMaskHead'},
#         'train_cfg': {'assigner': {'cls_cost': {'type': 'FocalLossCost',
#                                                 'weight': 2.0},
#                                     'iou_cost': {'iou_mode': 'giou',
#                                                 'type': 'IoUCost',
#                                                 'weight': 2.0},
#                                     'reg_cost': {'box_format': 'xywh',
#                                                 'type': 'BBoxL1Cost',
#                                                 'weight': 5.0},
#                                     'type': 'HungarianAssigner'},
#                     'assigner_with_mask': {'cls_cost': {'type': 'FocalLossCost',
#                                                         'weight': 2.0},
#                                             'iou_cost': {'iou_mode': 'giou',
#                                                         'type': 'IoUCost',
#                                                         'weight': 2.0},
#                                             'mask_cost': {'type': 'DiceCost',
#                                                             'weight': 2.0},
#                                             'reg_cost': {'box_format': 'xywh',
#                                                         'type': 'BBoxL1Cost',
#                                                         'weight': 5.0},
#                                             'type': 'HungarianAssigner_multi_info'},
#                     'sampler': {'type': 'PseudoSampler'},
#                     'sampler_with_mask': {'type': 'PseudoSampler_segformer'}},
#         'transformer': {'decoder': {'num_layers': 6,
#                                     'return_intermediate': True,
#                                     'transformerlayers': {'attn_cfgs': [{'dropout': 0.1,
#                                                                         'embed_dims': 256,
#                                                                         'num_heads': 8,
#                                                                         'type': 'MultiheadAttention'},
#                                                                         {'embed_dims': 256,
#                                                                         'num_levels': 4,
#                                                                         'type': 'MultiScaleDeformableAttention'}],
#                                                         'feedforward_channels': 512,
#                                                         'ffn_dropout': 0.1,
#                                                         'operation_order': ('self_attn',
#                                                                             'norm',
#                                                                             'cross_attn',
#                                                                             'norm',
#                                                                             'ffn',
#                                                                             'norm'),
#                                                         'type': 'DetrTransformerDecoderLayer'},
#                                     'type': 'DeformableDetrTransformerDecoder'},
#                         'encoder': {'num_layers': 6,
#                                     'transformerlayers': {'attn_cfgs': {'embed_dims': 256,
#                                                                         'num_levels': 4,
#                                                                         'type': 'MultiScaleDeformableAttention'},
#                                                         'feedforward_channels': 512,
#                                                         'ffn_dropout': 0.1,
#                                                         'operation_order': ('self_attn',
#                                                                             'norm',
#                                                                             'ffn',
#                                                                             'norm'),
#                                                         'type': 'BaseTransformerLayer'},
#                                     'type': 'DetrTransformerEncoder'},
#                         'type': 'SegDeformableTransformer'},
#         'type': 'PansegformerHead',
#         'with_box_refine': True}
#     from src.seg_head.panseg_head import PansegformerHead
#     pdb.set_trace()
#     head = PansegformerHead(**cfg).to("cuda")
    
#     import pickle
#     with open("./tests/inputs/seg_head.pickle","rb") as f:
#         seg_head_input = pickle.load(f)
        
#     bev_embed = seg_head_input['pts_feats']
#     gt_lane_labels = seg_head_input['gt_lane_labels']
#     gt_lane_masks = seg_head_input['gt_lane_masks']
#     img_metas = seg_head_input['img_metas']
#     rescale = seg_head_input['rescale']
    
#     import time
#     for _ in range(100):
#         torch.cuda.synchronize()
#         start = time.perf_counter()
#         results_seg = head.forward_test(bev_embed,gt_lane_labels,gt_lane_masks,img_metas,rescale)
#         torch.cuda.synchronize()
#         end = time.perf_counter()
#         print(f"seg-head:{(end - start) * 1000}ms")

if __name__ == "__main__":
    # cfg = {'type': 'PerceptionTransformer', 'rotate_prev_bev': True, 'use_shift': True, 'use_can_bus': True, 'embed_dims': 256, 'encoder': {'type': 'BEVFormerEncoder', 'num_layers': 6, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'num_points_in_pillar': 4, 'return_intermediate': False, 'transformerlayers': {'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DetectionTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'CustomMSDeformableAttention', 'embed_dims': 256, 'num_levels': 1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    # from src.track_head.transformer import PerceptionTransformer
    # transformer = PerceptionTransformer(**cfg)
    from src.track_head.track_head import BEVFormerTrackHead
    from src.utils.builder import build_head
    cfg = {'type': 'BEVFormerTrackHead', 'bev_h': 200, 'bev_w': 200, 'num_query': 900, 'num_classes': 10, 'in_channels': 256, 'sync_cls_avg_factor': True, 'with_box_refine': True, 'as_two_stage': False, 'past_steps': 4, 'fut_steps': 4, 'transformer': {'type': 'PerceptionTransformer', 'rotate_prev_bev': True, 'use_shift': True, 'use_can_bus': True, 'embed_dims': 256, 'encoder': {'type': 'BEVFormerEncoder', 'num_layers': 6, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'num_points_in_pillar': 4, 'return_intermediate': False, 'transformerlayers': {'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DetectionTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'CustomMSDeformableAttention', 'embed_dims': 256, 'num_levels': 1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}, 'bbox_coder': {'type': 'NMSFreeCoder', 'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'max_num': 300, 'voxel_size': [0.2, 0.2, 8], 'num_classes': 10}, 'positional_encoding': {'type': 'LearnedPositionalEncoding', 'num_feats': 128, 'row_num_embed': 200, 'col_num_embed': 200}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 0.25}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 0.0}, 'train_cfg': None, 'test_cfg': None}
    head = build_head(cfg)
    