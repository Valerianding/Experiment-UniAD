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



import torch
from src.utils.distributed import MMDistributedDataParallel
if __name__ == "__main__":
    from src.uniad_e2e import UniAD
    cfg = \
    {'filter_score_thresh': 0.35,
    'freeze_bev_encoder': True,
    'freeze_bn': True,
    'freeze_img_backbone': True,
    'freeze_img_neck': True,
    'gt_iou_threshold': 0.3,
    'img_backbone': {'dcn': {'deform_groups': 1,
                            'fallback_on_stride': False,
                            'type': 'DCNv2'},
                    'depth': 101,
                    'frozen_stages': 4,
                    'norm_cfg': {'requires_grad': False, 'type': 'BN2d'},
                    'norm_eval': True,
                    'num_stages': 4,
                    'out_indices': (1, 2, 3),
                    'stage_with_dcn': (False, False, True, True),
                    'style': 'caffe',
                    'type': 'ResNet'},
    'img_neck': {'add_extra_convs': 'on_output',
                'in_channels': [512, 1024, 2048],
                'num_outs': 4,
                'out_channels': 256,
                'relu_before_extra_convs': True,
                'start_level': 0,
                'type': 'FPN'},
    'loss_cfg': {'assigner': {'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0},
                            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            'reg_cost': {'type': 'BBox3DL1Cost', 'weight': 0.25},
                            'type': 'HungarianAssigner3DTrack'},
                'code_weights': [1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                0.2,
                                0.2],
                'loss_bbox': {'loss_weight': 0.25, 'type': 'L1Loss'},
                'loss_cls': {'alpha': 0.25,
                            'gamma': 2.0,
                            'loss_weight': 2.0,
                            'type': 'FocalLoss',
                            'use_sigmoid': True},
                'num_classes': 10,
                'type': 'ClipMatcher',
                'weight_dict': None},
    'mem_args': {'memory_bank_len': 4,
                'memory_bank_score_thresh': 0.0,
                'memory_bank_type': 'MemoryBank'},
    'motion_head': {'type': 'MotionHead', 'bev_h': 200, 'bev_w': 200, 'num_query': 300, 'num_classes': 10, 'predict_steps': 12, 'predict_modes': 6, 'embed_dims': 256, 'loss_traj': {'type': 'TrajLoss', 'use_variance': True, 'cls_loss_weight': 0.5, 'nll_loss_weight': 0.5, 'loss_weight_minade': 0.0, 'loss_weight_minfde': 0.25}, 'num_cls_fcs': 3, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'group_id_list': [[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]], 'num_anchor': 6, 'use_nonlinear_optimizer': True, 'anchor_info_path': './tests/weights/motion_anchor_infos_mode6.pkl', 'transformerlayers': {'type': 'MotionTransformerDecoder', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'embed_dims': 256, 'num_layers': 3, 'transformerlayers': {'type': 'MotionTransformerAttentionLayer', 'batch_first': True, 'attn_cfgs': [{'type': 'MotionDeformableAttention', 'num_steps': 12, 'embed_dims': 256, 'num_levels': 1, 'num_heads': 8, 'num_points': 4, 'sample_index': -1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('cross_attn', 'norm', 'ffn', 'norm')}}},
    'num_classes': 10,
    'num_query': 900,
    'occ_head':  {'type': 'OccHead', 'grid_conf': {'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'zbound': [-10.0, 10.0, 20.0]}, 'ignore_index': 255, 'bev_proj_dim': 256, 'bev_proj_nlayers': 4, 'attn_mask_thresh': 0.3, 'transformer_decoder': {'type': 'DetrTransformerDecoder', 'return_intermediate': True, 'num_layers': 5, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': {'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'attn_drop': 0.0, 'proj_drop': 0.0, 'dropout_layer': None, 'batch_first': False}, 'ffn_cfgs': {'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 2048, 'num_fcs': 2, 'act_cfg': {'type': 'ReLU', 'inplace': True}, 'ffn_drop': 0.0, 'dropout_layer': None, 'add_identity': True}, 'feedforward_channels': 2048, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}, 'init_cfg': None}, 'query_dim': 256, 'query_mlp_layers': 3, 'aux_loss_weight': 1.0, 'loss_mask': {'type': 'FieryBinarySegmentationLoss', 'use_top_k': True, 'top_k_ratio': 0.25, 'future_discount': 0.95, 'loss_weight': 5.0, 'ignore_index': 255}, 'loss_dice': {'type': 'DiceLossWithMasks', 'use_sigmoid': True, 'activate': True, 'reduction': 'mean', 'naive_dice': True, 'eps': 1.0, 'ignore_index': 255, 'loss_weight': 1.0}, 'pan_eval': True, 'test_seg_thresh': 0.1, 'test_with_track_score': True},
    'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    'planning_head': {'embed_dims': 256,
                    'loss_collision': [{'delta': 0.0,
                                        'type': 'CollisionLoss',
                                        'weight': 2.5},
                                        {'delta': 0.5,
                                        'type': 'CollisionLoss',
                                        'weight': 1.0},
                                        {'delta': 1.0,
                                        'type': 'CollisionLoss',
                                        'weight': 0.25}],
                    'loss_planning': {'type': 'PlanningLoss'},
                    'planning_eval': True,
                    'planning_steps': 6,
                    'type': 'PlanningHeadSingleMode',
                    'use_col_optim': True,
                    'with_adapter': True},
    'pretrained': None,
    'pts_bbox_head': {'as_two_stage': False,
                    'bbox_coder': {'max_num': 300,
                                    'num_classes': 10,
                                    'pc_range': [-51.2,
                                                -51.2,
                                                -5.0,
                                                51.2,
                                                51.2,
                                                3.0],
                                    'post_center_range': [-61.2,
                                                            -61.2,
                                                            -10.0,
                                                            61.2,
                                                            61.2,
                                                            10.0],
                                    'type': 'NMSFreeCoder',
                                    'voxel_size': [0.2, 0.2, 8]},
                    'bev_h': 200,
                    'bev_w': 200,
                    'fut_steps': 4,
                    'in_channels': 256,
                    'loss_bbox': {'loss_weight': 0.25, 'type': 'L1Loss'},
                    'loss_cls': {'alpha': 0.25,
                                    'gamma': 2.0,
                                    'loss_weight': 2.0,
                                    'type': 'FocalLoss',
                                    'use_sigmoid': True},
                    'loss_iou': {'loss_weight': 0.0, 'type': 'GIoULoss'},
                    'num_classes': 10,
                    'num_query': 900,
                    'past_steps': 4,
                    'positional_encoding': {'col_num_embed': 200,
                                            'num_feats': 128,
                                            'row_num_embed': 200,
                                            'type': 'LearnedPositionalEncoding'},
                    'sync_cls_avg_factor': True,
                    'transformer': {'decoder': {'num_layers': 6,
                                                'return_intermediate': True,
                                                'transformerlayers': {'attn_cfgs': [{'dropout': 0.1,
                                                                                        'embed_dims': 256,
                                                                                        'num_heads': 8,
                                                                                        'type': 'MultiheadAttention'},
                                                                                    {'embed_dims': 256,
                                                                                        'num_levels': 1,
                                                                                        'type': 'CustomMSDeformableAttention'}],
                                                                        'feedforward_channels': 512,
                                                                        'ffn_dropout': 0.1,
                                                                        'operation_order': ('self_attn',
                                                                                            'norm',
                                                                                            'cross_attn',
                                                                                            'norm',
                                                                                            'ffn',
                                                                                            'norm'),
                                                                        'type': 'DetrTransformerDecoderLayer'}
    ,
                                                'type': 'DetectionTransformerDecoder'},
                                    'embed_dims': 256,
                                    'encoder': {'num_layers': 6,
                                                'num_points_in_pillar': 4,
                                                'pc_range': [-51.2,
                                                                -51.2,
                                                                -5.0,
                                                                51.2,
                                                                51.2,
                                                                3.0],
                                                'return_intermediate': False,
                                                'transformerlayers': {'attn_cfgs': [{'embed_dims': 256,
                                                                                        'num_levels': 1,
                                                                                        'type': 'TemporalSelfAttention'},
                                                                                    {'deformable_attention': {'embed_dims': 256,
                                                                                                                'num_levels': 4,
                                                                                                                'num_points': 8,
                                                                                                                'type': 'MSDeformableAttention3D'},
                                                                                        'embed_dims': 256,
                                                                                        'pc_range': [-51.2,
                                                                                                    -51.2,
                                                                                                    -5.0,
                                                                                                    51.2,
                                                                                                    51.2,
                                                                                                    3.0],
                                                                                        'type': 'SpatialCrossAttention'}],
                                                                        'feedforward_channels': 512,
                                                                        'ffn_dropout': 0.1,
                                                                        'operation_order': ('self_attn',
                                                                                            'norm',
                                                                                            'cross_attn',
                                                                                            'norm',
                                                                                            'ffn',
                                                                                            'norm'),
                                                                        'type': 'BEVFormerLayer'},
                                                'type': 'BEVFormerEncoder'},
                                    'rotate_prev_bev': True,
                                    'type': 'PerceptionTransformer',
                                    'use_can_bus': True,
                                    'use_shift': True},
                    'type': 'BEVFormerTrackHead',
                    'with_box_refine': True},
    'qim_args': {'fp_ratio': 0.3,
                'merger_dropout': 0,
                'qim_type': 'QIMBase',
                'random_drop': 0.1,
                'update_query_pos': True},
    'queue_length': 3,
    'score_thresh': 0.4,
    'seg_head': {'as_two_stage': False,
                'bev_h': 200,
                'bev_w': 200,
                'canvas_size': (200, 200),
                'in_channels': 2048,
                'loss_bbox': {'loss_weight': 5.0, 'type': 'L1Loss'},
                'loss_cls': {'alpha': 0.25,
                            'gamma': 2.0,
                            'loss_weight': 2.0,
                            'type': 'FocalLoss',
                            'use_sigmoid': True},
                'loss_iou': {'loss_weight': 2.0, 'type': 'GIoULoss'},
                'loss_mask': {'loss_weight': 2.0, 'type': 'DiceLoss'},
                'num_classes': 4,
                'num_query': 300,
                'num_stuff_classes': 1,
                'num_things_classes': 3,
                'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                'positional_encoding': {'normalize': True,
                                        'num_feats': 128,
                                        'offset': -0.5,
                                        'type': 'SinePositionalEncoding'},
                'stuff_transformer_head': {'d_model': 256,
                                            'nhead': 8,
                                            'num_decoder_layers': 6,
                                            'self_attn': True,
                                            'type': 'SegMaskHead'},
                'sync_cls_avg_factor': True,
                'thing_transformer_head': {'d_model': 256,
                                            'nhead': 8,
                                            'num_decoder_layers': 4,
                                            'type': 'SegMaskHead'},
                'train_cfg': {'assigner': {'cls_cost': {'type': 'FocalLossCost',
                                                        'weight': 2.0},
                                            'iou_cost': {'iou_mode': 'giou',
                                                        'type': 'IoUCost',
                                                        'weight': 2.0},
                                            'reg_cost': {'box_format': 'xywh',
                                                        'type': 'BBoxL1Cost',
                                                        'weight': 5.0},
                                            'type': 'HungarianAssigner'},
                                'assigner_with_mask': {'cls_cost': {'type': 'FocalLossCost',
                                                                    'weight': 2.0},
                                                    'iou_cost': {'iou_mode': 'giou',
                                                                    'type': 'IoUCost',
                                                                    'weight': 2.0},
                                                    'mask_cost': {'type': 'DiceCost',
                                                                    'weight': 2.0},
                                                    'reg_cost': {'box_format': 'xywh',
                                                                    'type': 'BBoxL1Cost',
                                                                    'weight': 5.0},
                                                    'type': 'HungarianAssigner_multi_info'},
                                'sampler': {'type': 'PseudoSampler'},
                                'sampler_with_mask': {'type': 'PseudoSampler_segformer'}},
                'transformer': {'decoder': {'num_layers': 6,
                                            'return_intermediate': True,
                                            'transformerlayers': {'attn_cfgs': [{'dropout': 0.1,
                                                                                'embed_dims': 256,
                                                                                'num_heads': 8,
                                                                                'type': 'MultiheadAttention'},
                                                                                {'embed_dims': 256,
                                                                                'num_levels': 4,
                                                                                'type': 'MultiScaleDeformableAttention'}],
                                                                    'feedforward_channels': 512,
                                                                    'ffn_dropout': 0.1,
                                                                    'operation_order': ('self_attn',
                                                                                        'norm',
                                                                                        'cross_attn',
                                                                                        'norm',
                                                                                        'ffn',
                                                                                        'norm'),
                                                                    'type': 'DetrTransformerDecoderLayer'},
                                            'type': 'DeformableDetrTransformerDecoder'},
                                'encoder': {'num_layers': 6,
                                            'transformerlayers': {'attn_cfgs': {'embed_dims': 256,
                                                                                'num_levels': 4,
                                                                                'type': 'MultiScaleDeformableAttention'},
                                                                    'feedforward_channels': 512,
                                                                    'ffn_dropout': 0.1,
                                                                    'operation_order': ('self_attn',
                                                                                        'norm',
                                                                                        'ffn',
                                                                                        'norm'),
                                                                    'type': 'BaseTransformerLayer'},
                                            'type': 'DetrTransformerEncoder'},
                                'type': 'SegDeformableTransformer'},
                'type': 'PansegformerHead',
                'with_box_refine': True},
    'train_cfg': None,
    'type': 'UniAD',
    'use_grid_mask': True,
    'vehicle_id_list': [0, 1, 2, 3, 4, 6, 7],
    'video_test_mode': True}
    
    
    cfg.pop('type')
    model = UniAD(**cfg).to("cuda")

    try:
        uniad_dict = torch.load("./tests/weights/uniad.pth")
    except:
        print(f"load checkpoints failed! check if file exits")
        
    state_dict = model.state_dict()
    print("loading uniad weights")
    for name, param in uniad_dict.items():
        if name in state_dict:
            state_dict[name] = param
    
    #TODO: 补充加载输入和forward_test部分