from src.utils.builder import get_transformer

def bev_compile():
    import pickle
    import time
    with open("./tests/inputs/bev_encoder.pickle","rb") as f:
        bev_encode_input = pickle.load(f)

    bev_queries = bev_encode_input["query"]
    feat_flatten = bev_encode_input["key"]
    feat_flatten = bev_encode_input["value"]
    bev_h = bev_encode_input["bev_h"]
    bev_w = bev_encode_input["bev_w"]
    bev_pos = bev_encode_input["bev_pos"]
    spatial_shapes = bev_encode_input["spatial_shapes"]
    level_start_index = bev_encode_input["level_start_index"]
    prev_bev = bev_encode_input["prev_bev"]
    shift = bev_encode_input["shift"]
    img_metas = bev_encode_input["img_metas"]
    
    assert bev_h == 200
    
    encoder = build_bev_encoder()
    
    encoder.eval()
    
    import torch._dynamo
    torch._dynamo.reset()
    opt_encoder = torch.compile(encoder,mode="max-autotune")
    
    
    with torch.no_grad():
        for _ in range(100): 
            torch.cuda.synchronize()
            start = time.perf_counter() 
            output = opt_encoder(
                bev_queries,
                feat_flatten,
                feat_flatten,
                bev_h = bev_h,
                bev_w = bev_w,
                bev_pos = bev_pos,
                spatial_shapes = spatial_shapes,
                level_start_index = level_start_index,
                prev_bev = prev_bev,
                shift = shift,
                img_metas = img_metas,
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            print(f"encoder: {(end - start) * 1000}ms")


def img_backbone_compile():
    import pickle
    import time
    
    with open("./tests/inputs/img_backbone.pickle","rb") as file:
        img = pickle.load(file)
        
    resnet_config = {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': 4, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False}, 'norm_eval': True, 'style': 'caffe', 'dcn': {'type': 'DCNv2', 'deform_groups': 1}, 'stage_with_dcn': (False, False, True, True)}
    resnet = build_backbone(resnet_config)
    
    resnet.eval()
    
    import torch._dynamo
    torch._dynamo.reset()
    
    opt_resnet = torch.compile(resnet,mode="reduce-overhead")
    
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            img_feats = opt_resnet(img)
            torch.cuda.synchronize()
            end = time.perf_counter()
            print(f"resnet: {(end - start) * 1000}ms")

def seg_deform_transformer_compile():
    transformer = {}
if __name__ == "__main__":
    # from src.seg_head.seg_detr_head import SegDETRHead
    # train_cfg = {'assigner': {'type': 'HungarianAssigner', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}}, 'assigner_with_mask': {'type': 'HungarianAssigner_multi_info', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}, 'mask_cost': {'type': 'DiceCost', 'weight': 2.0}}, 'sampler': {'type': 'PseudoSampler'}, 'sampler_with_mask': {'type': 'PseudoSampler_segformer'}}
    # transformer = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    # kwargs = {'num_query': 300, 'num_classes': 4, 'num_things_classes': 3, 'num_stuff_classes': 1, 'in_channels': 2048, 'sync_cls_avg_factor': True, 'positional_encoding': {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 5.0}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 2.0}}
    # head = SegDETRHead(train_cfg=train_cfg,
    #             transformer=transformer,  
    #             **kwargs)
    #{'type': 'SegMaskHead', 'd_model': 256, 'nhead': 8, 'num_decoder_layers': 4}
    #{'type': 'SegMaskHead', 'd_model': 256, 'nhead': 8, 'num_decoder_layers': 6, 'self_attn': True}
    import pdb
    import torch
    from src.utils.builder import build_norm_layer, build_backbone, build_bev_encoder
    # cfg = {'type': 'BN2d', 'requires_grad': False}
    # name, layer = build_norm_layer(cfg,num_features=64,postfix=1)
    # resnet_config = {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': 4, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False}, 'norm_eval': True, 'style': 'caffe', 'dcn': {'type': 'DCNv2', 'deform_groups': 1}, 'stage_with_dcn': (False, False, True, True)}
    # resnet = build_backbone(resnet_config)
    # pdb.set_trace()
    
    img_backbone_compile()
    