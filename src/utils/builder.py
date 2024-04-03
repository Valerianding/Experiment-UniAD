import torch
from torch import nn

def build_norm_layer(cfg, num_features):
    cfg_ = cfg.copy()
    layer_type = cfg_['type']
    cfg_.pop('type')
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    assert layer_type == 'LN'
    layer = nn.LayerNorm(num_features, **cfg_).to("cuda")
    
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return layer

def build_attention(cfg):
    assert isinstance(cfg,dict)
    
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    if type == 'TemporalSelfAttention':
        from src.bevformer.temporal_self_attention import TemporalSelfAttention
        attention = TemporalSelfAttention(**cfg_).to("cuda")
        return attention
    elif type == 'SpatialCrossAttention':
        from src.bevformer.spatial_cross_attention import SpatialCrossAttention
        attention = SpatialCrossAttention(**cfg_).to("cuda")
        return attention
    elif type == 'MSDeformableAttention3D':
        from src.bevformer.spatial_cross_attention import MSDeformableAttention3D
        attention = MSDeformableAttention3D(**cfg_).to("cuda")
        return attention
    elif type == "MultiScaleDeformableAttention":
        from src.seg_head.multi_scale_deform_attn import MultiScaleDeformableAttention
        attention = MultiScaleDeformableAttention(**cfg_).to("cuda")
        return attention
    elif type == "MultiheadAttention":
        from src.seg_head.multi_head_attention import MultiheadAttention
        attention = MultiheadAttention(**cfg_).to("cuda")
        return attention
    else:
        assert False, f"{type} is not supported"

#{'type': 'ReLU', 'inplace': True}
def build_activation_layer(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "ReLU"
    layer = nn.ReLU(**cfg_).to("cuda")
    return layer
    
def build_dropout(cfg,default_args=None):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "Dropout"
    from src.utils.utils import Dropout
    dropout = Dropout(**cfg_).to("cuda")
    return dropout
    
#{'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 512, 'num_fcs': 2, 'ffn_drop': 0.1, 'act_cfg': {'type': 'ReLU', 'inplace': True}}
def build_feedforward_network(cfg,default_args=None):
    from src.modules.ffn import FFN
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "FFN"
    layer = FFN(**cfg_).to("cuda")
    return layer

def build_transformer_layer(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    if type == "BEVFormerLayer":
        from src.bevformer.custom_encoder import BEVFormerLayer
        layer = BEVFormerLayer(**cfg_).to("cuda")
        return layer
    elif type == "BaseTransformerLayer":
        from src.seg_head.transformer import BaseTransformerLayer
        layer = BaseTransformerLayer(**cfg_).to("cuda")
        return layer
    elif type == "DetrTransformerDecoderLayer":
        from src.seg_head.transformer import DetrTransformerDecoderLayer
        layer = DetrTransformerDecoderLayer(**cfg_).to("cuda")
        return layer
    else:
        assert False, f"{type} is not supported!"
    
def build_transformer_layer_sequence(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    if type == "DetrTransformerEncoder":
        from src.seg_head.transformer import DetrTransformerEncoder
        encoder = DetrTransformerEncoder(**cfg_).to("cuda")
        return encoder
    elif type == "DeformableDetrTransformerDecoder":
        from src.seg_head.transformer import DeformableDetrTransformerDecoder
        decoder = DeformableDetrTransformerDecoder(**cfg_).to("cuda")
        return decoder
    else:
        assert False
    
def build_transformer(cfg):
    cfg_  = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "SegDeformableTransformer", f"{type} is not supported!"
    from src.seg_head.seg_deformable_transformer import SegDeformableTransformer
    transformer = SegDeformableTransformer(**cfg_).to("cuda")
    return transformer

#{'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}
def build_positional_encoding(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "SinePositionalEncoding"
    from src.modules.positional_encoding import SinePositionalEncoding
    encoding = SinePositionalEncoding(**cfg_).to("cuda")
    return encoding


def get_transformer():
    cfg = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    transformer = build_transformer(cfg)
    return transformer
