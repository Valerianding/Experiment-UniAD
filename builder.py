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
        from temporal_self_attention import TemporalSelfAttention
        attention = TemporalSelfAttention(**cfg_).to("cuda")
        return attention
    elif type == 'SpatialCrossAttention':
        from spatial_cross_attention import SpatialCrossAttention
        attention = SpatialCrossAttention(**cfg_).to("cuda")
        return attention
    elif type == 'MSDeformableAttention3D':
        from spatial_cross_attention import MSDeformableAttention3D
        attention = MSDeformableAttention3D(**cfg_).to("cuda")
        return attention
    elif type == "MultiScaleDeformableAttention":
        from multi_scale_deform_attn import MultiScaleDeformableAttention
        attention = MultiScaleDeformableAttention(**cfg_).to("cuda")
        return attention
    else:
        assert False

#{'type': 'ReLU', 'inplace': True}
def build_activation_layer(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "ReLU"
    layer = nn.ReLU(**cfg_).to("cuda")
    return layer
    
def build_dropout(cfg,default_args=None):
    assert False
    
#{'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 512, 'num_fcs': 2, 'ffn_drop': 0.1, 'act_cfg': {'type': 'ReLU', 'inplace': True}}
def build_feedforward_network(cfg,default_args=None):
    from ffn import FFN
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
        from custom_encoder import BEVFormerLayer
        layer = BEVFormerLayer(**cfg_).to("cuda")
        return layer
    elif type == "BaseTransformerLayer":
        from transformer import BaseTransformerLayer
        layer = BaseTransformerLayer(**cfg_).to("cuda")
        return layer
    elif type == "DetrTransformerDecoderLayer":
        from transformer import DetrTransformerDecoderLayer
        layer = DetrTransformerDecoderLayer(**cfg_).to("cuda")
        return layer
    else:
        assert False, f"{type} is not supported!"
    
def build_transformer_layer_sequence(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    if type == "DetrTransformerEncoder":
        from transformer import DetrTransformerEncoder
        encoder = DetrTransformerEncoder(**cfg_).to("cuda")
        return encoder
    elif type == "DeformableDetrTransformerDecoder":
        from transformer import DeformableDetrTransformerDecoder
        decoder = DeformableDetrTransformerDecoder(**cfg_).to("cuda")
        return decoder
    else:
        assert False
    

def build_transformer(cfg):
    cfg_  = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "SegDeformableTransformer"
    return 0

#{'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}
def build_positional_encoding(cfg):
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "SinePositionalEncoding"
    from positional_encoding import SinePositionalEncoding
    encoding = SinePositionalEncoding(**cfg_).to("cuda")
    return encoding