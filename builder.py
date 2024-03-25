import torch
from torch import nn
from temporal_self_attention import TemporalSelfAttention
from spatial_cross_attention import SpatialCrossAttention, MSDeformableAttention3D
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
        attention = TemporalSelfAttention(**cfg_).to("cuda")
        return attention
    elif type == 'SpatialCrossAttention':
        attention = SpatialCrossAttention(**cfg_).to("cuda")
        return attention
    elif type == 'MSDeformableAttention3D':
        attention = MSDeformableAttention3D(**cfg_).to("cuda")
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
def build_feedforward_network(cfg):
    from ffn import FFN
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "FFN"
    layer = FFN(**cfg_).to("cuda")
    return layer

def build_transformer_layer(cfg):
    from encoder import BEVFormerLayer
    cfg_ = cfg.copy()
    type = cfg_['type']
    cfg_.pop('type')
    assert type == "BEVFormerLayer"
    layer = BEVFormerLayer(**cfg_).to("cuda")
    return layer