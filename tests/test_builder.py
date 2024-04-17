def test_build_positional_encoding():
    from src.utils.builder import build_positional_encoding
    cfg = {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}
    encoding = build_positional_encoding(cfg)


def test_build_attention():
    from src.utils.builder import build_attention
    cfg = {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}
    attention = build_attention(cfg)
    
    
def test_build_transformer_layer():
    from src.utils.builder import build_transformer_layer
    cfg1 = {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}
    layer = build_transformer_layer(cfg1)
   

def test_build_transformer_layer_sequence():
    from src.utils.builder import build_transformer_layer_sequence
    cfg = {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}
    encoder = build_transformer_layer_sequence(cfg)
    cfg = {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}
    decoder = build_transformer_layer_sequence(cfg)
   

def test_build_transformer():
    from src.utils.builder import build_transformer
    cfg = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    transformer = build_transformer(cfg)
   

def test_build_conv_layer():
    from src.utils.builder import build_conv_layer
    cfg = None
    in_channels = 3
    stem_channels = 64
    kernel_size = 7
    stride = 2
    padding = 3
    bias = False
    layer = build_conv_layer(cfg,in_channels,stem_channels,kernel_size=7,stride=2,padding=3,bias=False)
   

def test_build_norm_layer():
    from src.utils.builder import build_norm_layer
    cfg = {'type': 'BN2d', 'requires_grad': False}
    name, layer = build_norm_layer(cfg,num_features=64,postfix=1)
    

def test_build_backbone():
    from src.utils.builder import build_backbone
    cfg = {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': 4, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False}, 'norm_eval': True, 'style': 'caffe', 'dcn': {'type': 'DCNv2', 'deform_groups': 1, 'fallback_on_stride': False}, 'stage_with_dcn': (False, False, True, True)}
    resnet = build_backbone(cfg)
    

def test_build_neck():
    from src.utils.builder import build_neck
    cfg = {'type': 'FPN', 'in_channels': [512, 1024, 2048], 'out_channels': 256, 'start_level': 0, 'add_extra_convs': 'on_output', 'num_outs': 4, 'relu_before_extra_convs': True}
    build_neck(cfg)
    
    