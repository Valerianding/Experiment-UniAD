def test_build_positional_encoding():
    from src.utils.builder import build_positional_encoding
    cfg = {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}
    encoding = build_positional_encoding(cfg)
    return 0

def test_build_attention():
    from src.utils.builder import build_attention
    cfg = {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}
    attention = build_attention(cfg)
    return 0
    
def test_build_transformer_layer():
    from src.utils.builder import build_transformer_layer
    cfg1 = {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}
    layer = build_transformer_layer(cfg1)
    return 0

def test_build_transformer_layer_sequence():
    from src.utils.builder import build_transformer_layer_sequence
    cfg = {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}
    encoder = build_transformer_layer_sequence(cfg)
    cfg = {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}
    decoder = build_transformer_layer_sequence(cfg)
    return 0

def test_build_transformer():
    from src.utils.builder import build_transformer
    cfg = {'type': 'PerceptionTransformer', 'rotate_prev_bev': True, 'use_shift': True, 'use_can_bus': True, 'embed_dims': 256, 'encoder': {'type': 'BEVFormerEncoder', 'num_layers': 6, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'num_points_in_pillar': 4, 'return_intermediate': False, 'transformerlayers': {'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DetectionTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'CustomMSDeformableAttention', 'embed_dims': 256, 'num_levels': 1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    transformer = build_transformer(cfg)
    return 0


    