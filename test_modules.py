
def test_transformer_layer_sequence():
    from transformer import TransformerLayerSequence
    transformerlayers = {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}
    layerSequence = TransformerLayerSequence(transformerlayers=transformerlayers,num_layers=6)
    return 0