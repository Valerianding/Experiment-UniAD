import torch
from torch import nn
from ffn import FFN
from utils import Linear, Conv2d
from builder import build_activation_layer, build_positional_encoding
class SegDETRHead(nn.Module):
    def __init__(
            self,
            num_classes,
            num_things_classes,
            num_stuff_classes,
            in_channels,
            num_query=100,
            num_reg_fcs=2,
            transformer=None,
            sync_cls_avg_factor=False,
            positional_encoding=dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            loss_cls=dict(type='CrossEntropyLoss',
                          bg_cls_weight=0.1,
                          use_sigmoid=False,
                          loss_weight=1.0,
                          class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            train_cfg=dict(assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
            test_cfg=dict(max_per_img=100),
            init_cfg=None,
            **kwargs):
        
        # init_cfg -> None
        super(SegDETRHead,self).__init__()
        
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is SegDETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_things_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_things_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        
        self.num_query = num_query
        self.num_classes = num_classes
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_things_classes
        else:
            self.cls_out_channels = num_things_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        
        self.transformer = build_transformer(transformer)
        
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(self.in_channels,
                                 self.embed_dims,
                                 kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(self.embed_dims,
                           self.embed_dims,
                           self.num_reg_fcs,
                           self.act_cfg,
                           dropout=0.0,
                           add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)