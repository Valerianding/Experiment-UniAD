import torch
import copy
from torch import nn
import torch.nn.functional as F
from ffn import FFN
from seg_detr_head import SegDETRHead
from utils import inverse_sigmoid, bias_init_with_prob, constant_init, Linear
from builder import build_transformer

class PansegformerHead(SegDETRHead):
    def __init__(
            self,
            *args,
            bev_h,
            bev_w,
            canvas_size,
            pc_range,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            quality_threshold_things=0.25,
            quality_threshold_stuff=0.25,
            overlap_threshold_things=0.4,
            overlap_threshold_stuff=0.2,
            thing_transformer_head=dict(
                type='TransformerHead',  # mask decoder for things
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            stuff_transformer_head=dict(
                type='TransformerHead',  # mask decoder for stuff
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',
                              cls_cost=dict(type='ClassificationCost',
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                              iou_cost=dict(type='IoUCost',
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),
            ),
            **kwargs):
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = 0.1
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.fp16_enabled = False

        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_dec_things = thing_transformer_head['num_decoder_layers']
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']
        super(PansegformerHead, self).__init__(*args,
                                        transformer=transformer,
                                        train_cfg=train_cfg,
                                        **kwargs)
        
        # if train_cfg:
            # sampler_cfg = train_cfg['sampler_with_mask']
            # self.sampler_with_mask = build_sampler(sampler_cfg, context=self)
            # assigner_cfg = train_cfg['assigner_with_mask']
            # self.assigner_with_mask = build_assigner(assigner_cfg)
            # self.assigner_filter = build_assigner(
            #     dict(
            #         type='HungarianAssigner_filter',
            #         cls_cost=dict(type='FocalLossCost', weight=2.0),
            #         reg_cost=dict(type='BBoxL1Cost',
            #                       weight=5.0,
            #                       box_format='xywh'),
            #         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            #         max_pos=
            #         3  # Depends on GPU memory, setting it to 1, model can be trained on 1080Ti
            #     ), )
            # num = 0
        
        # self.loss_mask = build_loss(loss_mask)
        self.things_mask_head = build_transformer(thing_transformer_head)
        self.stuff_mask_head = build_transformer(stuff_transformer_head)
        self.count = 0
        
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things) # used in mask decoder
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things) # used in mask decoder
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff) # used in mask deocder
        
        
    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_thing_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches2:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
        
        
    def forward(self,bev_embed):
        _, bs, _ = bev_embed.shape

        # possible point 1
        mlvl_feats = [torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w ,-1)).permute(0, 3, 1, 2)]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))
        
        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            
            #positional_encoding needs to be build
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
            
        #query_embedding
        #self.transformer
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
        )
        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        # we should feed these to mask deocder.
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

    
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        outs = {
            'bev_embed': None if self.as_two_stage else bev_embed,
            'outputs_classes': outputs_classes,
            'outputs_coords': outputs_coords,
            'enc_outputs_class': enc_outputs_class if self.as_two_stage else None,
            'enc_outputs_coord': enc_outputs_coord.sigmoid() if self.as_two_stage else None,
            'args_tuple': args_tuple,
            'reference': reference,
        }
        return outs