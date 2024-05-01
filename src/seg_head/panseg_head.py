import torch
import copy
from torch import nn
import torch.nn.functional as F
from src.modules.ffn import FFN
from src.seg_head.seg_detr_head import SegDETRHead
from src.seg_head.seg_utils import IOU,bbox_cxcywh_to_xyxy
from src.utils.utils import inverse_sigmoid, bias_init_with_prob, constant_init, Linear
from src.utils.builder import build_transformer

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
    
    def _get_bboxes_single(self,
                        cls_score,
                        bbox_pred,
                        img_shape,
                        scale_factor,
                        rescale=False):
        """
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # exclude background
        # if self.loss_cls.use_sigmoid:
        if True:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels
        
    def get_bboxes(
            self,
            all_cls_scores,
            all_bbox_preds,
            enc_cls_scores,
            enc_bbox_preds,
            args_tuple,
            reference,
            img_metas,
            rescale=False,
        ):
        """
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        seg_list = []
        stuff_score_list = []
        panoptic_list = []
        bbox_list = []
        labels_list = []
        drivable_list = []
        lane_list = []
        lane_score_list = []
        score_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            # img_shape = img_metas[img_id]['img_shape']
            # ori_shape = img_metas[img_id]['ori_shape']
            # scale_factor = img_metas[img_id]['scale_factor']
            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            index, bbox, labels = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            thing_query = query[i:i + 1, index, :]
            thing_query_pos = query_pos[i:i + 1, index, :]
            joint_query = torch.cat([
                thing_query, self.stuff_query.weight[None, :, :self.embed_dims]
            ], 1)

            stuff_query_pos = self.stuff_query.weight[None, :,
                                                      self.embed_dims:]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, :-self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl)
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, -self.num_stuff_classes:],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl)

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # BS, NQ, N_head,LEN

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](
                stuff_query).sigmoid().reshape(-1)

            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            mask_pred = F.interpolate(mask_pred.unsqueeze(0),
                                      size=ori_shape[:2],
                                      mode='bilinear').squeeze(0)

            masks_all = mask_pred
            score_list.append(masks_all)
            drivable_list.append(masks_all[-1] > 0.5)
            masks_all = masks_all[:-self.num_stuff_classes]
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1
            # scores_all = torch.cat([bbox[:, -1], scores_stuff], 0)
            # bboxes_all = torch.cat([bbox, torch.zeros([self.num_stuff_classes, 5], device=labels.device)], 0)
            # labels_all = torch.cat([labels, torch.arange(self.num_things_classes, self.num_things_classes+self.num_stuff_classes).to(labels.device)], 0)
            scores_all = bbox[:, -1]
            bboxes_all = bbox
            labels_all = labels

            ## mask wise merging
            seg_scores = (masks_all * seg_all.float()).sum(
                (1, 2)) / sum_seg_all
            scores_all *= (seg_scores**2)

            scores_all, index = torch.sort(scores_all, descending=True)

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            # MDS: select things for instance segmeantion
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]
            
            stuff_score_list.append(scores_st)

            results = torch.zeros((2, *mask_pred.shape[-2:]),
                                  device=mask_pred.device).to(torch.long)
            id_unique = 1
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            lane_score =  torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(mask_pred.dtype)
            for i, scores in enumerate(scores_all):
                # MDS: things and sutff have different threholds may perform a little bit better
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)
                results[0, _mask] = labels_all[i]
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique
                    id_unique += 1
            file_name = img_metas[img_id]['pts_filename'].split('/')[-1].split('.')[0]
            panoptic_list.append(
                (results.permute(1, 2, 0).cpu().numpy(), file_name, ori_shape))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)
        results = []
        for i in range(len(img_metas)):
            results.append({
                'bbox': bbox_list[i],
                'segm': seg_list[i],
                'labels': labels_list[i],
                'panoptic': panoptic_list[i],
                'drivable': drivable_list[i],
                'score_list': score_list[i],
                'lane': lane_list[i],
                'lane_score': lane_score_list[i],
                'stuff_score_list' : stuff_score_list[i],
            })
        return results
    
    
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
            
        
        # print(f"mlvl: {mlvl_positional_encodings}")
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
        # print(f"hs2:{hs}")
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
    
    def forward_test(self,
                    pts_feats=None,
                    gt_lane_labels=None,
                    gt_lane_masks=None,
                    img_metas=None,
                    rescale=False):
        bbox_list = [dict() for i in range(len(img_metas))]
        
        # save the statics for correctness
        torch.cuda.synchronize()
        import time
        start = time.perf_counter()
        pred_seg_dict = self(pts_feats)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        print(f"seg-head: {(end - start) * 1000}ms")
        results = self.get_bboxes(pred_seg_dict['outputs_classes'],
                                           pred_seg_dict['outputs_coords'],
                                           pred_seg_dict['enc_outputs_class'],
                                           pred_seg_dict['enc_outputs_coord'],
                                           pred_seg_dict['args_tuple'],
                                           pred_seg_dict['reference'],
                                           img_metas,
                                           rescale=rescale)
        
        with torch.no_grad():
            drivable_pred = results[0]['drivable']
            drivable_gt = gt_lane_masks[0][0, -1]
            drivable_iou, drivable_intersection, drivable_union = IOU(drivable_pred.view(1, -1), drivable_gt.view(1, -1))

            lane_pred = results[0]['lane']
            lanes_pred = (results[0]['lane'].sum(0) > 0).int()
            lanes_gt = (gt_lane_masks[0][0][:-1].sum(0) > 0).int()
            lanes_iou, lanes_intersection, lanes_union = IOU(lanes_pred.view(1, -1), lanes_gt.view(1, -1))

            divider_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 0].sum(0) > 0).int()
            crossing_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 1].sum(0) > 0).int()
            contour_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 2].sum(0) > 0).int()
            divider_iou, divider_intersection, divider_union = IOU(lane_pred[0].view(1, -1), divider_gt.view(1, -1))
            crossing_iou, crossing_intersection, crossing_union = IOU(lane_pred[1].view(1, -1), crossing_gt.view(1, -1))
            contour_iou, contour_intersection, contour_union = IOU(lane_pred[2].view(1, -1), contour_gt.view(1, -1))

            ret_iou = {'drivable_intersection': drivable_intersection,
                       'drivable_union': drivable_union,
                       'lanes_intersection': lanes_intersection,
                       'lanes_union': lanes_union,
                       'divider_intersection': divider_intersection,
                       'divider_union': divider_union,
                       'crossing_intersection': crossing_intersection,
                       'crossing_union': crossing_union,
                       'contour_intersection': contour_intersection,
                       'contour_union': contour_union,
                       'drivable_iou': drivable_iou,
                       'lanes_iou': lanes_iou,
                       'divider_iou': divider_iou,
                       'crossing_iou': crossing_iou,
                       'contour_iou': contour_iou}
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
            result_dict['args_tuple'] = pred_seg_dict['args_tuple']

        return bbox_list