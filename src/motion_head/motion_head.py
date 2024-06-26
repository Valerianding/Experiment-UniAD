
import torch
import torch.nn as nn
import copy
from src.utils.utils import bivariate_gaussian_activation, norm_points, pos2posemb2d, anchor_coordinate_transform
from .base_motion_head import BaseMotionHead


class MotionHead(BaseMotionHead):
    """
    MotionHead module for a neural network, which predicts motion trajectories and is used in an autonomous driving task.

    Args:
        *args: Variable length argument list.
        predict_steps (int): The number of steps to predict motion trajectories.
        transformerlayers (dict): A dictionary defining the configuration of transformer layers.
        bbox_coder: An instance of a bbox coder to be used for encoding/decoding boxes.
        num_cls_fcs (int): The number of fully-connected layers in the classification branch.
        bev_h (int): The height of the bird's-eye-view map.
        bev_w (int): The width of the bird's-eye-view map.
        embed_dims (int): The number of dimensions to use for the query and key vectors in transformer layers.
        num_anchor (int): The number of anchor points.
        det_layer_num (int): The number of layers in the transformer model.
        group_id_list (list): A list of group IDs to use for grouping the classes.
        pc_range: The range of the point cloud.
        use_nonlinear_optimizer (bool): A boolean indicating whether to use a non-linear optimizer for training.
        anchor_info_path (str): The path to the file containing the anchor information.
        vehicle_id_list(list[int]): class id of vehicle class, used for filtering out non-vehicle objects
    """
    def __init__(self,
                 *args,
                 predict_steps=12,
                 transformerlayers=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 bev_h=30,
                 bev_w=30,
                 embed_dims=256,
                 num_anchor=6,
                 det_layer_num=6,
                 group_id_list=[],
                 pc_range=None,
                 use_nonlinear_optimizer=False,
                 anchor_info_path=None,
                 loss_traj=dict(),
                 num_classes=0,
                 vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
                 **kwargs):
        super(MotionHead, self).__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_cls_fcs - 1
        self.embed_dims = embed_dims        
        self.num_anchor = num_anchor
        self.num_anchor_group = len(group_id_list)
        
        # we merge the classes into groups for anchor assignment
        self.cls2group = [0 for i in range(num_classes)]
        for i, grouped_ids in enumerate(group_id_list):
            for gid in grouped_ids:
                self.cls2group[gid] = i
        self.cls2group = torch.tensor(self.cls2group)
        self.pc_range = pc_range
        self.predict_steps = predict_steps
        self.vehicle_id_list = vehicle_id_list
        
        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self.unflatten_traj = nn.Unflatten(3, (self.predict_steps, 5))
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._load_anchors(anchor_info_path)
        self._build_layers(transformerlayers, det_layer_num)
        self._init_layers()

   

    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        """Test function"""
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']
        
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        
        import pdb
        pdb.set_trace()
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)      
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
        #import time;s_time = time.time()
        #import pdb;pdb.set_trace()
        import time
        torch.cuda.synchronize()
        start = time.perf_counter()
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"motion-head: {(end - start) * 1000}")
        # print("motion_head infer time is {}".format(1000 * (e_time - s_time)))
        traj_results = self.get_trajs(outs_motion, track_boxes)
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion['track_scores'] = scores[None, :]
        labels[-1] = 0
        def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
            if len(labels) < 1:  # No other obj query except sdc query.
                return None

            # select vehicle query according to vehicle_id_list
            vehicle_mask = torch.zeros_like(labels)
            for veh_id in vehicle_id_list:
                vehicle_mask |=  labels == veh_id
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
            outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
            outs_motion['track_scores'] = outs_motion['track_scores'][:, vehicle_mask>0]
            return outs_motion
        
        outs_motion = filter_vehicle_query(outs_motion, labels, self.vehicle_id_list)
        
        # filter sdc query
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]
        outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]

        return traj_results, outs_motion

    def forward(self, 
                bev_embed, 
                track_query, 
                lane_query, 
                lane_query_pos, 
                track_bbox_results):
        """
        Applies forward pass on the model for motion prediction using bird's eye view (BEV) embedding, track query, lane query, and track bounding box results.

        Args:
        bev_embed (torch.Tensor): A tensor of shape (h*w, B, D) representing the bird's eye view embedding.
        track_query (torch.Tensor): A tensor of shape (B, num_dec, A_track, D) representing the track query.
        lane_query (torch.Tensor): A tensor of shape (N, M_thing, D) representing the lane query.
        lane_query_pos (torch.Tensor): A tensor of shape (N, M_thing, D) representing the position of the lane query.
        track_bbox_results (List[torch.Tensor]): A list of tensors containing the tracking bounding box results for each image in the batch.

        Returns:
        dict: A dictionary containing the following keys and values:
        - 'all_traj_scores': A tensor of shape (num_levels, B, A_track, num_points) with trajectory scores for each level.
        - 'all_traj_preds': A tensor of shape (num_levels, B, A_track, num_points, num_future_steps, 2) with predicted trajectories for each level.
        - 'valid_traj_masks': A tensor of shape (B, A_track) indicating the validity of trajectory masks.
        - 'traj_query': A tensor containing intermediate states of the trajectory queries.
        - 'track_query': A tensor containing the input track queries.
        - 'track_query_pos': A tensor containing the positional embeddings of the track queries.
        """
        
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]

        # extract the last frame of the track query
        track_query = track_query[:, -1]
        
        # encode the center point of the track query
        reference_points_track = self._extract_tracking_centers(
            track_bbox_results, self.pc_range)
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))  # B, A, D
        
        # construct the learnable query positional embedding
        # split and stack according to groups
        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)  # latent anchor (P*G, D)
        learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))

        # construct the agent level/scene-level query positional embedding 
        # (num_groups, num_anchor, 12, 2)
        # to incorporate the information of different groups and coordinates, and embed the headding and location informa878tion
        agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(num_groups, self.num_anchor, self.predict_steps, 2).detach()
        scene_level_ego_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=True)  # B, A, G, P ,12 ,2
        scene_level_offset_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=False)  # B, A, G, P ,12 ,2

        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)

        # we only use the last point of the anchor
        agent_level_embedding = self.agent_level_embedding_layer(
            pos2posemb2d(agent_level_norm[..., -1, :]))  # G, P, D
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
            pos2posemb2d(scene_level_ego_norm[..., -1, :]))  # B, A, G, P , D
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
            pos2posemb2d(scene_level_offset_norm[..., -1, :]))  # B, A, G, P , D

        batch_size, num_agents = scene_level_ego_embedding.shape[:2]
        agent_level_embedding = agent_level_embedding[None,None, ...].expand(batch_size, num_agents, -1, -1, -1)
        learnable_embed = learnable_query_pos[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)

        
        # save for latter, anchors
        # B, A, G, P ,12 ,2 -> B, A, P ,12 ,2
        scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)  

        # select class embedding
        # B, A, G, P , D-> B, A, P , D
        agent_level_embedding = self.group_mode_query_pos(
            track_bbox_results, agent_level_embedding)  
        scene_level_ego_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_ego_embedding)  # B, A, G, P , D-> B, A, P , D
        
        # B, A, G, P , D -> B, A, P , D
        scene_level_offset_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_offset_embedding)  
        learnable_embed = self.group_mode_query_pos(
            track_bbox_results, learnable_embed)  

        init_reference = scene_level_offset_anchors.detach()

        outputs_traj_scores = []
        outputs_trajs = []
        import time;s_time = time.time()
        inter_states, inter_references = self.motionformer(
            track_query,  # B, A_track, D
            lane_query,  # B, M, D
            track_query_pos=track_query_pos,
            lane_query_pos=lane_query_pos,
            track_bbox_results=track_bbox_results,
            bev_embed=bev_embed,
            reference_trajs=init_reference,
            traj_reg_branches=self.traj_reg_branches,
            traj_cls_branches=self.traj_cls_branches,
            # anchor embeddings 
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
            learnable_embed=learnable_embed,
            # anchor positional embeddings layers
            agent_level_embedding_layer=self.agent_level_embedding_layer,
            scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
            scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
            spatial_shapes=torch.tensor(
                [[self.bev_h, self.bev_w]], device=device),
            level_start_index=torch.tensor([0], device=device))
        e_time = time.time()
        torch.cuda.synchronize()
        print("motionformer latency is {}".format(1000 * (e_time - s_time)))

        for lvl in range(inter_states.shape[0]):
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)
            
            # we use cumsum trick here to get the trajectory 
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            for bs in range(tmp.shape[0]):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])
            outputs_trajs.append(tmp)
        outputs_traj_scores = torch.stack(outputs_traj_scores)
        outputs_trajs = torch.stack(outputs_trajs)

        B, A_track, D = track_query.shape
        valid_traj_masks = track_query.new_ones((B, A_track)) > 0
        outs = {
            'all_traj_scores': outputs_traj_scores,
            'all_traj_preds': outputs_trajs,
            'valid_traj_masks': valid_traj_masks,
            'traj_query': inter_states,
            'track_query': track_query,
            'track_query_pos': track_query_pos,
        }

        return outs

    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        """
        Group mode query positions based on the input bounding box results.
        
        Args:
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
            mode_query_pos (torch.Tensor): A tensor of shape (B, A, G, P, D) representing the mode query positions.
        
        Returns:
            torch.Tensor: A tensor of shape (B, A, P, D) representing the classified mode query positions.
        """
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)
        # TODO: vectorize this
        # group the embeddings based on the class
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            grouped_label = self.cls2group[label]
            grouped_mode_query_pos = []
            for j in range(agent_num):
                grouped_mode_query_pos.append(
                    mode_query_pos[i, j, grouped_label[j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)

   
    def get_trajs(self, preds_dicts, bbox_results):
        """
        Generates trajectories from the prediction results, bounding box results.
        
        Args:
            preds_dicts (tuple[list[dict]]): A tuple containing lists of dictionaries with prediction results.
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        
        Returns:
            List[dict]: A list of dictionaries containing decoded bounding boxes, scores, and labels after non-maximum suppression.
        """
        num_samples = len(bbox_results)
        num_layers = preds_dicts['all_traj_preds'].shape[0]
        ret_list = []
        for i in range(num_samples):
            preds = dict()
            for j in range(num_layers):
                subfix = '_' + str(j) if j < (num_layers - 1) else ''
                traj = preds_dicts['all_traj_preds'][j, i]
                traj_scores = preds_dicts['all_traj_scores'][j, i]

                traj_scores, traj = traj_scores.cpu(), traj.cpu()
                preds['traj' + subfix] = traj
                preds['traj_scores' + subfix] = traj_scores
            ret_list.append(preds)
        return ret_list
