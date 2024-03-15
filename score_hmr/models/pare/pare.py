# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch.nn as nn
from .backbone import *
from .head import PareHead
from .backbone.hrnet import hrnet_w32, hrnet_w48


class PARE(nn.Module):
    def __init__(
        self,
        num_joints=24,
        softmax_temp=1.0,
        num_features_smpl=64,
        backbone="resnet50",
        iterative_regression=False,
        iter_residual=False,
        num_iterations=3,
        shape_input_type="feats",  # 'feats.all_pose.shape.cam',
        pose_input_type="feats",  # 'feats.neighbor_pose_feats.all_pose.self_pose.neighbor_pose.shape.cam'
        pose_mlp_num_layers=1,
        shape_mlp_num_layers=1,
        pose_mlp_hidden_size=256,
        shape_mlp_hidden_size=256,
        use_keypoint_features_for_smpl_regression=False,
        use_heatmaps="",
        use_keypoint_attention=False,
        keypoint_attention_act="softmax",
        use_postconv_keypoint_attention=False,
        use_scale_keypoint_attention=False,
        use_final_nonlocal=None,
        use_branch_nonlocal=None,
        use_hmr_regression=False,
        use_coattention=False,
        num_coattention_iter=1,
        coattention_conv="simple",
        deconv_conv_kernel_size=4,
        use_upsampling=False,
        use_soft_attention=False,
        num_branch_iteration=0,
        branch_deeper=False,
        num_deconv_layers=3,
        num_deconv_filters=256,
        use_resnet_conv_hrnet=False,
        use_position_encodings=False,
        use_mean_camshape=False,
        use_mean_pose=False,
        init_xavier=False,
        use_cam=False,
    ):
        super(PARE, self).__init__()
        if backbone.startswith("hrnet"):
            backbone, use_conv = backbone.split("-")
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(
                pretrained=True, downsample=False, use_conv=(use_conv == "conv")
            )
        else:
            self.backbone = eval(backbone)(pretrained=True)

        self.head = PareHead(
            num_joints=num_joints,
            num_input_features=get_backbone_info(backbone)["n_output_channels"],
            softmax_temp=softmax_temp,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=[num_deconv_filters] * num_deconv_layers,
            num_deconv_kernels=[deconv_conv_kernel_size] * num_deconv_layers,
            num_features_smpl=num_features_smpl,
            final_conv_kernel=1,
            iterative_regression=iterative_regression,
            iter_residual=iter_residual,
            num_iterations=num_iterations,
            shape_input_type=shape_input_type,
            pose_input_type=pose_input_type,
            pose_mlp_num_layers=pose_mlp_num_layers,
            shape_mlp_num_layers=shape_mlp_num_layers,
            pose_mlp_hidden_size=pose_mlp_hidden_size,
            shape_mlp_hidden_size=shape_mlp_hidden_size,
            use_keypoint_features_for_smpl_regression=use_keypoint_features_for_smpl_regression,
            use_heatmaps=use_heatmaps,
            use_keypoint_attention=use_keypoint_attention,
            use_postconv_keypoint_attention=use_postconv_keypoint_attention,
            keypoint_attention_act=keypoint_attention_act,
            use_scale_keypoint_attention=use_scale_keypoint_attention,
            use_branch_nonlocal=use_branch_nonlocal,
            use_final_nonlocal=use_final_nonlocal,
            backbone=backbone,
            use_hmr_regression=use_hmr_regression,
            use_coattention=use_coattention,
            num_coattention_iter=num_coattention_iter,
            coattention_conv=coattention_conv,
            use_upsampling=use_upsampling,
            use_soft_attention=use_soft_attention,
            num_branch_iteration=num_branch_iteration,
            branch_deeper=branch_deeper,
            use_resnet_conv_hrnet=use_resnet_conv_hrnet,
            use_position_encodings=use_position_encodings,
            use_mean_camshape=use_mean_camshape,
            use_mean_pose=use_mean_pose,
            init_xavier=init_xavier,
        )
        self.use_cam = use_cam

    def forward(
        self,
        images,
        get_feats=False,
        gt_segm=None,
    ):
        features = self.backbone(images)
        hmr_output = self.head(features, gt_segm=gt_segm)
        # hmr_output is a dict with the following keys:
        # ['pred_segm_mask', 'pred_pose', 'pred_cam', 'pred_shape', 'pose_feats', 'cam_shape_feats']

        # Return only the image features.
        if get_feats:
            return {
                "pose_feats": hmr_output["pose_feats"],
                "cam_shape_feats": hmr_output["cam_shape_feats"],
            }

        return hmr_output