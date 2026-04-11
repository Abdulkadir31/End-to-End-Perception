from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)










@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomDetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False


    def forward(self,
            query,
            *args,
            reference_points=None,
            reg_branches=None,
            key_padding_mask=None,
            **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []




        # for layer in self.operation_order:
        #     # temporal self attention
        #     if layer == 'self_attn':

        #         query = self.attentions[attn_index](
        #             query,
        #             prev_bev,
        #             prev_bev,
        #             identity if self.pre_norm else None,
        #             query_pos=bev_pos,
        #             key_pos=bev_pos,
        #             attn_mask=attn_masks[attn_index],
        #             key_padding_mask=query_key_padding_mask,
        #             reference_points=ref_2d,
        #             spatial_shapes=torch.tensor(
        #                 [[bev_h, bev_w]], device=query.device),
        #             level_start_index=torch.tensor([0], device=query.device),
        #             **kwargs)
        #         attn_index += 1
        #         identity = query

        #     elif layer == 'norm':
        #         query = self.norms[norm_index](query)
        #         norm_index += 1

        #     # spaital cross attention
        #     elif layer == 'cross_attn':
        #         query = self.attentions[attn_index](
        #             query,
        #             key,
        #             value,
        #             identity if self.pre_norm else None,
        #             query_pos=query_pos,
        #             key_pos=key_pos,
        #             reference_points=ref_3d,
        #             reference_points_cam=reference_points_cam,
        #             mask=mask,
        #             attn_mask=attn_masks[attn_index],
        #             key_padding_mask=key_padding_mask,
        #             spatial_shapes=spatial_shapes,
        #             level_start_index=level_start_index,
        #             **kwargs)
        #         attn_index += 1
        #         identity = query

        #     elif layer == 'ffn':
        #         query = self.ffns[ffn_index](
        #             query, identity if self.pre_norm else None)
        #         ffn_index += 1





        for lid, _ in enumerate(self.layers):
            print('##############Here###########')
            print(layer)
            print('##############Here###########')
            print(lid)
            print('##############Here###########')
            
            
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
        #     output = layer(
        #         output,
        #         *args,
        #         reference_points=reference_points_input,
        #         key_padding_mask=key_padding_mask,
        #         **kwargs)

        #     output = output.permute(1, 0, 2)
        #     if reg_branches is not None:
        #         tmp = reg_branches[lid](output)

        #         assert reference_points.shape[-1] == 3

        #         new_reference_points = torch.zeros_like(reference_points)
    
        #         new_reference_points[..., :2] = tmp[
        #             ..., :2] + inverse_sigmoid(reference_points[..., :2])
        #         new_reference_points[..., 2:3] = tmp[
        #             ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

        #         new_reference_points = new_reference_points.sigmoid()

        #         reference_points = new_reference_points.detach()

        #     output = output.permute(1, 0, 2)
        #     if self.return_intermediate:
        #         intermediate.append(output)
        #         intermediate_reference_points.append(reference_points)

        # if self.return_intermediate:
        #     return torch.stack(intermediate), torch.stack(
        #         intermediate_reference_points)
            attn_index = lid
            for layer in self.operation_order:
                    # temporal self attention
                if layer == 'temp_self_attn':

                    query = self.attentions[attn_index](
                        query,
                        prev_bev,
                        prev_bev,
                        identity if self.pre_norm else None,
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        reference_points=ref_2d,
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        **kwargs)
                    attn_index += 1
                    identity = query

                elif layer == 'norm':
                    query = self.norms[norm_index](query)
                    norm_index += 1

                # spaital cross attention
                elif layer == 'cross_attn':
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        reference_points=ref_3d,
                        reference_points_cam=reference_points_cam,
                        mask=mask,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        **kwargs)
                    attn_index += 1
                    identity = query

                elif layer == 'ffn':
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                    ffn_index += 1
                output = output.permute(1, 0, 2)
                
                if reg_branches is not None:
                    tmp = reg_branches[lid](output)

                    assert reference_points.shape[-1] == 3

                    new_reference_points = torch.zeros_like(reference_points)

                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points[..., :2])
                    new_reference_points[..., 2:3] = tmp[
                        ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                    new_reference_points = new_reference_points.sigmoid()

                    reference_points = new_reference_points.detach()

                output = output.permute(1, 0, 2)
                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)





        return output, reference_points

        
