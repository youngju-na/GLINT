import torch
from typing import Optional
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import cfg
from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.depth_utils import normalize_depth, depth_to_normal
from easyvolcap.utils.loss_utils import l1, l2, l1_reg, l2_reg, cos, mIoU_loss, mse, ScaleAndShiftInvariantMSELoss, lpips


from einops import rearrange

@SUPERVISORS.register_module()
class GlintSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,

                 norm_loss_weight: float = 0.0,
                 norm_loss_weight_final: float = None,
                 norm_loss_start_iter: int = 7000,
                 norm_loss_until_iter: int = None,
                 use_acc_scale_norm_loss: bool = False,
                 use_dpt_scale_norm_loss: bool = False,
                 max_dpt_scale_percet: bool = False,
                 use_spec_scale_norm_loss: bool = False,
                 use_spec_scale_norm_loss_start_iter: int = 7000,
                 use_spec_scale_norm_loss_until_iter: int = None,
                 
                 #! depth loss (new)
                 dpt_loss_weight: float = 0.0,  # depth supervision
                 dpt_loss_start_iter: int = 7000,
                 dpt_loss_until_iter: int = None,         
                 use_dpt_patch_smooth_loss: bool = False,  # whether to use patch-wise depth smoothness loss           
                 dpt_smooth_loss_start_iter: int = 7000,
                 dpt_smooth_loss_until_iter: int = None,
                 
                 

                 #! transmap reg loss (new)
                 trans_map_reg_loss_weight: float = 0.0,
                 trans_reg_loss_start_iter: int = 7000,
                 trans_reg_loss_until_iter: int = None,

                 #! plausibility loss (new)
                 plausibility_loss_weight: float = 0.0,  # use physical plausibility
                 plausibility_loss_start_iter: int = 7000,  # Start iter for plausibility loss
                 plausibility_loss_end_iter: int = None,  # End iter for plausibility loss
                 pp_loss_type: str = 'lpips',  # Type of plausibility loss, can be 'lpips', 'l1', or 'l2'
                 
                 trans_guidance_loss_weight: float = 0.0,  # use transmission guidance
                 trans_guidance_loss_start_iter: int = 1000,  # Start iter for transmission guidance
                 trans_guidance_loss_until_iter: int = None,  # Until iter for transmission guidance
                 albedo_threshold: float = 0.1,  # Threshold for low albedo
                 basecolor_threshold: float = 0.1,  # Threshold for high base color
                 depth_discrepancy_threshold: float = 0.1,  # Threshold for
                 
                 #! new 
                 init_trans_in_stage2: bool = False,  # whether to initialize transmission Gaussians in stage 2
                 use_morphological_filtering: bool = False,  # whether to use morphological filtering for transmission map
                 
                 trans_label_smoothing: float = 0.02,     # 0~0.1 권장
                 trans_pos_weight: float = 1.0,           # 양성(투명) 가중치
                 trans_neg_weight: float = 1.0,           # 음성(불투명) 가중치
                 trans_mask_shrink_kernel: int = 3,       # 경계 오염 방지(erosion 커널)
                 trans_min_iter_for_rgb: int = 3000,      # diffuse 감독은 충분히 수렴 후 사용
                 trans_conf_temperature: float = 1.0,     # 신뢰도 마스크의 온도(가중치 샤프닝)

                 normal_cos_threshold_iter: int = 3000,  # after this iteration, only compute normal loss for pixels with cos>threshold
                 
                 gs_norm_loss_weight: float = 0.0,
                 gs_norm_loss_weight_final: float = None,
                 gs_dist_loss_weight: float = 0.0,
                 gs_norm_loss_start_iter: int = 7000,
                 gs_norm_loss_until_iter: int = None,
                 use_acc_scale_gs_norm_loss: bool = False,
                 use_dpt_scale_gs_norm_loss: bool = False,
                 use_spec_scale_gs_norm_loss: bool = False,
                 use_spec_scale_gs_norm_loss_start_iter: int = 7000,
                 use_spec_scale_gs_norm_loss_until_iter: int = None,
                 gs_dist_loss_start_iter: int = 3000,
                 gs_dist_loss_until_iter: int = None,

                 pcd_opacity_loss_weight: float = 0.0,
                 pcd_opacity_loss_type: str = 'sparse',
                 pcd_opacity_loss_start_iter: int = 0,
                 pcd_opacity_loss_until_iter: int = None,
                 
                 env_opacity_loss_weight: float = 0.0,
                 env_opacity_loss_type: str = 'sparse',
                 env_opacity_loss_start_iter: int = 0,
                 env_opacity_loss_until_iter: int = None,
                 
                 trans_env_opacity_loss_weight: float = 0.0,
                 trans_env_opacity_loss_type: str = 'sparse',
                 trans_env_opacity_loss_start_iter: int = 0,
                 trans_env_opacity_loss_until_iter: int = None,

                 # Mask mIoU loss
                 msk_loss_weight: float = 0.0,  # mask mIoU loss
                 msk_loss_start_iter: int = 7000,
                 msk_loss_until_iter: int = None,

                 # Normal smoothness loss
                 norm_smooth_loss_weight: float = 0.0,
                 norm_smooth_loss_start_iter: int = 7000,
                 norm_smooth_loss_until_iter: int = None,
                 use_edge_aware_smooth: bool = True,
                 use_dpt_scale_norm_smooth_loss: bool = True,
                 
                 # Roughness map smoothness loss
                 rough_map_smooth_loss_weight: float = 0.0,
                 rough_map_smooth_loss_start_iter: int = 7000,
                 rough_map_smooth_loss_until_iter: int = None,

                 # Residual normal loss
                 res_norm_loss_weight: float = 0.001,
                 
                 update_dif_gaussians_until_iter: int = 45000, #! 30000

                 # Specular loss
                 specular_loss_weight: float = 0.0,
                 specular_loss_start_iter: int = 7000,
                 specular_loss_until_iter: int = 9000,
                 specular_target: float = 0.8,
                 min_specular_percent: float = 0.5,

                 # Reflection color loss
                 ref_rgb_loss_weight: float = 0.0,
                 ref_rgb_loss_start_iter: int = 7000,
                 ref_rgb_loss_until_iter: int = 9000,
                 
                 # Multi-view consistency loss parameters
                 multi_view_loss_weight: float = 0.1,  # overall multi-view loss weight
                 multi_view_geo_weight: float = 0.01,  # geometric consistency weight 
                 multi_view_ncc_weight: float = 0.02,  # photometric NCC weight
                 multi_view_start_iter: int = 5000,  # start iteration for multi-view loss
                 multi_view_until_iter: Optional[int] = None,  # end iteration for multi-view loss (None = no limit)
                 
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        # Normal loss
        self.norm_loss_weight = norm_loss_weight
        self.norm_loss_weight_final = norm_loss_weight_final
        self.norm_loss_start_iter = norm_loss_start_iter
        self.norm_loss_until_iter = norm_loss_until_iter
        self.use_acc_scale_norm_loss = use_acc_scale_norm_loss
        self.use_dpt_scale_norm_loss = use_dpt_scale_norm_loss
        self.max_dpt_scale_percet = max_dpt_scale_percet
        self.use_spec_scale_norm_loss = use_spec_scale_norm_loss
        self.use_spec_scale_norm_loss_start_iter = use_spec_scale_norm_loss_start_iter
        self.use_spec_scale_norm_loss_until_iter = use_spec_scale_norm_loss_until_iter
        
        # Depth Loss (New)
        self.dpt_loss_weight = dpt_loss_weight
        self.dpt_loss_start_iter = dpt_loss_start_iter
        self.dpt_loss_until_iter = dpt_loss_until_iter
        self.use_dpt_patch_smooth_loss = use_dpt_patch_smooth_loss  
        self.dpt_smooth_loss_start_iter = dpt_smooth_loss_start_iter
        self.dpt_smooth_loss_until_iter = dpt_smooth_loss_until_iter
        
        # trans_map regularization loss (New)
        self.trans_map_reg_loss_weight = trans_map_reg_loss_weight
        self.trans_reg_loss_start_iter = trans_reg_loss_start_iter 
        self.trans_reg_loss_until_iter = trans_reg_loss_until_iter
        
        # Physical Plausibility Loss (New)
        self.plausibility_loss_weight = plausibility_loss_weight  # use physical plausibility loss
        self.plausibility_loss_start_iter = plausibility_loss_start_iter
        self.plausibility_loss_end_iter = plausibility_loss_end_iter
        self.pp_loss_type = pp_loss_type  # Type of plausibility loss, can be 'lpips', 'l1', or 'l2'
        
        self.trans_guidance_loss_weight = trans_guidance_loss_weight
        self.trans_guidance_loss_start_iter = trans_guidance_loss_start_iter
        self.trans_guidance_loss_until_iter = trans_guidance_loss_until_iter
        
        self.albedo_threshold = albedo_threshold  # Threshold for low albedo
        self.basecolor_threshold = basecolor_threshold  # Threshold for high base color
        self.depth_discrepancy_threshold = depth_discrepancy_threshold  # Threshold for depth discrepancy
        
        self.trans_label_smoothing = trans_label_smoothing
        self.trans_pos_weight = trans_pos_weight         # 양성(투명) 가중치 
        self.trans_neg_weight = trans_neg_weight         # 음성(불투명) 가중치
        self.trans_mask_shrink_kernel = trans_mask_shrink_kernel      # 경계 오염 방지(erosion 커널)
        self.trans_min_iter_for_rgb = trans_min_iter_for_rgb 
        self.trans_conf_temperature = trans_conf_temperature     # 신뢰도 마스크의 온도(가중치 샤프닝)
        
        self.update_dif_gaussians_until_iter = update_dif_gaussians_until_iter
        
        self.normal_cos_threshold_iter = normal_cos_threshold_iter
        

        
        self.init_trans_in_stage2 = init_trans_in_stage2
        self.use_morphological_filtering = use_morphological_filtering
        
        self.gs_norm_loss_weight = gs_norm_loss_weight
        self.gs_norm_loss_weight_final = gs_norm_loss_weight_final
        self.gs_dist_loss_weight = gs_dist_loss_weight
        self.gs_norm_loss_start_iter = gs_norm_loss_start_iter
        self.gs_norm_loss_until_iter = gs_norm_loss_until_iter
        self.use_acc_scale_gs_norm_loss = use_acc_scale_gs_norm_loss
        self.use_dpt_scale_gs_norm_loss = use_dpt_scale_gs_norm_loss
        self.use_spec_scale_gs_norm_loss = use_spec_scale_gs_norm_loss
        self.use_spec_scale_gs_norm_loss_start_iter = use_spec_scale_gs_norm_loss_start_iter
        self.use_spec_scale_gs_norm_loss_until_iter = use_spec_scale_gs_norm_loss_until_iter
        self.gs_dist_loss_start_iter = gs_dist_loss_start_iter
        self.gs_dist_loss_until_iter = gs_dist_loss_until_iter

        self.pcd_opacity_loss_weight = pcd_opacity_loss_weight
        self.pcd_opacity_loss_type = pcd_opacity_loss_type
        self.pcd_opacity_loss_start_iter = pcd_opacity_loss_start_iter
        self.pcd_opacity_loss_until_iter = pcd_opacity_loss_until_iter

        self.env_opacity_loss_weight = env_opacity_loss_weight
        self.env_opacity_loss_type = env_opacity_loss_type
        self.env_opacity_loss_start_iter = env_opacity_loss_start_iter
        self.env_opacity_loss_until_iter = env_opacity_loss_until_iter

        self.trans_env_opacity_loss_weight = trans_env_opacity_loss_weight
        self.trans_env_opacity_loss_type = trans_env_opacity_loss_type
        self.trans_env_opacity_loss_start_iter = trans_env_opacity_loss_start_iter
        self.trans_env_opacity_loss_until_iter = trans_env_opacity_loss_until_iter

        # Mask mIoU loss
        self.msk_loss_weight = msk_loss_weight
        self.msk_loss_start_iter = msk_loss_start_iter
        self.msk_loss_until_iter = msk_loss_until_iter

        # Smooth loss
        self.norm_smooth_loss_weight = norm_smooth_loss_weight
        self.norm_smooth_loss_start_iter = norm_smooth_loss_start_iter
        self.norm_smooth_loss_until_iter = norm_smooth_loss_until_iter
        self.use_edge_aware_smooth = use_edge_aware_smooth
        self.use_dpt_scale_norm_smooth_loss = use_dpt_scale_norm_smooth_loss

        # Roughness map smoothness loss
        self.rough_map_smooth_loss_weight = rough_map_smooth_loss_weight
        self.rough_map_smooth_loss_start_iter = rough_map_smooth_loss_start_iter
        self.rough_map_smooth_loss_until_iter = rough_map_smooth_loss_until_iter

        # Residual normal loss
        self.res_norm_loss_weight = res_norm_loss_weight

        # Specular loss
        self.specular_loss_weight = specular_loss_weight
        self.specular_loss_start_iter = specular_loss_start_iter
        self.specular_loss_until_iter = specular_loss_until_iter
        self.specular_target = specular_target
        self.min_specular_percent = min_specular_percent

        # Reflection color loss
        self.ref_rgb_loss_weight = ref_rgb_loss_weight
        self.ref_rgb_loss_start_iter = ref_rgb_loss_start_iter
        self.ref_rgb_loss_until_iter = ref_rgb_loss_until_iter

        # Multi-view consistency loss
        self.multi_view_loss_weight = multi_view_loss_weight
        self.multi_view_geo_weight = multi_view_geo_weight
        self.multi_view_ncc_weight = multi_view_ncc_weight
        self.multi_view_start_iter = multi_view_start_iter
        self.multi_view_until_iter = multi_view_until_iter

        # Compute the total number of iterations
        self.total_iter = cfg.runner_cfg.epochs * cfg.runner_cfg.ep_iter
        self.depth_loss = ScaleAndShiftInvariantMSELoss()
        
        self.valid_normal_mask = None  # To store the valid normal mask
        

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'pcd_opacity' in output and self.pcd_opacity_loss_weight > 0:
            if output.iter >= self.pcd_opacity_loss_start_iter and \
               (self.pcd_opacity_loss_until_iter is None or output.iter < self.pcd_opacity_loss_until_iter):
                if self.pcd_opacity_loss_type == 'sparse':
                    epsilon = 1e-3
                    v = torch.clamp(output.pcd_opacity, epsilon, 1 - epsilon)
                    # Non-negative binary entropy; minimized at v≈0 or 1
                    pcd_opacity_loss = torch.mean(-(v * torch.log(v) + (1 - v) * torch.log(1 - v)))
                elif self.pcd_opacity_loss_type == 'l1':
                    pcd_opacity_loss = l1_reg(1 - output.pcd_opacity)
                else:
                    pcd_opacity_loss = 0
                scalar_stats.pcd_opacity_loss = pcd_opacity_loss
                loss += self.pcd_opacity_loss_weight * pcd_opacity_loss
                
        if 'env_opacity' in output and output.env_opacity is not None and self.env_opacity_loss_weight > 0:
            if output.iter >= self.env_opacity_loss_start_iter and \
               (self.env_opacity_loss_until_iter is None or output.iter < self.env_opacity_loss_until_iter):
                if self.env_opacity_loss_type == 'sparse':
                    epsilon = 1e-3
                    v = torch.clamp(output.env_opacity, epsilon, 1 - epsilon)
                    # Non-negative binary entropy; minimized at v≈0 or 1
                    env_opacity_loss = torch.mean(-(v * torch.log(v) + (1 - v) * torch.log(1 - v)))
                elif self.env_opacity_loss_type == 'l1':
                    env_opacity_loss = l1_reg(1 - output.env_opacity)
                else:
                    env_opacity_loss = 0
                
                scalar_stats.env_opacity_loss = env_opacity_loss
                loss += self.env_opacity_loss_weight * env_opacity_loss
        
        if 'trans_env_opacity' in output and output.trans_env_opacity is not None and self.trans_env_opacity_loss_weight > 0:
            if output.iter >= self.trans_env_opacity_loss_start_iter and \
               (self.trans_env_opacity_loss_until_iter is None or output.iter < self.trans_env_opacity_loss_until_iter):
                # Ensure variable is always defined
                if self.trans_env_opacity_loss_type == 'sparse':
                    epsilon = 1e-3
                    v = torch.clamp(output.trans_env_opacity, epsilon, 1 - epsilon)
                    # Non-negative binary entropy; minimized at v≈0 or 1
                    trans_env_opacity_loss = torch.mean(-(v * torch.log(v) + (1 - v) * torch.log(1 - v)))
                elif self.trans_env_opacity_loss_type == 'l1':
                    trans_env_opacity_loss = l1_reg(1 - output.trans_env_opacity)
                else:
                    trans_env_opacity_loss = 0
                
                scalar_stats.trans_env_opacity_loss = trans_env_opacity_loss
                loss += self.trans_env_opacity_loss_weight * trans_env_opacity_loss
                
    
        if 'norm_map' in output and self.norm_loss_weight > 0:
            if output.iter >= self.norm_loss_start_iter and \
                (self.norm_loss_until_iter is None or output.iter < self.norm_loss_until_iter):

                # Transform the normal map to the local coordinate system
                # norm_map = normalize(output.surf_norm_map)
                norm_map = normalize(output.norm_map)
                norm_map = norm_map @ batch.R.mT  # convert to view space
                norm_map = normalize(norm_map)

                norm = None
                if 'normal' in batch:
                    norm = batch.normal * 2. - 1. #! range [0,1] -> [-1,1]
                    norm = normalize(norm)
                
                elif 'norm' in batch:
                    # Process the ground truth normal map
                    norm = batch.norm * 2. - 1.  #! Option 1: Stable Normal !!!                    
                
                if norm is not None:
                    if output.iter >= self.normal_cos_threshold_iter:
                        normal_threshold = 0.3
                        cosine_similarity = (norm * (norm_map)).sum(dim=-1)
                        self.valid_normal_mask = (cosine_similarity > normal_threshold)[..., None]
                    else:
                        self.valid_normal_mask = torch.ones_like(norm[..., 0:1], dtype=torch.bool)

                    # Compute normal loss
                    norm_loss = 1 - F.cosine_similarity(norm_map * self.valid_normal_mask, norm * self.valid_normal_mask, dim=-1)  # MARK: SYNC
                else:
                    norm_loss = 0

                # Maybe scale the normal loss with acc_map
                if self.use_acc_scale_norm_loss:
                    scale_acc = output.acc_map[..., 0].detach().clone()
                    norm_loss = norm_loss * scale_acc
                # Maybe scale the normal loss with inverse normalized depth 
                if self.use_dpt_scale_norm_loss:
                    if self.max_dpt_scale_percet:
                        # Exclude the points with large depth and zero depth
                        dpt_msk = output.dpt_map[..., 0].detach().clone() > 0
                        dpt_msk = torch.logical_and(dpt_msk, output.dpt_map[..., 0].detach().clone() <= torch.quantile(output.dpt_map[dpt_msk], self.max_dpt_scale_percet))
                        norm_loss[~dpt_msk] = 0
                    else:
                        # Scale by inverse normalized depth
                        scale_dpt = normalize_depth(output.dpt_map[..., 0].detach().clone())
                        norm_loss = norm_loss * scale_dpt

                norm_loss = norm_loss.mean()
                scalar_stats.norm_loss = norm_loss
                loss += self.norm_loss_weight * norm_loss
                
        if 'dpt_map' in output and 'depth' in batch and self.dpt_loss_weight > 0:
            if output.iter >= self.dpt_loss_start_iter and \
               (self.dpt_loss_until_iter is None or output.iter < self.dpt_loss_until_iter):
                mask = (batch.depth > 0.)
                assert output.dpt_map.shape[-1] == batch.depth.shape[-1], \
                    f"Output depth map shape {output.dpt_map.shape} does not match batch depth shape {batch.depth.shape}"
                
                dpt_loss = self.depth_loss(output.dpt_map, batch.depth, mask) 
                scalar_stats.dpt_loss = dpt_loss
                loss += self.dpt_loss_weight * dpt_loss
                
                # add patch-wise depth smoothness loss (vectorized, mask-aware)
                if self.use_dpt_patch_smooth_loss and \
                     output.iter >= self.dpt_smooth_loss_start_iter and \
                        (self.dpt_smooth_loss_until_iter is None or output.iter < self.dpt_smooth_loss_until_iter):
                    dpt = output.dpt_map[..., 0]
                    B, HW = dpt.shape
                    H, W = batch.H, batch.W
                    dpt_2d = dpt.view(B, 1, H, W)  # (B,1,H,W)

                    # 유효 깊이 마스크 (원 코드와 동일한 기준)
                    valid = ((dpt_2d > 0) & (dpt_2d < 100.0)).float()  # (B,1,H,W)

                    # 패치 설정
                    patch_size = 8               # 8x8 패치
                    stride = patch_size // 2     # 50% overlap = 4
                    k = patch_size

                    # “합”을 만드는 1-kernel (고정 가중치)
                    # groups=1, bias=None, requires_grad=False → 순수한 박스필터 합
                    weight = torch.ones((1, 1, k, k), device=dpt_2d.device, dtype=dpt_2d.dtype)

                    # (가려진) 합, 제곱합, 카운트
                    # sum(x * m), sum(x^2 * m), sum(m)
                    sum_x   = torch.conv2d(dpt_2d * valid, weight, stride=stride, padding=0)
                    sum_x2  = torch.conv2d((dpt_2d ** 2) * valid, weight, stride=stride, padding=0)
                    cnt     = torch.conv2d(valid, weight, stride=stride, padding=0) + 1e-8  # div-by-zero 방지

                    # 유효 패치(>=50% 유효 픽셀)
                    valid_ratio = cnt / (k * k)
                    patch_mask = (valid_ratio >= 0.5).float()  # (B,1,H',W')

                    # 마스크 평균/분산 (E[x], E[x^2] - E[x]^2)
                    mean = sum_x / cnt
                    var  = (sum_x2 / cnt) - mean.pow(2)

                    # 스케일 불변 정규화: var / mean^2
                    norm = (mean.abs().clamp(min=1e-6)).pow(2)
                    norm_var = var / norm

                    # 최종 loss: 유효 패치 평균
                    # (detach 유지: 위에서 dpt_2d가 detach라 그래프에 안 엮임)
                    dpt_smooth_loss = (norm_var * patch_mask).sum() / (patch_mask.sum() + 1e-8)

                    scalar_stats.dpt_smooth_loss = dpt_smooth_loss #! * 0.1 (0906 버전에서는 on)
                    loss += self.dpt_loss_weight * dpt_smooth_loss
               
                
                
                
        if 'norm_map' in output and 'surf_norm_map' in output and self.gs_norm_loss_weight > 0:
            # Compute the normal consistency loss after a certain iteration
            if output.iter >= self.gs_norm_loss_start_iter and \
                (self.gs_norm_loss_until_iter is None or output.iter < self.gs_norm_loss_until_iter):

                # Compute the normal consistency loss
                gs_norm_loss = 1 - (output.norm_map * output.surf_norm_map).sum(dim=-1)
                # Maybe scale the normal loss with acc_map
                if self.use_acc_scale_gs_norm_loss:
                    scale_acc = output.acc_map[..., 0].detach().clone()
                    gs_norm_loss = gs_norm_loss * scale_acc
                # Maybe scale the normal loss with inverse normalized depth
                if self.use_dpt_scale_gs_norm_loss:
                    if self.max_dpt_scale_percet:
                        # Exclude the points with large depth and zero depth
                        dpt_msk = output.dpt_map[..., 0].detach().clone() > 0
                        dpt_msk = torch.logical_and(dpt_msk, output.dpt_map[..., 0].detach().clone() <= torch.quantile(output.dpt_map[dpt_msk], self.max_dpt_scale_percet))
                        gs_norm_loss[~dpt_msk] = 0
                    else:
                        # Scale by inverse normalized depth
                        scale_dpt = normalize_depth(output.dpt_map[..., 0].detach().clone())
                        gs_norm_loss = gs_norm_loss * scale_dpt                
                
                gs_norm_loss = gs_norm_loss.mean()
                scalar_stats.gs_norm_loss = gs_norm_loss
                loss += self.gs_norm_loss_weight * gs_norm_loss

        if 'acc_map' in output and self.msk_loss_weight > 0:
            # Get the mask
            if output.iter >= self.msk_loss_start_iter and \
              (self.msk_loss_until_iter is None or output.iter < self.msk_loss_until_iter):
                mask = torch.logical_and(batch.msk[..., 0] > 0.5, torch.norm(batch.normal, dim=-1) > 0.25)[..., None]  # (B, P, 1)
                msk_loss = mse(output.acc_map, mask)
                scalar_stats.msk_loss = msk_loss
                loss += self.msk_loss_weight * msk_loss

        if 'dist_map' in output and self.gs_dist_loss_weight > 0:
            # Compute the distance consistency loss after a certain iteration
            if output.iter >= self.gs_dist_loss_start_iter and \
                (self.gs_dist_loss_until_iter is None or output.iter < self.gs_dist_loss_until_iter):
                # Compute the distance consistency loss
                gs_dist_loss = output.dist_map.mean()

                # Log and add the loss
                scalar_stats.gs_dist_loss = gs_dist_loss
                loss += self.gs_dist_loss_weight * gs_dist_loss
        
        if 'trans_map' in output and self.trans_map_reg_loss_weight > 0:
            # Compute the transmittance regularization loss after a certain iteration
            if output.iter >= self.trans_reg_loss_start_iter and \
               (self.trans_reg_loss_until_iter is None or output.iter < self.trans_reg_loss_until_iter):
                # Compute entropy loss to make trans_map close to 0 or 1
                epsilon = 1e-6
                trans_map_clamped = torch.clamp(output.trans_map, epsilon, 1 - epsilon)
                entropy_loss = - (trans_map_clamped * torch.log(trans_map_clamped) + 
                                  (1 - trans_map_clamped) * torch.log(1 - trans_map_clamped)).mean()

                # Add L1 regularization
                l1_loss = l1_reg(output.trans_map) 

                # Add smoothness regularizers
                trans_map = output.trans_map
                B, HW, C = trans_map.shape
                trans_map = trans_map.reshape(B, batch.H, batch.W, C)
                B, H, W, C = trans_map.shape

                trans_map_dy = trans_map[:, 1:, :, :] - trans_map[:, :-1, :, :]
                trans_map_dx = trans_map[:, :, 1:, :] - trans_map[:, :, :-1, :]

                # Use rgb image to compute edge-aware weights
                rgb_map = batch.rgb.reshape(B, H, W, -1)
                rgb_dy = rgb_map[:, 1:, :, :] - rgb_map[:, :-1, :, :]
                rgb_dx = rgb_map[:, :, 1:, :] - rgb_map[:, :, :-1, :]
                
                # Lower weight at edges
                weights_x = torch.exp(-torch.mean(torch.abs(rgb_dx), dim=-1, keepdim=True))
                weights_y = torch.exp(-torch.mean(torch.abs(rgb_dy), dim=-1, keepdim=True))
                
                smooth_loss_x = torch.mean(weights_x * torch.abs(trans_map_dx))
                smooth_loss_y = torch.mean(weights_y * torch.abs(trans_map_dy))
                smooth_loss = smooth_loss_x + smooth_loss_y
                
                # Combine the losses
                trans_reg_loss = entropy_loss + l1_loss + smooth_loss

                # Log and add the loss
                scalar_stats.gs_trans_reg_loss = trans_reg_loss
                loss += self.trans_map_reg_loss_weight * trans_reg_loss
                
        if ('trans_map' in output and 
            'confident_transparent_mask' in batch and 
            'confident_opaque_mask' in batch and
            self.trans_guidance_loss_weight > 0):

            if (output.iter >= self.trans_guidance_loss_start_iter and 
                (self.trans_guidance_loss_until_iter is None or output.iter < self.trans_guidance_loss_until_iter)):

                # --- 0) 준비/shape 정리 ---
                B, HW, C = output.trans_map.shape
                H, W = batch.H, batch.W

                trans_prob = output.trans_map.clamp(0.0, 1.0)  # 확률로 가정

                # 라벨 스무딩
                s = self.trans_label_smoothing
                pos_label = (1.0 - s)
                neg_label = (0.0 + s)

                # 신뢰도(가중치) – 온도로 샤프닝
                with torch.no_grad():
                    t_mask = batch.confident_transparent_mask.detach().float()  # (B,HW,1) in {0,1} or [0,1]
                    o_mask = batch.confident_opaque_mask.detach().float()
                    if self.trans_conf_temperature != 1.0:
                        t_mask = t_mask.pow(self.trans_conf_temperature)
                        o_mask = o_mask.pow(self.trans_conf_temperature)

                # --- 1) 경계 오염 방지: erosion으로 마스크 축소 ---
                # (B,HW,1)->(B,H,W,1) 후 erosion
                t_hw1 = t_mask.reshape(B, H, W, 1)
                o_hw1 = o_mask.reshape(B, H, W, 1)
                if self.trans_mask_shrink_kernel > 1:
                    t_hw1 = self._shrink_mask(t_hw1, self.trans_mask_shrink_kernel)
                    o_hw1 = self._shrink_mask(o_hw1, self.trans_mask_shrink_kernel)
                t_mask = t_hw1.reshape(B, HW, 1)
                o_mask = o_hw1.reshape(B, HW, 1)

                # --- 2) 분리된 BCE 감독(클래스 불균형 보정) ---
                # 투명: 1로, 불투명: 0으로
                pos_tgt = torch.full_like(trans_prob, pos_label)
                neg_tgt = torch.full_like(trans_prob, neg_label)

                # 마스크 가중 평균
                pos_loss = self._balanced_bce(trans_prob, pos_tgt, 
                                            pos_w=self.trans_pos_weight, 
                                            neg_w=0.0)
                neg_loss = self._balanced_bce(trans_prob, neg_tgt, 
                                            pos_w=0.0, 
                                            neg_w=self.trans_neg_weight)

                # 마스크로 실제 적용 영역 제한
                if t_mask.any():
                    pos_loss = (self._bce_prob(trans_prob, pos_tgt) * t_mask).sum() / (t_mask.sum() + 1e-8)
                else:
                    pos_loss = trans_prob.new_tensor(0.0)

                if o_mask.any():
                    neg_loss = (self._bce_prob(trans_prob, neg_tgt) * o_mask).sum() / (o_mask.sum() + 1e-8)
                else:
                    neg_loss = trans_prob.new_tensor(0.0)

                trans_loss = pos_loss + neg_loss

                # # --- 3) Diffuse 알베도 일치 항 (투명영역만, 일정 이터 이후) ---
                # if ('dif_rgb_map' in output and 'diffuse_albedo' in batch and 
                #     output.iter >= self.trans_min_iter_for_rgb):

                #     # (B,HW,C) -> 투명 영역만, 또 trans_prob로 가중
                #     mask = t_mask.squeeze(-1).bool()
                #     if mask.any():
                #         # trans_prob로 신뢰 기반 가중(초기 흔들림 완화)
                #         w = trans_prob.squeeze(-1).detach()
                #         w = torch.where(mask, w, torch.zeros_like(w))
                #         # 안전 평균
                #         num = (w[mask] * (output.dif_rgb_map[mask] - batch.diffuse_albedo[mask]).pow(2).mean(dim=-1)).sum()
                #         den = w[mask].sum() + 1e-8
                #         diffuse_loss = num / den
                #         trans_loss = trans_loss + diffuse_loss

                # --- 4) (선택) 깊이 불일치 억제: 투명영역에서 직접광 깊이와 충돌 방지 ---
                if ('dpt_map' in output and 'depth' in batch):
                    # 투명인데 예측 깊이가 지나치게 앞이면(유리 앞에 뜬 경우) 추가 벌점
                    depth_err = (batch.depth - output.dpt_map)[..., 0]  # +면 예측이 얕음
                    viol = (depth_err > getattr(self, 'free_space_margin', 0.03)) & (batch.depth[...,0] > 0)
                    if viol.any():
                        # 투명 마스크도 함께 고려
                        viol = viol & t_mask[...,0].bool()
                        if viol.any():
                            fs = trans_prob[...,0][viol].mean()  # 앞에서 누적된 투명도 억제
                            trans_loss = trans_loss + 0.5*fs  # 약하게 결합 (필요시 조절)

                scalar_stats.trans_guidance_loss = trans_loss
                loss += self.trans_guidance_loss_weight * trans_loss


        if 'rough_map' in output and self.rough_map_smooth_loss_weight > 0:
            # Compute the roughness regularization loss after a certain iteration
            if output.iter >= self.rough_map_smooth_loss_start_iter and \
               (self.rough_map_smooth_loss_until_iter is None or output.iter < self.rough_map_smooth_loss_until_iter):
                # Compute entropy loss to make rough_map close to 0 or 1

                # Add smoothness regularizers
                rough_map = output.rough_map
                B, HW, C = rough_map.shape
                rough_map = rough_map.reshape(B, batch.H, batch.W, C)
                B, H, W, C = rough_map.shape

                rough_map_dy = rough_map[:, 1:, :, :] - rough_map[:, :-1, :, :]
                rough_map_dx = rough_map[:, :, 1:, :] - rough_map[:, :, :-1, :]

                # Use rgb image to compute edge-aware weights
                rgb_map = batch.rgb.reshape(B, H, W, -1)
                rgb_dy = rgb_map[:, 1:, :, :] - rgb_map[:, :-1, :, :]
                rgb_dx = rgb_map[:, :, 1:, :] - rgb_map[:, :, :-1, :]
                
                # Lower weight at edges
                weights_x = torch.exp(-torch.mean(torch.abs(rgb_dx), dim=-1, keepdim=True))
                weights_y = torch.exp(-torch.mean(torch.abs(rgb_dy), dim=-1, keepdim=True))

                smooth_loss_x = torch.mean(weights_x * torch.abs(rough_map_dx))
                smooth_loss_y = torch.mean(weights_y * torch.abs(rough_map_dy))
                smooth_loss = smooth_loss_x + smooth_loss_y
                
                # Log and add the loss
                scalar_stats.gs_rough_smooth_loss = smooth_loss
                loss += self.rough_map_smooth_loss_weight * smooth_loss

       
        # normal depth smoothness loss
        if 'norm_map' in output and self.norm_smooth_loss_weight > 0:
            if output.iter >= self.norm_smooth_loss_start_iter and \
                (self.norm_smooth_loss_until_iter is None or output.iter < self.norm_smooth_loss_until_iter):
                
                norm_map = output.norm_map
                B, HW, C = norm_map.shape
                norm_map = norm_map.reshape(B, batch.H, batch.W, C)
                B, H, W, C = norm_map.shape

                # Compute normal gradients
                norm_map_dy = norm_map[:, 1:, :, :] - norm_map[:, :-1, :, :]
                norm_map_dx = norm_map[:, :, 1:, :] - norm_map[:, :, :-1, :]

                trans_norm_map_dx = None
                trans_norm_map_dy = None
                if "trans_norm_map" in output and output.trans_norm_map is not None:
                    trans_norm_map = output.trans_norm_map
                    B, HW, C = trans_norm_map.shape
                    trans_norm_map = trans_norm_map.reshape(B, batch.H, batch.W, C)
                    B, H, W, C = trans_norm_map.shape

                    trans_norm_map_dy = trans_norm_map[:, 1:, :, :] - trans_norm_map[:, :-1, :, :]
                    trans_norm_map_dx = trans_norm_map[:, :, 1:, :] - trans_norm_map[:, :, :-1, :]
                
                
                smooth_loss = 0
                trans_smooth_loss = 0
                
                if self.use_edge_aware_smooth:
                    # Use rgb image to compute edge-aware weights
                    rgb_map = batch.rgb.reshape(B, H, W, -1)
                    rgb_dy = rgb_map[:, 1:, :, :] - rgb_map[:, :-1, :, :]
                    rgb_dx = rgb_map[:, :, 1:, :] - rgb_map[:, :, :-1, :]
                    
                    # Lower weight at edges
                    weights_x = torch.exp(-torch.mean(torch.abs(rgb_dx), dim=-1, keepdim=True))
                    weights_y = torch.exp(-torch.mean(torch.abs(rgb_dy), dim=-1, keepdim=True))
                    
                    smooth_loss_x = torch.mean(weights_x * torch.abs(norm_map_dx))
                    smooth_loss_y = torch.mean(weights_y * torch.abs(norm_map_dy))

                    
                    if trans_norm_map_dy is not None and trans_norm_map_dx is not None:
                        trans_smooth_loss_x = torch.mean(weights_x * torch.abs(trans_norm_map_dx))
                        trans_smooth_loss_y = torch.mean(weights_y * torch.abs(trans_norm_map_dy))
                        trans_smooth_loss = trans_smooth_loss_x + trans_smooth_loss_y
                        scalar_stats.trans_smooth_loss = trans_smooth_loss
                        loss += self.norm_smooth_loss_weight * trans_smooth_loss * 0.1 # smaller weight for transparency

                    smooth_loss = smooth_loss_x + smooth_loss_y
                else:
                    # Basic L1 smoothness
                    smooth_loss = torch.mean(torch.abs(norm_map_dx)) + torch.mean(torch.abs(norm_map_dy))
                    
                scalar_stats.norm_smooth_loss = smooth_loss
                
                loss += self.norm_smooth_loss_weight * smooth_loss
        
        
        # ! Modified: Depth-Discrepancy-Aware Transparency Guidance using direct rasterization
        # if 'trans_map' in output and 'confident_transparent_mask' in batch and 'diffuse_albedo' in batch and self.trans_guidance_loss_weight > 0:
        #     if output.iter >= self.trans_guidance_loss_start_iter and \
        #        (self.trans_guidance_loss_until_iter is None or output.iter < self.trans_guidance_loss_until_iter):
                
        #         with torch.no_grad():
        #             # Get masks
        #             confident_transparent_mask = batch.confident_transparent_mask.detach()  # (B, P, 1)
        #             confident_opaque_mask = batch.confident_opaque_mask.detach()  # (B, P, 1)

        #         # Separate losses for transparent and opaque regions
        #         trans_loss = torch.tensor(0.0, device=confident_transparent_mask.device)

        #         if confident_transparent_mask.any():
        #             # Compute loss for transparent regions
        #             transparent_loss = l1(
        #                 output.trans_map * confident_transparent_mask,
        #                 torch.ones_like(confident_transparent_mask) * confident_transparent_mask,
        #             )
        #             trans_loss += transparent_loss
                    
        #             mask_squeezed = confident_transparent_mask.squeeze(-1).bool()
        #             diffuse_loss = l2(output.dif_rgb_map[mask_squeezed], batch.diffuse_albedo[mask_squeezed])
        #             trans_loss += diffuse_loss

        #         if confident_opaque_mask.any():
        #             # Loss for confident opaque regions (should be close to 0.0)  
        #             if confident_opaque_mask.sum() > 0:
        #                 opaque_loss = l1(
        #                     output.trans_map * confident_opaque_mask,
        #                     torch.zeros_like(confident_opaque_mask) * confident_opaque_mask, 
        #                 )
        #                 trans_loss += opaque_loss

        #         scalar_stats.trans_guidance_loss = trans_loss
        #         loss += self.trans_guidance_loss_weight * trans_loss
                

        # ! New: Physical Plausibility Loss for env and trans_env
        if  self.plausibility_loss_weight > 0 and output.iter >= self.plausibility_loss_start_iter:
            
            plausibility_loss = 0
            
            if 'env_rgb_direct' in output:
                mask = (output.env_acc_direct > 0).float().detach()
                mask = mask.float().detach()
                if self.pp_loss_type == "lpips":
                    env_rgb_masked = rearrange(output.env_rgb_direct * mask, 'b (H W) C -> b C H W', H=batch.H, W=batch.W) 
                    batch_rgb = rearrange(batch.rgb * mask, 'b (H W) C -> b C H W', H=batch.H, W=batch.W) 
                    env_plausibility_loss = lpips(env_rgb_masked, batch_rgb)
                elif self.pp_loss_type == "l1":
                    # no rearrange required for l1 loss
                    env_plausibility_loss = l1(output.env_rgb_direct * mask, batch.rgb * mask)
                elif self.pp_loss_type == "l2":
                    # no rearrange required for l2 loss
                    env_plausibility_loss = l2(output.env_rgb_direct * mask, batch.rgb * mask)
                else:
                    raise ValueError(f"Unsupported plausibility loss type: {self.pp_loss_type}")
                
                plausibility_loss += env_plausibility_loss
                
            if 'trans_env_rgb_direct' in output:
                opaque_mask = batch.confident_opaque_mask.detach()
                geo_mask = opaque_mask.bool() & (batch.depth > 0) if 'depth' in batch else opaque_mask.bool()
                # rgb_mask = (output.trans_env_acc_direct > 0).float().detach()
                geo_mask = geo_mask.float().detach()
                # trans_env_plausibility_loss_rgb = l2(output.trans_env_rgb_direct * rgb_mask, batch.rgb * rgb_mask)
                trans_env_plausibility_loss_depth = self.depth_loss(output.trans_dpt_map, output.dpt_map.clone().detach(), geo_mask)

                norm = None                
                # Compute normal loss
                if 'normal' in batch:
                    norm = batch.normal * 2. - 1.  # ! Option 2: diffren normal
                    norm = normalize(norm)
                elif 'norm' in batch:
                    norm = batch.norm * 2. - 1.  # ! Option 1: Stable Normal
                    norm = normalize(norm)

                trans_norm_map = output.trans_norm_map
                # view space
                trans_norm_map = trans_norm_map @ batch.R.mT  # convert to view space
                trans_norm_map = normalize(trans_norm_map)

                # 기존: 마스크 없이 전체 평균
                # trans_env_plausibility_loss_norm = (1 - F.cosine_similarity(output.trans_norm_map, norm, dim=-1)).mean()

                trans_env_plausibility_loss_norm = 0
                # 변경: geo_mask로 가린 뒤 마스크 평균
                if norm is not None:
                    cos_sim = F.cosine_similarity(trans_norm_map, norm, dim=-1)  # [B, H*W]
                    masked_norm_loss = (1.0 - cos_sim) * geo_mask[..., 0]  # geo_mask: [B, H*W], float(), detach() already done
                    trans_env_plausibility_loss_norm = masked_norm_loss.sum() / (geo_mask[..., 0].sum() + 1e-8)
                
                plausibility_loss += (trans_env_plausibility_loss_depth + trans_env_plausibility_loss_norm)
                scalar_stats.pp_loss = plausibility_loss

            loss += self.plausibility_loss_weight * plausibility_loss

        # Multi-view Consistency Loss
        if self.multi_view_loss_weight > 0 and hasattr(output, 'geo_loss') and hasattr(output, 'ncc_loss'):
            multi_view_loss = 0

            # geo loss
            if output.iter >= self.multi_view_start_iter and \
               (self.multi_view_until_iter is None or output.iter < self.multi_view_until_iter):
                if self.multi_view_geo_weight > 0:
                    scalar_stats.multi_view_geo_loss = output.geo_loss * self.multi_view_geo_weight
                    multi_view_loss += output.geo_loss * self.multi_view_geo_weight

                if self.multi_view_ncc_weight > 0:
                    scalar_stats.multi_view_ncc_loss = output.ncc_loss * self.multi_view_ncc_weight
                    multi_view_loss += output.ncc_loss * self.multi_view_ncc_weight
                    
            scalar_stats.multi_view_loss = multi_view_loss * self.multi_view_loss_weight
            loss += multi_view_loss
                    
        return loss
    
    def _safe_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_tensor(0.0)
        return x.mean()

    def _minpool2d(self, x4: torch.Tensor, k: int) -> torch.Tensor:
        # erosion 대체: minpool = -maxpool(-x)
        return -F.max_pool2d(-x4, kernel_size=k, stride=1, padding=k//2)

    def _shrink_mask(self, m_hw1: torch.Tensor, k: int) -> torch.Tensor:
        # m_hw1: (B,H,W,1) in {0,1}
        m = m_hw1.permute(0,3,1,2)  # B,1,H,W
        eroded = self._minpool2d(m, k)
        return (eroded > 0.5).float().permute(0,2,3,1)

    def _bce_prob(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        pred = torch.clamp(pred, eps, 1.0 - eps)
        return -(target*pred.log() + (1-target)*(1-pred).log())

    def _balanced_bce(self, pred_prob: torch.Tensor, target: torch.Tensor,
                    pos_w: float, neg_w: float) -> torch.Tensor:
        # pred_prob/target: (B,HW,1) in [0,1]
        bce = self._bce_prob(pred_prob, target)
        w = target*pos_w + (1-target)*neg_w
        return (w*bce).mean()
