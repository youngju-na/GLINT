import copy
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

import cv2
import os
from einops import rearrange, repeat, reduce

from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.networks.noop_network import NoopNetwork
from easyvolcap.models.samplers.gaussian2d_sampler import Gaussian2DSampler


from easyvolcap.utils.sh_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.grid_utils import sample_points_subgrid
from easyvolcap.utils.colmap_utils import load_sfm_ply, save_sfm_ply
from easyvolcap.utils.net_utils import freeze_module, make_params, make_buffer
from easyvolcap.utils.gaussian2d_utils import GaussianModel, render, prepare_gaussian_camera, sh02rgb
from easyvolcap.utils.graphics_utils import patch_offsets, patch_warp, lncc
from easyvolcap.utils.data_utils import load_pts, export_pts, to_x, to_cuda, to_cpu, to_tensor, remove_batch
from easyvolcap.utils.fusion_utils import compute_consistency  # Import geometric consistency function

import kornia
import kornia.morphology as K

@SAMPLERS.register_module()
class GlintSampler(Gaussian2DSampler):
    def __init__(self,
                 # Legacy APIs
                 network: NoopNetwork = None,  # ignore this

                 # 3DGS-DR related configs
                 sh_start_iter: int = 10000,
                 densify_until_iter: int = 30000,
                 init_densification_interval: int = 100,
                 norm_densification_interval: int = 500,
                 normal_prop_until_iter: int = 24000,
                 normal_prop_interval: int = 1000,
                 opacity_lr0_interval: int = 200,
                 opacity_lr: float = 0.05,
                 color_sabotage_until_iter: int = 24000,
                 color_sabotage_interval: int = 1000,
                 reset_specular_all: bool = False,

                 # Gaussian configs
                 env_preload_gs: str = '',
                 env_bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 # SHs configs
                 env_sh_deg: int = 3,
                 env_init_sh_deg: int = 0,
                 env_sh_start_iter: int = 0,
                 env_sh_update_iter: int = 1000,
                 # Opacity and scale configs
                 env_init_occ: float = 0.1,
                 # Densify & pruning configs
                 env_densify_from_iter: int = 500,
                 env_densify_until_iter: int = 15000,
                 env_densification_interval: int = 100,
                 env_opacity_reset_interval: int = 3000,
                 
                 trans_env_densify_from_iter: int = 500,
                 trans_env_densify_until_iter: int = 15000,
                 trans_env_densification_interval: int = 200,
                 trans_env_opacity_reset_interval: int = 3000,
                 trans_env_densify_grad_threshold: float = 0.01,
                 trans_env_min_opacity: float = 0.05, #! 0.05
                 
                 env_densify_grad_threshold: float = 0.0002,
                 env_min_opacity: float = 0.08, #! 0.05
                 env_densify_size_threshold: float = 0.01,  # alias for `percent_dense` as in the original code, https://github.com/hbb1/2d-gaussian-splatting/blob/6d249deeec734ad07760496fc32be3b91ac236fc/scene/gaussian_model.py#L378
                 env_prune_large_gs: bool = True,
                 env_prune_visibility: bool = False,
                 env_max_scene_threshold: float = 0.1,  # default 0.1, same as the original 2DGS
                 env_max_screen_threshold: float = None,  # not used in the original 3DGS/2DGS, they wrote a bug, though `max_screen_threshold=20`
                 env_min_weight_threshold: float = None,
                 # EasyVolcap additional densify & pruning tricks
                 env_screen_until_iter: int = int(4000 / 60 * cfg.runner_cfg.epochs),
                 env_split_screen_threshold: float = None,
                 env_min_gradient: float = None,
                 # Rendering configs
                 env_white_bg: bool = False,  # always set to False !!!
                 env_bg_brightness: float = 0.0,  # used in the original renderer

                 # Reflection related parameters
                 render_reflection: bool = True,  # default is True here
                 render_reflection_start_iter: int = 3000,  # need a initial geometry to model reflection
                 detach: bool = False,  # detach the reflected rays for training the reflection model
                 
                 # Refraction related parameters
                 render_transmission: bool = True, 
                 render_transmission_start_iter: int = 3000,
                 
                 trans_env_initialized: bool = False,  # whether the transmission environment Gaussians are initialized

                 # Ray tracing configs
                 use_trans_env_tracing: bool = True, #! False to ablate only using rasterization 
                 use_optix_tracing: bool = True,
                 use_base_tracing: bool = False,
                 tracing_backend: str = 'cpp',
                 env_max_gs: int = 7e5,  # control the maximum number of gaussians
                 env_max_gs_threshold: float = 0.9,  # percentage of the visibility pruning
                 prune_visibility: bool = True,  # whether to prune the gaussians based on accumulated weights
                 max_trace_depth: int = 0,
                 specular_threshold: float = 0.0,  # specular threshold for reflection rendering
                 n_sample_dirs: int = 1, # number of sampled reflected directions
                 specular_filtering_start_iter: int = -1,  # start to filter pixels with large specular values
                 specular_filtering_percent: float = 0.75,  # percentage of pixels to be filtered
                 acc_filtering_start_iter: int = -1,  # start to filter pixels with large accumulated weights
                 multi_sampling_start_iter: int = -1,  # start to use multi-sample for reflection rendering

                 # Multi-view consistency loss parameters
                 multi_view_patch_size: int = 3,  # patch size for NCC computation
                 multi_view_pixel_noise_th: float = 1.0,  # pixel noise threshold for geometric consistency
                 multi_view_sample_num: int = 102400,  # number of pixels to sample for multi-view loss
                 wo_use_geo_occ_aware: bool = False, # not use the occlusion-aware weighting in geometric consistency

                 # Default parameters for Gaussian2DSampler
                 **kwargs,
                 ):
        # Inherit from the default `VolumetricVideoDataset`
        call_from_cfg(super().__init__,
                      kwargs,
                      network=network,
                      sh_start_iter=sh_start_iter,
                      densify_until_iter=densify_until_iter,
                      render_reflection=render_reflection,
                      use_optix_tracing=use_optix_tracing,
                      tracing_backend=tracing_backend,
                      prune_visibility=prune_visibility,
                      max_trace_depth=max_trace_depth,
                      specular_threshold=specular_threshold)

        # 3DGS-DR related configs
        self.init_densification_interval = init_densification_interval
        self.norm_densification_interval = norm_densification_interval
        self.normal_prop_until_iter = normal_prop_until_iter
        self.normal_prop_interval = normal_prop_interval
        self.opacity_lr0_interval = opacity_lr0_interval
        self.opacity_lr = opacity_lr
        self.color_sabotage_until_iter = color_sabotage_until_iter
        self.color_sabotage_interval = color_sabotage_interval
        self.reset_specular_all = reset_specular_all

        # Reflection related parameters
        self.use_trans_env_tracing = use_trans_env_tracing
        self.use_base_tracing = use_base_tracing
        self.render_reflection_start_iter = render_reflection_start_iter
        self.render_transmission_start_iter = render_transmission_start_iter #! new
        self.n_sample_dirs = n_sample_dirs
        self.detach = detach
        self.specular_filtering_start_iter = specular_filtering_start_iter
        self.specular_filtering_percent = specular_filtering_percent
        self.acc_filtering_start_iter = acc_filtering_start_iter
        self.multi_sampling_start_iter = multi_sampling_start_iter

        # Multi-view consistency loss parameters
        self.multi_view_patch_size = multi_view_patch_size
        self.multi_view_pixel_noise_th = multi_view_pixel_noise_th
        self.multi_view_sample_num = multi_view_sample_num
        self.wo_use_geo_occ_aware = wo_use_geo_occ_aware

        # Environment Gaussian related parameters
        self.env_preload_gs = env_preload_gs
        self.env_bounds = env_bounds
        # Environment SH related parameters
        self.env_sh_deg = env_sh_deg
        self.env_init_sh_deg = env_init_sh_deg
        self.env_sh_start_iter = env_sh_start_iter
        self.env_sh_update_iter = env_sh_update_iter
        # Environment opacity and scale parameters
        self.env_init_occ = env_init_occ
        # Densify & pruning parameters
        self.env_densify_from_iter = env_densify_from_iter
        self.env_densify_until_iter = env_densify_until_iter
        self.env_densification_interval = env_densification_interval
        self.env_opacity_reset_interval = env_opacity_reset_interval
        self.env_densify_grad_threshold = env_densify_grad_threshold
        self.env_min_opacity = env_min_opacity

        self.trans_env_densify_grad_threshold = trans_env_densify_grad_threshold
        self.trans_env_min_opacity = trans_env_min_opacity

        self.env_densify_size_threshold = env_densify_size_threshold
        self.env_prune_large_gs = env_prune_large_gs
        self.env_prune_visibility = env_prune_visibility
        self.env_max_scene_threshold = env_max_scene_threshold
        self.env_max_screen_threshold = env_max_screen_threshold
        self.env_min_weight_threshold = env_min_weight_threshold
        
        # trans env setting
        self.trans_env_densify_from_iter = trans_env_densify_from_iter
        self.trans_env_densify_until_iter = trans_env_densify_until_iter
        self.trans_env_densification_interval = trans_env_densification_interval
        self.trans_env_opacity_reset_interval = trans_env_opacity_reset_interval

        # EasyVolcap additional densify & pruning tricks
        self.env_screen_until_iter = env_screen_until_iter
        self.env_split_screen_threshold = env_split_screen_threshold
        self.env_min_gradient = env_min_gradient
        self.env_max_gs = env_max_gs
        self.env_max_gs_threshold = env_max_gs_threshold
        # Store the last output for updating the gaussians
        self.last_output_env_refl = None
        self.last_output_env_trans = None
        self.trans_env_initialized = trans_env_initialized

        xyz, colors = self.init_env_points(self.env_preload_gs)
        # Create environment Gaussians (G_env)
        self.env = GaussianModel(
            xyz=xyz,
            colors=colors,
            init_occ=self.env_init_occ,
            init_scale=None,
            sh_degree=self.env_sh_deg,
            init_sh_degree=self.env_init_sh_deg,
            spatial_scale=self.spatial_scale,
            xyz_lr_scheduler=self.xyz_lr_scheduler,
            render_reflection=False,
            max_gs=self.env_max_gs,
            max_gs_threshold=self.env_max_gs_threshold
        )
        
        if not self.training:  
            self.trans_env_initialized = True  # for inference, we assume the transmission Gaussians are already initialized
        
        # Update `self.pipe`
        self.pipe.convert_SHs_python = True  # enable SH -> RGB conversion in Python
        if self.use_base_tracing: self.pipe.convert_SHs_python = False
        self.pipe_env = copy.deepcopy(self.pipe)
        self.pipe_env.convert_SHs_python = False
        
        self.pipe_trans_env = copy.deepcopy(self.pipe_env)
       
        # Rendering configs of environment Gaussian
        self.env_white_bg = env_white_bg
        self.env_bg_brightness = 1. if env_white_bg else env_bg_brightness
        self.env_bg_channel = 3
        self.env_bg_color = make_buffer(torch.Tensor([self.env_bg_brightness] * self.env_bg_channel))

        # Time statistics
        self.times = []
        
    def init_env_points(self, ply_file: str = None, S: int = 32, N: int = 5):
        # Try to load the ply file
        try:
            xyz, rgb = load_sfm_ply(ply_file)  # (P, 3), (P, 3)
            log(yellow(f"Loaded the point cloud from {ply_file}."))
            xyz = torch.as_tensor(xyz, dtype=torch.float)
            rgb = torch.as_tensor(rgb, dtype=torch.float)  # already normalized to [0, 1]
        # If the file does not exist, generate random points and save them
        except:
            log(yellow(f"Failed to load the point cloud from {ply_file}, generating random points."))
            xyz = sample_points_subgrid(torch.as_tensor(self.env_bounds), S, N).float()  # (P, 3)
            rgb = torch.rand(xyz.shape, dtype=torch.float) / 255.0  # (P, 3)
            save_sfm_ply(ply_file, xyz.numpy(), rgb.numpy() * 255.0)

        return xyz, rgb
    
    def init_trans_env(self, optimizer: torch.optim.Optimizer, init_from=None):
        # log(yellow_slim(f"Initializing transmission Gaussians from the base model at iteration {self.render_reflection_start_iter}"))
        
        if init_from == "base":
            self.trans_env.replace_with(self.pcd, optimizer, 'sampler.trans_env.')
        elif init_from == "env":
            self.trans_env.replace_with(self.env, optimizer, 'sampler.trans_env.')
            self.pcd.reset_transmission_coeff(optimizer=optimizer, prefix='sampler.pcd.')
        else:
            # Initialize the transmission Gaussians with the same points as the environment Gaussians
            pass
        
        self.trans_env_initialized = True

    @torch.no_grad()
    def update_dif_gaussians(self, batch: dotdict):
        if not self.training: return

        # Update the densification interval
        if batch.meta.iter < self.render_reflection_start_iter: self.densification_interval = self.init_densification_interval
        elif batch.meta.iter < self.normal_prop_until_iter: self.densification_interval = self.norm_densification_interval
        else: self.densification_interval = self.init_densification_interval

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.num_pts = self.pcd.number
        batch.output.scalar_stats = scalar_stats
        # Log the last opacity reset iteration
        batch.output.last_opacity_reset_iter = self.opacity_reset_interval * (iter // self.opacity_reset_interval)

        # Update the learning rate
        self.pcd.update_learning_rate(iter.item(), optimizer, prefix='sampler.pcd.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.densify_until_iter and iter % self.sh_update_iter == 0 and self.sh_start_iter is not None and iter > self.sh_start_iter:
            changed = self.pcd.oneupSHdegree()
            if changed: log(yellow_slim(f'[ONEUP SH DEGREE] sh_deg: {self.pcd.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            pcd: GaussianModel = self.pcd

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: pcd.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: pcd.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Update gaussian splatting radii for update
            if not self.use_optix_tracing:
                radii = output.radii
                pcd.max_radii2D[visibility_filter] = torch.max(pcd.max_radii2D[visibility_filter], radii[visibility_filter])

            # Perform densification and pruning
            if iter > self.densify_from_iter and iter % self.densification_interval == 0:
                log(yellow_slim(f'Start updating gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                split_screen_threshold = self.split_screen_threshold if iter < self.screen_until_iter else None
                max_screen_threshold = self.max_screen_threshold if iter > self.opacity_reset_interval else None
                # Perform actual densification and pruning
                pcd.densify_and_prune(
                    self.min_opacity,
                    self.min_gradient,
                    self.densify_grad_threshold,
                    self.densify_size_threshold,
                    split_screen_threshold,
                    self.max_scene_threshold,
                    max_screen_threshold,
                    self.min_weight_threshold,
                    self.prune_visibility,
                    optimizer,
                    self.prune_large_gs,
                    prefix='sampler.pcd.'
                )
                log(yellow_slim('Densification and pruning done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f} ' +
                                f'number of points: {pcd.get_xyz.shape[0]}'))

            opacity_reset_flag = False
            # Perform opacity reset
            if iter % self.opacity_reset_interval == 0:
                # Reset the opacity of the gaussians to 0.01 (default)
                pcd.reset_opacity(optimizer=optimizer, prefix='sampler.pcd.')
                #! add reset_transparency
                pcd.reset_transmission_coeff(optimizer=optimizer, prefix='sampler.pcd.')
                # pcd.reset_roughness(optimizer=optimizer, prefix='sampler.pcd.')
                
                log(yellow_slim('Resetting opacity done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f}'))
                opacity_reset_flag = True

                if iter > self.opacity_reset_interval and iter > self.render_reflection_start_iter and 'specular' in output:
                    # Reset the specular of the gaussians to 0.001 (default)
                    pcd.reset_specular(
                        reset_specular=self.init_specular,
                        reset_specular_all=self.reset_specular_all,
                        optimizer=optimizer,
                        prefix='sampler.pcd.'
                    )
                    log(yellow_slim('Resetting specular done! ' +
                                    f'min specular: {pcd.get_specular.min().item():.4f} ' +
                                    f'max specular: {pcd.get_specular.max().item():.4f}'))

            if self.opacity_lr0_interval > 0 and iter % self.opacity_lr0_interval == 0 and self.render_reflection_start_iter < iter <= self.normal_prop_until_iter:
                pcd.update_learning_rate_by_name(
                    name='_opacity',
                    lr=self.opacity_lr,
                    optimizer=optimizer,
                    prefix='sampler.pcd.'
                )

            # Make individual color sabotage
            if self.render_reflection_start_iter < iter <= self.color_sabotage_until_iter and iter % self.color_sabotage_interval == 0 and not opacity_reset_flag:
                pcd.distort_color(optimizer=optimizer, prefix='sampler.pcd.')

            if self.render_reflection_start_iter < iter <= self.normal_prop_until_iter and iter % self.normal_prop_interval == 0 and not opacity_reset_flag:
                # Reset the opacity of the gaussians to 0.9 (default)
                pcd.enlarge_opacity(optimizer=optimizer, prefix='sampler.pcd.')
                pcd.enlarge_scaling(optimizer=optimizer, prefix='sampler.pcd.')
                if self.opacity_lr0_interval > 0 and iter != self.normal_prop_until_iter:
                    pcd.update_learning_rate_by_name(
                        name='_opacity',
                        lr=0.0,
                        optimizer=optimizer,
                        prefix='sampler.pcd.'
                    )

    @torch.no_grad()
    def freeze_dif_gaussians(self, batch: dotdict):
        log(yellow_slim(f'Freezing the diffuse Gaussians at iteration {batch.meta.iter}'))
        freeze_module(self.pcd)
    
    @torch.no_grad()
    def update_env_gaussians(self, batch: dotdict):
        if not self.training: return

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.env_num_pts = self.env.number
        batch.output.scalar_stats = scalar_stats

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output_env_refl  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer
        # Return if we're not in the update iteration
        if iter <= self.render_reflection_start_iter: return

        # Update the learning rate
        self.env.update_learning_rate(iter.item(), optimizer, prefix='sampler.env.')
        
        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.env_densify_until_iter and iter % self.env_sh_update_iter == 0 and self.env_sh_start_iter is not None and iter > self.env_sh_start_iter:
            changed = self.env.oneupSHdegree()
            if changed: log(green_slim(f'[ONEUP SH DEGREE] sh_deg: {self.env.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.env_densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            env: GaussianModel = self.env

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: env.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: env.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Perform densification and pruning
            if iter > self.env_densify_from_iter and iter % self.env_densification_interval == 0:
                log(green_slim(f'Start updating gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                env_split_screen_threshold = self.env_split_screen_threshold if iter < self.env_screen_until_iter else None
                env_max_screen_threshold = self.env_max_screen_threshold if iter > self.env_opacity_reset_interval else None
                # Perform actual densification and pruning
                env.densify_and_prune(
                    self.env_min_opacity,
                    self.env_min_gradient,
                    self.env_densify_grad_threshold,
                    self.env_densify_size_threshold,
                    env_split_screen_threshold,
                    self.env_max_scene_threshold,
                    env_max_screen_threshold,
                    self.env_min_weight_threshold,
                    self.env_prune_visibility,
                    optimizer,
                    self.env_prune_large_gs,
                    prefix='sampler.env.'
                )
                log(green_slim('Densification and pruning done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f} ' +
                                f'number of points: {env.get_xyz.shape[0]}'))

            # Perform opacity reset
            if iter % self.env_opacity_reset_interval == 0:
                env.reset_opacity(optimizer=optimizer, prefix='sampler.env.')
                log(green_slim('Resetting opacity done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f}'))
                
                
    @torch.no_grad()
    def update_env_trans_gaussians(self, batch: dotdict):
        if not self.training: return

        env: GaussianModel = self.trans_env #! change to self.trans_pcd for seperate Environment Gaussian Training 

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.trans_env_num_pts = env.number
        batch.output.scalar_stats = scalar_stats

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output_env_trans  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer
        # Return if we're not in the update iteration
        if iter <= self.render_transmission_start_iter: return

        # Update the learning rate
        env.update_learning_rate(iter.item(), optimizer, prefix='sampler.trans_env.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.trans_env_densify_until_iter and iter % self.env_sh_update_iter == 0 and self.env_sh_start_iter is not None and iter > self.env_sh_start_iter:
            changed = env.oneupSHdegree()
            if changed: log(green_slim(f'[ONEUP SH DEGREE] sh_deg: {env.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.trans_env_densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: env.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: env.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Perform densification and pruning
            if iter > self.trans_env_densify_from_iter and iter % self.trans_env_densification_interval == 0:
                log(green_slim(f'Start updating gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                env_split_screen_threshold = self.env_split_screen_threshold if iter < self.env_screen_until_iter else None
                env_max_screen_threshold = self.env_max_screen_threshold if iter > self.env_opacity_reset_interval else None
                # Perform actual densification and pruning[]
                env.densify_and_prune(
                    self.trans_env_min_opacity,
                    self.env_min_gradient,
                    self.trans_env_densify_grad_threshold,
                    self.env_densify_size_threshold,
                    env_split_screen_threshold,
                    self.env_max_scene_threshold,
                    env_max_screen_threshold,
                    self.env_min_weight_threshold,
                    self.env_prune_visibility,
                    optimizer,
                    self.env_prune_large_gs,
                    prefix='sampler.trans_env.'
                )
                log(green_slim('Densification and pruning done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f} ' +
                                f'number of points: {env.get_xyz.shape[0]}'))

            # Perform opacity reset
            if iter % self.env_opacity_reset_interval == 0:
                env.reset_opacity(optimizer=optimizer, prefix='sampler.trans_env.')
                log(green_slim('Resetting opacity done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f}'))

    def store_dif_gaussian_output(self, middle: dotdict, batch: dotdict):
        # Reshape and permute the middle output
        middle = self.store_gaussian_output(middle, batch)

        output = dotdict()
        # Store the output for supervision and visualization
        output.acc_map       = middle.acc_map         # (B, P, 1)
        output.dpt_map       = middle.dpt_map         # (B, P, 1)
        output.pcd_dpt_map   = middle.dpt_map           # (B, P, 1) for supervision
        output.norm_map      = middle.norm_map        # (B, P, 3)
        output.dist_map      = middle.dist_map        # (B, P, 1)
        output.surf_norm_map = middle.surf_norm_map   # (B, P, 3)
        output.bg_color      = torch.full_like(output.norm_map, self.bg_brightness)  # only for training and comparing with gt
        
         # New: per-pixel opacity of base (pcd) path
        output.pcd_opacity   = middle.acc_map         # (B, P, 1)

        
        # Reflectance and Transmission related outputs based on TransparentGS
        # Assuming the renderer now outputs ior_map and trans_coeff_map
        if self.render_reflection and 'ior_map' in middle and 'trans_map' in middle:
            # Get ray directions for Fresnel calculation
            _, ray_d, _, _, _, _ = self.get_camera_rays(batch, n_rays=self.n_rays, patch_size=self.patch_size)
            
            # Fresnel calculation (Schlick's approximation)
            view_dir = -normalize(ray_d)
            normal = normalize(output.norm_map)
            cos_theta = torch.sum(view_dir * normal, dim=-1, keepdim=True).clamp(min=0.0)
            
            ### ----- uncomment to use ior  ----- ###
            # n1 = torch.ones_like(middle.ior_map)  # (B, P, 1), assuming air IOR is 1.0
            # n2 = middle.ior_map # (B, P, 1), from rendered Gaussians, ensure IOR is >= 1
            
            # # Base normal-incidence reflectance from IOR
            # r0_base = ((n1 - n2) / (n1 + n2))**2
            ### ---------------------------------- ###

            # Store for visualization or other purposes (available to later stages)
            output.ior_map = middle.ior_map
            output.trans_map = middle.trans_map  # (B, P, 1)
            
            output.spec_map = middle.spec_map  # (B, P, 1) for specular component

            # Learnable residual on F0 from rough_map (bounded via tanh). Acts as a data-driven correction while keeping physics.
            # rmax = 0.3
            # deltaF0 = rmax * output.rough_map # roughness already has been normalized to [0, 1] in the renderer (overall range: [0, 0.5])
            # F0 = (r0_base).clamp(0.0, 0.98)

            # fresnel_reflectance = F0 + (1 - F0) * (1 - cos_theta)**5
            fresnel_reflectance = 0.04 + (1 - 0.04) * (1 - cos_theta)**5 # assuming F0 of 0.04 for dielectrics, can be learned as well
            
            # Weights for specular and transmission components
            output.fresnel_refl = fresnel_reflectance.clamp(0.0, 1.0)  # w_specular = R
            output.fresnel_trans = (1.0 - fresnel_reflectance).clamp(0.0, 1.0)  # w_transmission = 1 - R

            # Also store F0 and its hemispherical average for energy-conserving mixing later
            output.f0_map = 0.04 * torch.ones_like(fresnel_reflectance)  # assuming dielectric F0=0.04, can be learned as well
        
            
        # The diffuse RGB output
        output.rgb_map       = middle.rgb_map          # (B, P, 3)

        # Don't forget the iteration number for later supervision retrieval
        output.iter = batch.meta.iter
        return output

    def get_reflect_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                         output: dotdict, batch: dotdict, trans_ref: bool = False):
        # Compute the reflected rays direction, -d+d' = -2(d·n)n -> d' = d - 2(d·n)n
        norm_map = output.norm_map if not trans_ref else output.trans_norm_map
        norm = normalize(norm_map)
        ref_d = ray_d - 2 * torch.sum(ray_d * norm, dim=-1, keepdim=True) * norm  # (B, P, 3)

        # Compute the surface coordinate as the intersection point
        depth_map = output.dpt_map if not trans_ref else output.trans_dpt_map  # (B, P, 1)
        ref_o = ray_o + ray_d * depth_map.detach()  # (B, P, 3)

        # Store the reflected rays for later supervision
        if not trans_ref:
            output.ref_o = ref_o  # (B, P, 3)
            output.ref_d = ref_d  # (B, P, 3)
        else:
            output.trans_ref_o = ref_o  # (B, P, 3)
            output.trans_ref_d = ref_d  # (B, P, 3)

        # Prepare for multi-sampling and specular filtering
        is_specular_filtering = self.specular_filtering_start_iter > 0 and batch.meta.iter >= self.specular_filtering_start_iter
        is_acc_filtering = self.acc_filtering_start_iter > 0 and batch.meta.iter >= self.acc_filtering_start_iter
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        if is_specular_filtering or is_acc_filtering:
            # Only perform reflection tracing on pixels with high specular values or accumulated weights
            if is_specular_filtering:
                ref_msk = output.spec_map[..., 0] > torch.quantile(output.spec_map[..., 0], self.specular_filtering_percent)
            else:
                ref_msk = output.acc_map[..., 0] > 0.75
            ref_o = ref_o[ref_msk][None]  # (N, S, 3)
            ref_d = ref_d[ref_msk][None]  # (N, S, 3)
            # Store the specular mask for later scattering
            output.ref_msk = ref_msk  # (B, P)

        if not (is_specular_filtering or is_acc_filtering):
            # This branch is for compatibility with the original code
            ref_o = ref_o.reshape(H, W, 3)  # (H, W, 3)
            ref_d = ref_d.reshape(H, W, 3)  # (H, W, 3)

        if self.detach: return ref_o.detach(), ref_d.detach()
        else: return ref_o, ref_d
        
    def get_transmitted_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                             output: dotdict, batch: dotdict):
        # For simple transmission (thin objects), the direction is the same as the incoming ray
        trans_d = ray_d  # (B, P, 3)

        # The origin of the transmitted ray is the surface intersection point
        trans_o = ray_o + ray_d * (output.dpt_map.detach())  # (B, P, 3)

        # Store the transmitted rays for later supervision or visualization
        output.trans_o = trans_o  # (B, P, 3)
        output.trans_d = trans_d  # (B, P, 3)

        
        is_transparency_filtering = False # Placeholder for future implementation
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        if is_transparency_filtering:
            # Only perform transmission tracing on pixels with high transparency values
            trans_msk = output.trans_map[..., 0] > 0.1 # Example threshold
            trans_o = trans_o[trans_msk][None]  # (N, S, 3)
            trans_d = trans_d[trans_msk][None]  # (N, S, 3)
            # Store the transparency mask for later scattering
            output.trans_msk = trans_msk  # (B, P)
        
        if not is_transparency_filtering:
            # Default behavior: reshape for the renderer
            trans_o = trans_o.reshape(H, W, 3)  # (H, W, 3)
            trans_d = trans_d.reshape(H, W, 3)  # (H, W, 3)

        if self.detach: return trans_o.detach(), trans_d.detach()
        else: return trans_o, trans_d

    def store_env_gaussian_output(self, middle_reflect: Optional[dotdict], middle_transmit: Optional[dotdict], 
                                 middle_trans_reflect: Optional[dotdict], output: dotdict, batch: dotdict):
        
        '''    
            # These are the weights for specular and transmission components
            
            # Store for visualization or other purposes
            output.ior_map = middle.ior_map
            output.trans_map = middle.trans_map  # (B, P, 1)
        '''
        
        # Get reflected color (primary reflection)
        if middle_reflect is not None:
            middle_reflect = self.store_gaussian_output(middle_reflect, batch)
            reflected_color = middle_reflect.rgb_map
        else:
            reflected_color = torch.zeros_like(output.rgb_map)
        
        # Get transmitted color
        if middle_transmit is not None:
            middle_transmit = self.store_gaussian_output(middle_transmit, batch)
            transmitted_color = middle_transmit.rgb_map
            # Extract spec_map from transmission for secondary reflection weighting
            trans_spec_map = middle_transmit.spec_map if 'spec_map' in middle_transmit else torch.zeros_like(output.rgb_map[..., :1])
        else:
            transmitted_color = torch.zeros_like(output.rgb_map)
            trans_spec_map = torch.zeros_like(output.rgb_map[..., :1])

        # Get secondary reflected color (reflection from transmission surface)
        if middle_trans_reflect is not None:
            middle_trans_reflect = self.store_gaussian_output(middle_trans_reflect, batch)
            secondary_reflected_color = middle_trans_reflect.rgb_map
        else:
            secondary_reflected_color = torch.zeros_like(output.rgb_map)
            
        # -----------------------------RENDER-----RENDER-----RENDER-----RENDER----RENDER----RENDER------------------------------------  
        
        # Prepare for multi-sampling and specular filtering
        is_specular_filtering = self.specular_filtering_start_iter > 0 and batch.meta.iter >= self.specular_filtering_start_iter
        is_acc_filtering = self.acc_filtering_start_iter > 0 and batch.meta.iter >= self.acc_filtering_start_iter
        

        # The object's own color from the first render pass (pure base color)
        color_interaction = output.rgb_map

        # Energy-conserving mixing with secondary reflection
        # Primary reflection weight
        ks1 = output.spec_map + (1 - output.spec_map) * output.fresnel_refl  # Combined primary reflection coefficient (R)
        ks1 = ks1.clamp(0.0, 1.0)
        
        # Split remaining energy into diffuse and initial transmission
        kd = (1.0 - ks1) * (1.0 - output.trans_map)  # Diffuse weight
        kt_initial = (1.0 - ks1) * output.trans_map  # Initially transmitted energy weight

        # Split transmitted energy at the second surface
        # trans_spec_map is the reflectivity of the surface hit by the transmitted ray
        if middle_reflect is not None and middle_trans_reflect is not None:
            ks2 = (kt_initial * trans_spec_map)
            kt = kt_initial * (1.0 - trans_spec_map)  # Final transmission weight (energy conservation)
        else:
            ks2 = torch.zeros_like(kt_initial)
            kt = kt_initial  # No secondary reflection, all goes to transmission
        
        # Final color mixing with 4 paths, ensuring energy conservation
        output.rgb_map = kd * color_interaction + \
                         ks1 * reflected_color + \
                         kt * transmitted_color + \
                         ks2 * secondary_reflected_color

        # energy conservation loss
        energy_sum = kd + ks1 + kt + ks2
        output.energy_sum = energy_sum  # for supervision
    
        # for visualization purposes
        output.ref_rgb_map = ks1 * reflected_color
        output.trans_rgb_map = kt * transmitted_color
        output.secondary_ref_rgb_map = ks2 * secondary_reflected_color  # New: secondary reflection
        output.dif_rgb_map = kd * color_interaction
        
        # Store spec_map from transmission for supervision
        output.trans_spec_map = trans_spec_map
        
        # Store the environment Gaussian output for supervision
        if middle_reflect is not None:
            output.env_opacity = middle_reflect.acc_map
        else:
            output.env_opacity = None
            
        if middle_transmit is not None:
            output.trans_env_opacity = middle_transmit.acc_map
        else:
            output.trans_env_opacity = None
            
        return output

    def forward(self, batch: dotdict):
        # Maybe update diffuse Gaussians: densification & pruning
        self.update_dif_gaussians(batch)
        # Maybe update environment Gaussians: densification & pruning
        self.update_env_gaussians(batch)
        
        if self.trans_env_initialized:
            self.update_env_trans_gaussians(batch)
        
        # Initialize transmission Gaussians at the specified iteration
        if self.training and not self.trans_env_initialized and batch.meta.iter >= self.render_transmission_start_iter:
            print(yellow_slim(f'Initializing transmission Gaussians at iteration {batch.meta.iter}'))
            self.init_trans_env(cfg.runner.optimizer, init_from="base")
        
            
        # Prepare the camera transformation for Gaussian
        viewpoint_camera = to_x(prepare_gaussian_camera(batch), torch.float)
        batch.viewpoint_camera = viewpoint_camera  # store for later use

        # Compute the camera ray origins and directions, and reflected rays
        ray_o, ray_d, coords, _, _, _ = self.get_camera_rays(
            batch,
            n_rays=self.n_rays,
            patch_size=self.patch_size
        )
        # Shape things
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

            
        '''
        🇧​​​​​🇦​​​​​🇸​​​​​🇪​​​​​ 🇬​​​​​🇦​​​​​🇺​​​​​🇸​​​​​🇸​​​​​🇮​​​​​🇦​​​​​🇳​​​​​ (interaction gaussians)
        '''

        # Invoke hardware ray tracer
        if self.use_base_tracing:
            if self.tracing_backend == 'cpp':
                dif_output = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ray_o.reshape(H, W, 3),
                    ray_d.reshape(H, W, 3),
                    self.pcd,
                    self.pipe,
                    self.bg_color,
                    0,
                    self.specular_threshold,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')
        # Rasterize diffuse Gaussians to image, obtain their radii (on screen)
        else:
            dif_output = self.render_gaussians( # use this
                viewpoint_camera,
                self.pcd,
                self.pipe,
                self.bg_color,
                self.scale_mod,
                override_color=None
            )


        # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
        if self.training: self.last_output = dif_output
        # Prepare output for supervision and visualization
        output = self.store_dif_gaussian_output(dif_output, batch)
        valid_mask = (output.acc_map > 0.9).detach() # 임계값 0.9는 조절 가능
        
        '''
        🇪​​​​​🇳​​​​​🇻​​​​​🇮​​​​​🇷​​​​​🇴​​​​​🇳​​​​​🇲​​​​​🇪​​​​​🇳​​​​​🇹​​​​​ 🇬​​​​​🇦​​​​​🇺​​​​​🇸​​​​​🇸​​​​​🇮​​​​​🇦​​​​​🇳​​​​​ (transmission gaussians)
        '''
        env_output_trans = None
        env_output_trans_refl = None  # Secondary reflection from transmitted surface
        if batch.meta.iter >= self.render_transmission_start_iter:
            # Compute the transmitted rays origins and directions
            trans_o, trans_d = self.get_transmitted_rays(ray_o, ray_d, coords, output, batch)

            # Invoke hardware ray tracer for transmission
            if self.tracing_backend == 'cpp':
                env_output_trans = self.diffop_trans.render_gaussians(
                    viewpoint_camera,
                    trans_o,
                    trans_d,
                    self.trans_env,  # Changed from self.env to self.pcd for testing
                    self.pipe_trans_env,  # Changed from self.pipe_env to self.pipe to match pcd
                    self.env_bg_color,  # Changed from self.env_bg_color to self.bg_color to match pcd
                    # self.env,
                    # self.pipe_env,
                    # self.env_bg_color,
                    0,
                    start_from_first=False,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
                
                # ! New: Conditional direct rasterization of trans_env for plausibility and depth comparison
                if cfg.model_cfg.supervisor_cfg.plausibility_loss_weight > 0 or cfg.model_cfg.supervisor_cfg.trans_guidance_loss_weight > 0:
                    self.pipe_trans_env.convert_SHs_python = True
                    trans_direct_output = self.render_gaussians(
                        viewpoint_camera, self.trans_env, self.pipe_trans_env, self.env_bg_color, self.scale_mod
                    )
                    self.pipe_trans_env.convert_SHs_python = False
                    trans_direct_output = self.store_gaussian_output(trans_direct_output, batch)
                    if cfg.model_cfg.supervisor_cfg.plausibility_loss_weight > 0:
                        output.trans_env_acc_direct = trans_direct_output.acc_map

                        trans_acc_map = (trans_direct_output.acc_map > 0.8).float()
                        batch.trans_msk = trans_acc_map  # Store the mask for supervision

                        output.trans_env_rgb_direct = trans_direct_output.rgb_map
                        output.trans_dpt_map = trans_direct_output.dpt_map
                        output.trans_norm_map = trans_direct_output.norm_map

            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')
            
            
            # Retain gradients after updates
            # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
            if self.training: self.last_output_env_trans = env_output_trans

        '''
        🇪​​​​​🇳​​​​​🇻​​​​​🇮​​​​​🇷​​​​​🇴​​​​​🇳​​​​​🇲​​​​​🇪​​​​​🇳​​​​​🇹​​​​​ 🇬​​​​​🇦​​​​​🇺​​​​​🇸​​​​​🇸​​​​​🇮​​​​​🇦​​​​​🇳​​​​​ (reflective gaussians)
        '''
        
        env_output_refl = None
        env_output_trans_refl = None
        
        if batch.meta.iter >= self.render_reflection_start_iter:
        
            # 1. 1차 반사 광선 계산
            ref_o, ref_d = self.get_reflect_rays(ray_o, ray_d, coords, output, batch, trans_ref=False)
            B, P, _ = ref_o.shape # (B, H*W, 3)

            # 2. 2차 반사 광선 계산 (존재하는 경우)
            if 'trans_dpt_map' in output and self.render_transmission_start_iter <= batch.meta.iter:
                trans_ref_o, trans_ref_d = self.get_reflect_rays(ray_o, ray_d, coords, output, batch, trans_ref=True)
                
                all_rays_o = torch.cat([ref_o, trans_ref_o], dim=1)  # (B, 2*P, 3)
                all_rays_d = torch.cat([ref_d, trans_ref_d], dim=1)  # (B, 2*P, 3)
            else:
                trans_ref_o, trans_ref_d = None, None
                all_rays_o = ref_o
                all_rays_d = ref_d
                
            if self.tracing_backend == 'cpp':
                
                H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
                
                merged_env_output = self.diffop.render_gaussians(
                    viewpoint_camera, all_rays_o, all_rays_d, self.env, self.pipe_env, self.env_bg_color, 0,
                    start_from_first=False, scaling_modifier=self.scale_mod, override_color=None, batch=batch
                )
                if trans_ref_o is not None:
                    num_primary_rays = ref_o.shape[1]  # 1차 반사 광선의 개수 (P)
                    env_output_refl = dotdict()
                    env_output_trans_refl = dotdict()

                    for k, v in merged_env_output.items():
                        if torch.is_tensor(v) and v.dim() == 3 and v.shape[2] == all_rays_o.shape[1]:
                            env_output_refl[k] = v[..., :num_primary_rays]
                            env_output_trans_refl[k] = v[..., num_primary_rays:]
                else:
                    env_output_refl = merged_env_output
                    env_output_trans_refl = None

            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')

            if self.training:
                self.last_output_env_refl = merged_env_output
            else:
                self.last_output_env_refl = None # 추론 시에는 필요 없음
            
            
        # Prepare output for supervision and visualization
        output = self.store_env_gaussian_output(env_output_refl, env_output_trans, env_output_trans_refl, output, batch)
        
        # Compute multi-view consistency if enabled and neighbors are available
        if (batch.meta.iter >= cfg.model_cfg.supervisor_cfg.multi_view_start_iter and 
            (hasattr(batch, 'meta_prev') or hasattr(batch, 'meta_next'))):
            consistency_info = self.compute_multi_view_consistency(batch, output)
            if consistency_info:
                output.update(consistency_info)
                batch.update(consistency_info)
        
        # Transmission map guidance masks computation
        if ('trans_dpt_map' in output and 
            (cfg.model_cfg.supervisor_cfg.trans_guidance_loss_start_iter <= batch.meta.iter or 
             cfg.model_cfg.supervisor_cfg.plausibility_loss_start_iter <= batch.meta.iter)):
            
            # Initialize masks based on geometry
            depth_exists_mask = (output.pcd_dpt_map > 0.0) & (output.trans_dpt_map > 0.0)
            
            # Depth discrepancy calculation
            if cfg.model_cfg.supervisor_cfg.depth_discrepancy_threshold is not None:
                depth_identical_mask = output.trans_dpt_map <= (output.pcd_dpt_map + 1e-3)
                depth_larger_mask = output.trans_dpt_map > (output.pcd_dpt_map + 1e-1)
            else:
                depth_identical_mask = torch.zeros_like(output.pcd_dpt_map, dtype=torch.bool)
                depth_larger_mask = torch.zeros_like(output.pcd_dpt_map, dtype=torch.bool)
            
            # Sky mask (always available)
            not_sky_mask = (batch.sky_masks < 0.5) if 'sky_masks' in batch else torch.ones_like(output.pcd_dpt_map, dtype=torch.bool)
            sky_mask = ~not_sky_mask
            
            # Material-based masks (only available when use_diffren is True)
            if cfg.dataloader_cfg.dataset_cfg.use_diffren:
                # Albedo-based masks
                albedo_luminance = torch.mean(batch.diffuse_albedo, dim=-1, keepdim=True)
                low_albedo_mask = albedo_luminance < cfg.model_cfg.supervisor_cfg.albedo_threshold
                high_albedo_mask = albedo_luminance > (cfg.model_cfg.supervisor_cfg.albedo_threshold + 0.05)
                
                # Base color-based masks
                base_color = torch.mean(batch.basecolor, dim=-1, keepdim=True)
                high_base_color_mask = base_color > cfg.model_cfg.supervisor_cfg.basecolor_threshold
                
                # Material-guided masks
                intrinsics_guided_trans_mask = low_albedo_mask & high_base_color_mask
                intrinsics_guided_opaque_mask = low_albedo_mask & ~high_base_color_mask
                
                # Confident opaque mask (material + geometry)
                confident_opaque_mask = (
                    high_albedo_mask |
                    intrinsics_guided_opaque_mask |
                    sky_mask |
                    depth_identical_mask
                ).float()
                
                # Confident transparent mask (material + geometry)
                confident_transparent_mask = (
                    intrinsics_guided_trans_mask & 
                    not_sky_mask & 
                    depth_exists_mask & 
                    (depth_larger_mask)
                ).float()
            else:
                # Geometry-only masks when diffren is not available
                confident_opaque_mask = (sky_mask | depth_identical_mask).float()
                confident_transparent_mask = (not_sky_mask & depth_exists_mask & depth_larger_mask).float()
            
            # Store final guidance masks
            batch.confident_transparent_mask = confident_transparent_mask
            batch.confident_opaque_mask = confident_opaque_mask
    
        # Update the output to the batch
        batch.output.update(output)
        
    def compute_multi_view_consistency(self, batch: dotdict, output: dotdict):
        """
        Compute multi-view consistency including geometric and photometric consistency.
        
        Args:
            batch: Input batch containing neighbor metadata, depth, and images  
            output: Current frame's output containing depth map and normals
            
        Returns:
            dict containing consistency masks and loss components
        """
        viewpoint_cam = batch.viewpoint_camera
        
        patch_size = self.multi_view_patch_size
        sample_num = self.multi_view_sample_num
        pixel_noise_th = self.multi_view_pixel_noise_th
        total_patch_size = (patch_size * 2 + 1) ** 2
        
        ## compute geometry consistency mask and loss
        H, W = int(batch.H), int(batch.W)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        pixels = torch.stack([ix, iy], dim=-1).float().to(output.dpt_map.device)
        
        ref_dpt_map = rearrange(output.dpt_map, '1 (h w) 1 -> 1 h w', h=H, w=W)
        batch.meta_prev.rgb = batch.rgb_prev
        batch.meta_next.rgb = batch.rgb_next
        nearest_cam = to_x(prepare_gaussian_camera(batch.meta_prev, iterations=batch.meta.iter), torch.float)

        output_nearest = self.render_gaussians(nearest_cam, self.pcd, self.pipe, self.bg_color)
        output_nearest = self.store_gaussian_output(output_nearest, batch)
        output_nearest.dpt_map = rearrange(output_nearest.dpt_map, '1 (h w) 1 -> 1 h w', h=H, w=W)
        
        #ANCHOR: Geometric Consistency
        pts = self.pcd.get_points_from_depth(viewpoint_cam, ref_dpt_map)
        pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
        map_z, d_mask = self.pcd.get_points_depth_in_depth_map(nearest_cam, output_nearest.dpt_map, pts_in_nearest_cam)
        
        pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
        pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
        R = torch.tensor(nearest_cam.R_c2w).float().cuda()
        T = torch.tensor(nearest_cam.T.squeeze()).float().cuda()
        pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
        pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
        pts_projections = torch.stack(
                    [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                    pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
        pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
        if not self.wo_use_geo_occ_aware:
            d_mask = d_mask & (pixel_noise < pixel_noise_th)
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0
        else:
            d_mask = d_mask
            weights = torch.ones_like(pixel_noise)
            weights[~d_mask] = 0

        gt_image = rearrange(batch.rgb, '1 (h w) c -> c h w', h=H, w=W)
        gt_image_gray = 0.2989 * gt_image[0:1] + 0.5870 * gt_image[1:2] + 0.1140 * gt_image[2:3]
        
        debug_path = os.path.join('data/result', cfg.exp_name, 'debug')
        os.makedirs(debug_path, exist_ok=True)
        if batch.meta.iter % 200 == 0:
            gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            # if 'app_image' in render_pkg:
            #     img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            # else:
            image = rearrange(output.rgb_map, '1 (h w) c -> c h w', h=H, w=W)
            normal = rearrange(output.norm_map, '1 (h w) c -> c h w', h=H, w=W)
            depth_normal = rearrange(output.surf_norm_map, '1 (h w) c -> c h w', h=H, w=W)
            
            img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
            d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
            
            depth =ref_dpt_map.squeeze().detach().cpu().numpy()
            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
            
            nearest_dpt = output_nearest.dpt_map.reshape(H,W).squeeze().detach().cpu().numpy()
            nearest_dpt_i = (nearest_dpt - nearest_dpt.min()) / (nearest_dpt.max() - nearest_dpt.min() + 1e-20)
            nearest_dpt_i = (nearest_dpt_i * 255).clip(0, 255).astype(np.uint8)
            nearest_dpt_show = cv2.applyColorMap(nearest_dpt_i, cv2.COLORMAP_JET)
            
            # distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
            # distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
            # distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
            # distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
            # image_weight = image_weight.detach().cpu().numpy()
            # image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
            # image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
            # neighbor image
            nearest_image, _ = nearest_cam.get_image()
            nearest_image = rearrange(nearest_image, 'h w c -> c h w')
            nearest_image_show = ((nearest_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            row0 = np.concatenate([nearest_image_show, gt_img_show, img_show, normal_show], axis=1)
            row1 = np.concatenate([nearest_dpt_show, d_mask_show_color, depth_color, depth_normal_show], axis=1)
            image_to_show = np.concatenate([row0, row1], axis=0)
            cv2.imwrite(os.path.join(debug_path, "%05d"%batch.meta.iter + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
        
        if d_mask.sum() > 0:
            geo_loss = ((weights * pixel_noise)[d_mask]).mean()
            with torch.no_grad():
                ## sample mask
                d_mask = d_mask.reshape(-1)
                valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                if d_mask.sum() > sample_num:
                    index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                    valid_indices = valid_indices[index]

                weights = weights.reshape(-1)[valid_indices]
                ## sample ref frame patch
                pixels = pixels.reshape(-1,2)[valid_indices]
                offsets = patch_offsets(patch_size, pixels.device)
                ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                
                H, W = gt_image_gray.squeeze().shape
                pixels_patch = ori_pixels_patch.clone()
                pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                ref_local_n = rearrange(output.norm_map, '1 (h w) c -> h w c', h=H, w=W) # world coordinate
                ref_local_n = ref_local_n @ batch.R.mT  # convert to view space

                depth_z = ref_dpt_map.squeeze()
                Hh, Ww = depth_z.shape
                device = depth_z.device
                
                # pixel -> camera rays (unit)
                u = torch.arange(Ww, device=device, dtype=torch.float32)
                v = torch.arange(Hh, device=device, dtype=torch.float32)
                uu, vv = torch.meshgrid(u, v, indexing='xy')                  # (H,W)
                ones = torch.ones_like(uu)
                pix = torch.stack([uu, vv, ones], dim=-1)                     # (H,W,3)

                Kinv = viewpoint_cam.get_inv_k(scale=1.0)                     # 풀 해상도 기준
                rays = torch.einsum('ij,hwj->hwi', Kinv, pix)                 # (H,W,3)
                rays = rays / (rays.norm(dim=-1, keepdim=True) + 1e-8)        # 단위화

                cos_nv = (ref_local_n * rays).sum(dim=-1).abs().clamp_min(1e-4)
                ref_local_d_map = (depth_z * cos_nv)                          # (H,W)
                ref_local_d = ref_local_d_map.reshape(-1)[valid_indices]      # 샘플 위치만 추출

                # clamping 
                ref_local_d = torch.clamp(ref_local_d, min=1e-1, max=1e3)
                
                # homography
                ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]
                
                H_ref_to_neareast = ref_to_neareast_r[None] - torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                
                ## compute neareast frame patch
                grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                _, nearest_image_gray = nearest_cam.get_image()
                nearest_image_gray = nearest_image_gray.permute(2, 0, 1)
                sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                
                ## compute loss
                ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                mask = ncc_mask.reshape(-1)
                ncc = ncc.reshape(-1) * weights
                ncc = ncc[mask].squeeze()
                if mask.sum() > 0:
                    ncc_loss = ncc.mean()
                    
        # update output for logging loss

        # geo loss 
        output.geo_loss = geo_loss if d_mask.sum() > 0 else torch.tensor(0.).to(output.dpt_map.device)
        output.ncc_loss = ncc_loss if (d_mask.sum() > 0 and mask.sum() > 0) else torch.tensor(0.).to(output.dpt_map.device)
        
        return {
            'geo_consistency_mask': d_mask.reshape(1, H*W, 1),
            'geo_loss': output.geo_loss,
            'ncc_loss': output.ncc_loss
        }
                