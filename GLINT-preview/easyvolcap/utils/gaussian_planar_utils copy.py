import os
import math
import numpy as np
from plyfile import PlyData, PlyElement

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn import functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.net_utils import make_buffer, make_params, typed
from easyvolcap.utils.graphics_utils import normal_from_depth_image

from simple_knn._C import distCUDA2

from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
    
# from diff_first_surface_rasterization import GaussianRasterizationSettings as FirstSurfaceGaussianRasterizationSettings
# from diff_first_surface_rasterization import GaussianRasterizer as FirstSurfaceGaussianRasterizer


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    
    # Calculate intrinsic matrix directly from viewpoint_cam dict
    # Extract focal lengths and center points from viewpoint_cam
    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
    fx = W / (2 * math.tan(viewpoint_cam.FoVx * 0.5))
    fy = H / (2 * math.tan(viewpoint_cam.FoVy * 0.5))
    cx = W / 2.0
    cy = H / 2.0
    
    intrinsic_matrix = torch.tensor([
        [fx/scale, 0, cx/scale], 
        [0, fy/scale, cy/scale], 
        [0, 0, 1]
    ]).float()

    extrinsic_matrix = viewpoint_cam.world_view_transform.transpose(0,1).contiguous() # cam2world

    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


@torch.jit.script
def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


@torch.jit.script
def getProjectionMatrix(fovx: torch.Tensor, fovy: torch.Tensor, znear: torch.Tensor, zfar: torch.Tensor):
    tanfovy = math.tan((fovy / 2))
    tanfovx = math.tan((fovx / 2))

    t = tanfovy * znear
    b = -t
    r = tanfovx * znear
    l = -r

    P = torch.zeros(4, 4, dtype=znear.dtype, device=znear.device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (r - l)
    P[1, 1] = 2.0 * znear / (t - b)

    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def prepare_gaussian_camera(batch):
    output = dotdict()
    H, W, K, R, T, n, f = batch.H[0], batch.W[0], batch.K[0], batch.R[0], batch.T[0], batch.n[0], batch.f[0]
    cpu_H, cpu_W, cpu_K, cpu_R, cpu_T, cpu_n, cpu_f = batch.meta.H[0], batch.meta.W[0], batch.meta.K[0], batch.meta.R[0], batch.meta.T[0], batch.meta.n[0], batch.meta.f[0]

    output.image_height = cpu_H
    output.image_width = cpu_W

    output.K = K
    output.R = R
    output.T = T

    fl_x = cpu_K[0, 0]  # use cpu K
    fl_y = cpu_K[1, 1]  # use cpu K
    FoVx = focal2fov(fl_x, cpu_W)
    FoVy = focal2fov(fl_y, cpu_H)

    if 'msk' in batch: output.gt_alpha_mask = batch.msk[0]  # FIXME: whatever for now

    output.world_view_transform = getWorld2View(R, T).transpose(0, 1)
    output.projection_matrix = getProjectionMatrix(FoVx, FoVy, n, f).transpose(0, 1)
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix)
    output.camera_center = (-R.mT @ T)[..., 0]  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.FoVx = FoVx
    output.FoVy = FoVy
    output.tanfovx = math.tan(FoVx * 0.5)
    output.tanfovy = math.tan(FoVy * 0.5)

    output.znear = n
    output.zfar = f

    return output


@torch.jit.script
def rgb2sh0(rgb: torch.Tensor):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


@torch.jit.script
def sh02rgb(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@torch.jit.script
def scaling_activation(x):
    return torch.exp(x)


@torch.jit.script
def scaling_inverse_activation(x):
    return torch.log(x.clamp(1e-6, 1e6))


@torch.jit.script
def opacity_activation(x):
    return torch.sigmoid(x)


@torch.jit.script
def inverse_opacity_activation(x):
    return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))


@torch.jit.script
def specular_activation(x):
    return torch.sigmoid(x)


@torch.jit.script
def inverse_specular_activation(x):
    return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))


@torch.jit.script
def ior_activation(x):
    return 1.0 + F.softplus(x)

@torch.jit.script
def inverse_ior_activation(n):
    # 기존: return torch.log(torch.clamp(n - 1.0, 1e-6, 1e6))
    inner_term = torch.exp(n - 1.0) - 1.0
    return torch.log(torch.clamp(inner_term, min=1e-6))


def build_rotation(r):
    """ Build a rotation matrix from a quaternion, the
        default quaternion convention is (w, x, y, z).

    Args:
        r (torch.Tensor), (..., 4): the quaternion.
    
    Returns:
        R (torch.Tensor), (..., 3, 3): the rotation matrix
    """

    # Normalize the quaternion
    s = torch.norm(r, dim=-1)[..., None]  # (..., 1)
    q = r / s  # (..., 4)

    # Extract the quaternion components in (w, x, y, z) order
    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # Build the rotation matrix, column-major
    R = torch.zeros(q.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)  # (..., 3, 3)
    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)

    return R


def build_scaling_rotation(s, r):
    L = torch.zeros(s.shape[:-1] + (3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[..., 0, 0] = s[..., 0]
    L[..., 1, 1] = s[..., 1]
    L[..., 2, 2] = s[..., 2]

    L = R @ L
    return L


@torch.jit.script
def build_cov(center: torch.Tensor, s: torch.Tensor, scaling_modifier: float, q: torch.Tensor):
    L = build_scaling_rotation(s * scaling_modifier, q).permute(0, 2, 1)
    T = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=L.device)
    T[:, :3, :3] = L
    T[:,  3, :3] = center
    T[:,  3,  3] = 1
    return T

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

@torch.jit.script
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def build_inverse_scaling_rotation(s, r):
    L = torch.zeros(s.shape[:-1] + (3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[..., 0, 0] = 1 / s[..., 0]
    L[..., 1, 1] = 1 / s[..., 1]
    L[..., 2, 2] = 1 / s[..., 2]

    L = L @ R.mT
    return L


@torch.jit.script
def build_inv_cov(center: torch.Tensor, s: torch.Tensor, scaling_modifier: float, q: torch.Tensor):
    L = build_inverse_scaling_rotation(s * scaling_modifier, q)  # S^{-1} @ R^T
    T = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=L.device)  # (P, 4, 4)
    T[:, :3, :3] = L  # (P, 3, 3)
    T[:, :3, 3:] = -L @ center[..., None]  # (P, 3, 1)
    T[:,  3,  3] = 1
    return T


def get_expon_lr_func(
    lr_init,
    lr_final,
    lr_delay_steps=0,
    lr_delay_mult=1.0,
    max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class GaussianModel(nn.Module):
    
    def __init__(
        self,
        xyz: torch.Tensor = None,
        colors: torch.Tensor = None,
        init_occ: float = 0.1,
        init_scale: torch.Tensor = None,
        sh_degree: int = 3,
        init_sh_degree: int = 0,
        spatial_scale: float = 1.0,
        xyz_lr_scheduler: dotdict = dotdict(max_steps=30000),
        # Reflection related parameters
        render_reflection: bool = False,
        specular_channels: int = 1,
        init_specular: float = 1e-3,
        init_roughness: float = 0.5,
        init_ior: float = 1.5,
        init_transmission_coeff: float = 0.1,
        max_gs: int = 1e6,
        max_gs_threshold: float = 0.9,
    ):
        super().__init__()

        self.setup_functions(
            scaling_activation=scaling_activation,
            scaling_inverse_activation=scaling_inverse_activation,
            opacity_activation=opacity_activation,
            inverse_opacity_activation=inverse_opacity_activation
        )

        # SH realte configs
        self.active_sh_degree = make_buffer(torch.full((1,), init_sh_degree, dtype=torch.long))  # save them, but need to keep a copy on cpu
        self.cpu_active_sh_degree = self.active_sh_degree.item()
        self.max_sh_degree = sh_degree

        # Set scene spatial scale
        self.spatial_scale = spatial_scale

        # Initalize trainable parameters
        self.create_from_pcd(xyz, colors, init_occ, init_scale, specular_channels, init_specular, init_roughness, init_ior, init_transmission_coeff)
        self.render_reflection = render_reflection
        self.specular_channels = specular_channels
        self.init_specular = init_specular
        self.init_roughness = init_roughness

        # Densification related parameters
        self.max_radii2D = make_buffer(torch.zeros(self.get_xyz.shape[0]))
        self.xyz_gradient_accum = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))
        self.denom = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))
        self.xyz_weight_accum = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))  # (P, 1)

        self.max_gs = max_gs
        self.max_gs_threshold = max_gs_threshold

        if xyz_lr_scheduler is not None:
            xyz_lr_scheduler['lr_init'] *= self.spatial_scale
            xyz_lr_scheduler['lr_final'] *= self.spatial_scale
            self.xyz_scheduler = get_expon_lr_func(**xyz_lr_scheduler)
            log(magenta(f'[INIT] Using xyz learning rate scheduler, lr_init: {xyz_lr_scheduler["lr_init"]}, lr_final: {xyz_lr_scheduler["lr_final"]}'))
        else:
            self.xyz_scheduler = None

        # Perform some model messaging before loading
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self.post_handle = self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def setup_functions(
        self,
        scaling_activation=torch.exp,
        scaling_inverse_activation=torch.log,
        opacity_activation=torch.sigmoid,
        inverse_opacity_activation=torch.logit,
        rotation_activation=F.normalize,
        specular_activation=torch.sigmoid,
        specular_inverse_activation=torch.logit,
        roughness_activation=torch.sigmoid,
        roughness_inverse_activation=torch.logit,
        transmission_coeff_activation=torch.sigmoid,
        transmission_coeff_inverse_activation=torch.logit,
        ior_activation=ior_activation,
        ior_inverse_activation=inverse_ior_activation,
    ):
        self.scaling_activation = getattr(torch, scaling_activation) if isinstance(scaling_activation, str) else scaling_activation
        self.opacity_activation = getattr(torch, opacity_activation) if isinstance(opacity_activation, str) else opacity_activation
        self.rotation_activation = getattr(torch, rotation_activation) if isinstance(rotation_activation, str) else rotation_activation
        self.scaling_inverse_activation = getattr(torch, scaling_inverse_activation) if isinstance(scaling_inverse_activation, str) else scaling_inverse_activation
        self.opacity_inverse_activation = getattr(torch, inverse_opacity_activation) if isinstance(inverse_opacity_activation, str) else inverse_opacity_activation
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.inverse_covariance_activation = build_inv_cov

        self.specular_activation = getattr(torch, specular_activation) if isinstance(specular_activation, str) else specular_activation
        self.specular_inverse_activation = getattr(torch, specular_inverse_activation) if isinstance(specular_inverse_activation, str) else specular_inverse_activation
        self.roughness_activation = getattr(torch, roughness_activation) if isinstance(roughness_activation, str) else roughness_activation
        self.roughness_inverse_activation = getattr(torch, roughness_inverse_activation) if isinstance(roughness_inverse_activation, str) else roughness_inverse_activation
        self.transmission_coeff_activation = getattr(torch, transmission_coeff_activation) if isinstance(transmission_coeff_activation, str) else transmission_coeff_activation
        self.transmission_coeff_inverse_activation = getattr(torch, transmission_coeff_inverse_activation) if isinstance(transmission_coeff_inverse_activation, str) else transmission_coeff_inverse_activation
        self.ior_activation = getattr(torch, ior_activation) if isinstance(ior_activation, str) else ior_activation
        self.ior_inverse_activation = getattr(torch, ior_inverse_activation) if isinstance(ior_inverse_activation, str) else ior_inverse_activation
        
    @property
    def device(self):
        return self.get_xyz.device

    @property
    def number(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_specular(self):
        return self.specular_activation(self._specular)
    
    @property
    def get_transmission_coeff(self):
        return self.transmission_coeff_activation(self._transmission_coeff)
    
    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)
    @property
    def get_ior(self):
        return self.ior_activation(self._ior)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1)**2

    def get_rotation_matrix(self):
        from pytorch3d.transforms import quaternion_to_matrix
        return quaternion_to_matrix(self.get_rotation)

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        changed = False
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.cpu_active_sh_degree = self.active_sh_degree.item()
            changed = True
        return changed

    def create_from_pcd(
        self,
        xyz: torch.Tensor,
        colors: torch.Tensor = None,
        opacities: float = 0.1,
        scales: torch.Tensor = None,
        specular_channels: int = 1,
        specular: float = 1e-3,
        roughness: float = 0.5,
        ior: float = 1.5,
        transmission_coeff: float = 0.1,
    ):
        if xyz is None:
            xyz = torch.empty(1, 3, device='cuda')  # by default, init empty gaussian model on CUDA

        features = torch.zeros((xyz.shape[0], 3, self.get_max_sh_channels))
        if colors is not None:
            features[:, :3, 0] = rgb2sh0(colors)
        features[:, 3: 1:] = 0.0

        log(magenta(f'[INIT] NUM POINTS: {xyz.shape[0]}'))

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001))
        scales = torch.log(dist)[...,None].repeat(1, 3)

        rots = torch.zeros((xyz.shape[0], 4))
        rots[:, 0] = 1.0  # identity quaternion

        opacities = inverse_opacity_activation(0.1 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = make_params(xyz)
        self._features_dc = make_params(features[:, :, :1].transpose(1, 2).contiguous())
        self._features_rest = make_params(features[:, :, 1:].transpose(1, 2).contiguous())
        self._scaling = make_params(scales)
        self._rotation = make_params(rots)
        self._opacity = make_params(opacities)

        if not isinstance(specular, torch.Tensor) or len(specular) != len(xyz):
            specular = specular * torch.ones((xyz.shape[0], specular_channels), dtype=torch.float)
        specular = self.specular_inverse_activation(specular)
        if not isinstance(roughness, torch.Tensor) or len(roughness) != len(xyz):
            roughness = roughness * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        roughness = self.roughness_inverse_activation(roughness)
        if not isinstance(transmission_coeff, torch.Tensor) or len(transmission_coeff) != len(xyz):
            transmission_coeff = transmission_coeff * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        transmission_coeff = self.transmission_coeff_inverse_activation(transmission_coeff)
        if not isinstance(ior, torch.Tensor) or len(ior) != len(xyz):
            ior = ior * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        ior = self.ior_inverse_activation(ior)
        
        self._specular = make_params(specular)
        self._roughness = make_params(roughness)
        self._transmission_coeff = make_params(transmission_coeff)
        self._ior = make_params(ior)

    @torch.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if prefix != '' and not prefix.endswith('.'): prefix = prefix + '.'  # special care for when we're loading the model directly
        for name, params in self.named_parameters():
            if f'{prefix}{name}' in state_dict:
                params.data = params.data.new_empty(state_dict[f'{prefix}{name}'].shape)

    @torch.no_grad()
    def _load_state_dict_post_hook(self, module, incompatible_keys):
        # TODO: make this a property that updates the cpu copy on change
        self.cpu_active_sh_degree = self.active_sh_degree.item()

    def distort_color(self, range: float = 0.4, threshold: float = 0.5, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[DISTORT COLOR]'))
        mask_spec = self.get_specular.max(dim=-1).values.flatten() > threshold  # (P,)
        mask_trans = self.get_transmission_coeff.max(dim=-1).values.flatten() > threshold  # (P,)
        
        mask = mask_spec | mask_trans
        
        features_dc = self._features_dc.clone()
        new_features_dc = features_dc + torch.rand_like(features_dc) * range * 2 - range
        new_features_dc[mask] = features_dc[mask]
        new_features_dc.grad = self._features_dc.grad
        self._features_dc = self.replace_tensor_to_optimizer(new_features_dc, "_features_dc", optimizer, prefix)

    def enlarge_scaling(self, ratio: float = 1.5, threshold: float = 0.02, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[ENLARGE SCALING] ENLARGE SCALING BY {ratio}'))
        mask_spec = self.get_specular.max(dim=-1).values.flatten() < threshold  # (P,)
        # mask_trans = self.get_transmission_coeff.max(dim=-1).values.flatten() < threshold  # (P,)
        mask = mask_spec
        new_scaling = self.scaling_inverse_activation(self.get_scaling * ratio)  # (P, 3)
        new_scaling[mask] = self._scaling[mask]  # (P, 3)
        new_scaling.grad = self._scaling.grad
        self._scaling = self.replace_tensor_to_optimizer(new_scaling, '_scaling', optimizer, prefix)

    def enlarge_opacity(self, enlarge_opacity: float = 0.9, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[ENLARGE OPACITY] ENLARGE OPACITY TO {enlarge_opacity}'))
        new_opacity = torch.max(self._opacity, self.opacity_inverse_activation(torch.ones_like(self._opacity, ) * enlarge_opacity))
        new_opacity.grad = self._opacity.grad
        self._opacity = self.replace_tensor_to_optimizer(new_opacity, '_opacity', optimizer, prefix)

    def reset_specular(self, reset_specular: float = 0.001, optimizer: Optimizer = None, prefix: str = '', reset_specular_all: bool = False):
        log(yellow_slim(f'[RESET SPECULAR] RESET SPECULAR TO {reset_specular}'))
        if reset_specular_all: new_specular = self.specular_inverse_activation(torch.ones_like(self._specular, ) * reset_specular)
        else: new_specular = torch.min(self._specular, self.specular_inverse_activation(torch.ones_like(self._specular, ) * reset_specular))
        new_specular.grad = self._specular.grad
        self._specular = self.replace_tensor_to_optimizer(new_specular, '_specular', optimizer, prefix)

    def reset_opacity(self, reset_opacity: float = 0.01, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[RESET OPACITY] RESET OPACITY TO {reset_opacity}'))
        new_opacity = torch.min(self._opacity, self.opacity_inverse_activation(torch.ones_like(self._opacity, ) * reset_opacity))
        self._opacity = self.replace_tensor_to_optimizer(new_opacity, '_opacity', optimizer, prefix)
    
    def reset_transmission_coeff(self, reset_transmission_coeff: float = 0.01, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[RESET TRANSMISSION COEFF] RESET TRANSMISSION COEFF TO {reset_transmission_coeff}'))
        new_transmission_coeff = torch.min(self._transmission_coeff, self.transmission_coeff_inverse_activation(torch.ones_like(self._transmission_coeff, ) * reset_transmission_coeff))
        new_transmission_coeff.grad = self._transmission_coeff.grad
        self._transmission_coeff = self.replace_tensor_to_optimizer(new_transmission_coeff, '_transmission_coeff', optimizer, prefix)
        # 애초에 transmission_coeff가 낮았던 불투명한 gaussian들은 reset_transmission_coeff를 통해서도 투명하게 되지 않도록 설정
    
    def reset_roughness(self, reset_roughness: float = 0.01, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[RESET ROUGHNESS] RESET ROUGHNESS TO {reset_roughness}'))
        new_roughness = torch.min(self._roughness, self.roughness_inverse_activation(torch.ones_like(self._roughness, ) * reset_roughness))
        self._roughness = self.replace_tensor_to_optimizer(new_roughness, '_roughness', optimizer, prefix)
        

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str, optimizer: Optimizer = None, prefix: str = ''):
        optimizable_tensor = None
        for group in optimizer.param_groups:
            if group["name"].replace(prefix, '') == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensor = group["params"][0]
        if optimizable_tensor is not None:
            return optimizable_tensor
        else:
            log(yellow_slim(f'{name} not found in optimizer'))
            return tensor

    def _prune_optimizer(self, mask: torch.Tensor, optimizer: Optimizer, prefix: str = ''):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            attr = getattr(self, group["name"].replace(prefix, ''), None)
            if attr is None: continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        if len(optimizable_tensors) == 0:
            log(yellow_slim(f'No optimizable tensors found in optimizer'))
            breakpoint()
        
        return optimizable_tensors

    def prune_points(self, mask, optimizer: Optimizer, prefix: str = ''):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer, prefix)
        for name, new_params in optimizable_tensors.items():
            setattr(self, name.replace(prefix, ''), new_params)

    def cat_tensors_to_optimizer(self, tensors_dict: dotdict, optimizer: Optimizer, prefix: str = ''):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict.get(group["name"].replace(prefix, ''), None)
            if extension_tensor is None: continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        if len(optimizable_tensors) == 0:
            log(yellow_slim(f'No optimizable tensors found in optimizer'))
            breakpoint()

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        optimizer: Optimizer,
        prefix: str,
        new_specular: torch.Tensor = None,
        new_roughness: torch.Tensor = None,
        new_transmission_coeff: torch.Tensor = None,
        new_ior: torch.Tensor = None,
    ):
        d = dotdict({
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacities,
            "_scaling": new_scaling,
            "_rotation": new_rotation,
        })

        d["_specular"] = new_specular
        d["_roughness"] = new_roughness
        d["_transmission_coeff"] = new_transmission_coeff
        d["_ior"] = new_ior

        optimizable_tensors = self.cat_tensors_to_optimizer(d, optimizer, prefix)
        for name, new_params in optimizable_tensors.items():
            setattr(self, name.replace(prefix, ''), new_params)

    def get_xyz_gradient_avg(self):
        avg = self.xyz_gradient_accum / self.denom
        avg[avg.isnan()] = 0.0
        return avg

    def get_xyz_weight_avg(self):
        avg = self.xyz_weight_accum / self.denom
        avg[avg.isnan()] = 0.0
        return avg

    def reset_stats(self):
        device = self.get_xyz.device
        self.xyz_gradient_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.denom.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.max_radii2D.set_(torch.zeros((self.get_xyz.shape[0]), device=device))
        self.xyz_weight_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))

    def prune_stats(self, mask):
        valid_points_mask = ~mask
        self.xyz_gradient_accum.set_(self.xyz_gradient_accum[valid_points_mask])
        self.denom.set_(self.denom[valid_points_mask])
        self.max_radii2D.set_(self.max_radii2D[valid_points_mask])
        self.xyz_weight_accum.set_(self.xyz_weight_accum[valid_points_mask])
        assert self.xyz_gradient_accum.shape[0] == self.get_xyz.shape[0]
        assert self.denom.shape[0] == self.get_xyz.shape[0]
        assert self.max_radii2D.shape[0] == self.get_xyz.shape[0]
        assert self.xyz_weight_accum.shape[0] == self.get_xyz.shape[0]

    def densify_stats(self, selected_pts_mask: torch.Tensor, split: int = 1, ratio: float = 1.0):
        new_xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, self.xyz_gradient_accum[selected_pts_mask].repeat(split, 1) * ratio], dim=0)
        new_denom = torch.cat([self.denom, self.denom[selected_pts_mask].repeat(split, 1)], dim=0)
        new_max_radii2D = torch.cat([self.max_radii2D, self.max_radii2D[selected_pts_mask].repeat(split) * ratio], dim=0)
        new_xyz_weight_accum = torch.cat([self.xyz_weight_accum, self.xyz_weight_accum[selected_pts_mask].repeat(split, 1) * self.xyz_weight_accum.max()], dim=0)
        self.xyz_gradient_accum.set_(new_xyz_gradient_accum)
        self.denom.set_(new_denom)
        self.max_radii2D.set_(new_max_radii2D)
        self.xyz_weight_accum.set_(new_xyz_weight_accum)
        assert self.xyz_gradient_accum.shape[0] == self.get_xyz.shape[0]
        assert self.denom.shape[0] == self.get_xyz.shape[0]
        assert self.max_radii2D.shape[0] == self.get_xyz.shape[0]
        assert self.xyz_weight_accum.shape[0] == self.get_xyz.shape[0]

    def create_stats(self, n_extra: int, ratio: float = 1.0):
        new_xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((n_extra, 1), device=self.device)], dim=0)
        new_denom = torch.cat([self.denom, torch.zeros((n_extra, 1), device=self.device)], dim=0)
        new_max_radii2D = torch.cat([self.max_radii2D, torch.zeros((n_extra,), device=self.device)], dim=0)
        new_xyz_weight_accum = torch.cat([self.xyz_weight_accum, torch.ones((n_extra, 1), device=self.device) * self.xyz_weight_accum.max()], dim=0)
        self.xyz_gradient_accum.set_(new_xyz_gradient_accum)
        self.denom.set_(new_denom)
        self.max_radii2D.set_(new_max_radii2D)
        self.xyz_weight_accum.set_(new_xyz_weight_accum)
        assert self.xyz_gradient_accum.shape[0] == self.get_xyz.shape[0]
        assert self.denom.shape[0] == self.get_xyz.shape[0]
        assert self.max_radii2D.shape[0] == self.get_xyz.shape[0]
        assert self.xyz_weight_accum.shape[0] == self.get_xyz.shape[0]

    def clone(self, mask, optimizer: Optimizer = None, prefix: str = ''):
        # Should we just copy? Or should we add some noise to the new points? # NOTE: add noise here
        new_xyz = self._xyz[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacities = self._opacity[mask]
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        new_specular = self._specular[mask]
        new_roughness = self._roughness[mask]
        new_transmission_coeff = self._transmission_coeff[mask]
        new_ior = self._ior[mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer, prefix, new_specular, new_roughness, new_transmission_coeff, new_ior)
        self.densify_stats(mask, 1, 1.0)

    def split(self, mask, N: int = 2, optimizer: Optimizer = None, prefix: str = '', ratio: float = 0.8):
        # Split xyz
        stds = self.get_scaling[mask].repeat(N, 1)  # (M * N, 3), NOTE: 3DGS has 3 scaling parameters
        means = torch.zeros_like(stds)  # (M * N, 3)
        # Only split along the longest axis to avoid floaters and make the optimization more stable
        samples = torch.normal(means, stds).to(device=self.device, dtype=self.get_xyz.dtype)  # (M * N, 3)
        rots = build_rotation(self._rotation[mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[mask].repeat(N, 1) / (ratio * N))  # NOTE: 3DGS has 3 scaling parameters
        # Split features
        new_rotation = self._rotation[mask].repeat(N, 1)
        new_features_dc = self._features_dc[mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[mask].repeat(N, 1, 1)
        new_opacity = self._opacity[mask].repeat(N, 1)
        new_specular = self._specular[mask].repeat(N, 1)
        new_roughness = self._roughness[mask].repeat(N, 1)
        new_transmission_coeff = self._transmission_coeff[mask].repeat(N, 1)
        new_ior = self._ior[mask].repeat(N, 1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, optimizer, prefix, new_specular, new_roughness, new_transmission_coeff, new_ior)
        self.densify_stats(mask, N, 1.0 / (ratio * N))
        # Prune splited points
        n_split = mask.sum().item()
        prune_mask = torch.cat((mask, torch.zeros((n_split * N,), device=self.device, dtype=bool)))
        self.prune_points(prune_mask, optimizer, prefix)
        self.prune_stats(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_clone(
        self,
        densify_grad_threshold: float,
        densify_size_threshold: float,
        optimizer: Optimizer = None,
        prefix: str = ''
    ):
        # Extract points that satisfy the gradient condition
        grads = self.get_xyz_gradient_avg()
        high_grads = (grads >= densify_grad_threshold).squeeze(-1)
        # Extract points that satisfy the size conditions
        selected_pts_mask = torch.max(self.get_scaling, dim=1).values <= densify_size_threshold * self.spatial_scale
        selected_pts_mask = torch.logical_and(selected_pts_mask, high_grads)
        n_clone = selected_pts_mask.sum().item()
        log(yellow_slim(f'[CLONE] num points clone: {n_clone}.'))
        # Actual cloning
        if n_clone > 0: self.clone(selected_pts_mask, optimizer, prefix)

    def densify_and_split(
        self,
        densify_grad_threshold: float,
        densify_size_threshold: float,
        split_screen_threshold: float = None,
        optimizer: Optimizer = None,
        prefix: str = '',
        N: int = 2
    ):
        # Extract points that satisfy the gradient condition
        grads = self.get_xyz_gradient_avg()
        high_grads = (grads >= densify_grad_threshold).squeeze(-1)
        # Extract points that satisfy the size conditions
        selected_pts_mask = torch.max(self.get_scaling, dim=1).values > densify_size_threshold * self.spatial_scale
        # Extract points that satisfy the screen condition
        if split_screen_threshold is not None:
            selected_pts_mask = torch.logical_or(selected_pts_mask, self.max_radii2D > split_screen_threshold)
        selected_pts_mask = torch.logical_and(selected_pts_mask, high_grads)
        n_split = selected_pts_mask.sum().item()
        log(yellow_slim(f'[SPLIT] num points split: {n_split}, num split: {N}.'))
        # Actual splitting
        if n_split > 0: self.split(selected_pts_mask, N, optimizer, prefix)

    def prune_min_opacity_and_gradients(
        self,
        min_opacity: float = None,
        min_gradient: float = None,
        optimizer: Optimizer = None,
        prefix: str = ''
    ):
        n_before = self.get_xyz.shape[0]

        if min_opacity is not None:
            min_occs = (self.get_opacity < min_opacity).squeeze(-1)
            n_min_occ = min_occs.sum().item()
        else:
            min_occs = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_min_occ = 0
        if min_gradient is not None:
            grads = self.get_xyz_gradient_avg()
            min_grads = ((grads <= min_gradient) & (self.denom != 0)).squeeze(-1)
            n_min_grad = min_grads.sum().item()
        else:
            min_grads = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_min_grad = 0

        prune_mask = torch.logical_or(min_occs, min_grads)
        if prune_mask.sum().item() > 0:
            self.prune_points(prune_mask, optimizer, prefix)
            self.prune_stats(prune_mask)
            torch.cuda.empty_cache()

        n_after = self.get_xyz.shape[0]
        log(yellow_slim(f'[PRUNE OCC AND GRAD] ' +
                        f'num points pruned: {n_before - n_after} ' +
                        f'num points min opacity: {n_min_occ} ' +
                        f'num points min gradient: {n_min_grad}.'))

    def prune_max_scene_and_screen(
        self,
        max_scene_threshold: float = None,
        max_screen_threshold: float = None,
        min_weight_threshold: float = None,
        optimizer: Optimizer = None,
        prefix: str = ''
    ):
        n_before = self.get_xyz.shape[0]

        if max_screen_threshold is not None:
            max_screens = self.max_radii2D > max_screen_threshold
            n_max_screen = max_screens.sum().item()
        else:
            max_screens = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_max_screen = 0
        if max_scene_threshold is not None:
            max_scenes = torch.max(self.get_scaling, dim=-1).values > self.spatial_scale * max_scene_threshold
            n_max_scene = max_scenes.sum().item()
        else:
            max_scenes = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_max_scene = 0
        if min_weight_threshold is not None:
            # Accumulated weight related prune/split mask
            weights = self.get_xyz_weight_avg()
            min_weights = (weights < torch.quantile(weights, min_weight_threshold)).squeeze(-1)
            n_min_weight = min_weights.sum().item()
        else:
            min_weights = torch.ones((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_min_weight = 0

        # Get the prune and split mask respectively
        prune_mask = torch.logical_or(max_screens, max_scenes)
        split_mask = torch.logical_and(prune_mask, ~min_weights)
        prune_mask = torch.logical_and(prune_mask, min_weights)
        split_mask = split_mask[~prune_mask]
        n_prune = prune_mask.sum().item()
        n_split = split_mask.sum().item()

        # Actual pruning
        if n_prune > 0:
            self.prune_points(prune_mask, optimizer, prefix)
            self.prune_stats(prune_mask)
            torch.cuda.empty_cache()
        # Actual splitting
        if n_split > 0:
            self.split(split_mask, 5, optimizer, prefix, 0.5)

        n_after = self.get_xyz.shape[0]
        log(yellow_slim(f'[PRUNE SCREEN AND SCENE] ' +
                        f'num points pruned: {n_prune} ' +
                        f'num points splitted: {n_split} ' +
                        f'num points max screen: {n_max_screen} ' +
                        f'num points max scene: {n_max_scene}.'))

    def prune_visibility(self, optimizer: Optimizer = None, prefix: str = ''):
        n_before = self.get_xyz.shape[0]
        n_after = int(self.max_gs * self.max_gs_threshold)
        n_prune = n_before - n_after

        if n_prune > 0:
            weights = self.get_xyz_weight_avg()
            # Find the mask of top n_prune smallest `self.xyz_weight_accum`
            _, indices = torch.topk(weights[..., 0], n_prune, largest=False)
            prune_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            prune_mask[indices] = True
            # Prune points
            self.prune_points(prune_mask, optimizer, prefix)
            self.prune_stats(prune_mask)
            torch.cuda.empty_cache()

            log(yellow_slim(f'[PRUNE VISIBILITY] num points pruned: {n_prune}.'))

    def densify_and_prune(
        self,
        min_opacity,
        min_gradient,
        densify_grad_threshold,
        densify_size_threshold,
        split_screen_threshold=None,
        max_scene_threshold=None,
        max_screen_threshold=None,
        min_weight_threshold=None,
        prune_visibility=False,
        optimizer=None,
        prune_large_gs=False,
        prefix=''
    ):
        grads = self.get_xyz_gradient_avg()
        log(yellow_slim(f'[D&P] min grad: {grads.min().item()}, max grad: {grads.max().item()}.'))
        log(yellow_slim(f'[D&P] num points: {self.get_xyz.shape[0]}.'))
        log(yellow_slim(f'[D&P] min radii2D: {self.max_radii2D.min().item()}, max radii2D: {self.max_radii2D.max().item()}.'))
        log(yellow_slim(f'[D&P] min occ: {self.get_opacity.min().item()}, max occ: {self.get_opacity.max().item()}. ' +
                        f'min scaling: {self.get_scaling.min().item()}, max scaling: {self.get_scaling.max().item()}.'))
        # The order of the following functions is important
        # 1. first we prune points that are in min opacity and max screen
        # 2. then we densify points that are in high gradient
        # 3. then we prune points that are in max screen and max scene
        self.densify_and_clone(densify_grad_threshold, densify_size_threshold, optimizer, prefix)
        self.densify_and_split(densify_grad_threshold, densify_size_threshold, split_screen_threshold, optimizer, prefix)
        self.prune_min_opacity_and_gradients(min_opacity, min_gradient, optimizer, prefix)
        if prune_large_gs:
            self.prune_max_scene_and_screen(max_scene_threshold, max_screen_threshold, min_weight_threshold, optimizer, prefix)
        if prune_visibility:
            self.prune_visibility(optimizer, prefix)
        self.reset_stats()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weight_accumulate=None):
        self.denom[update_filter] += 1

        # Update the accumulated gradient for each gaussian
        xyz_gradient_norm = torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += xyz_gradient_norm

        # Maybe update the accumulated weight for each gaussian
        if weight_accumulate is not None: self.xyz_weight_accum[update_filter] += weight_accumulate[update_filter]

    def update_learning_rate(self, iter: float, optimizer: Optimizer, prefix: str = ''):
        for param_group in optimizer.param_groups:
            if self.xyz_scheduler is not None and param_group["name"] == f"{prefix}_xyz":
                param_group['lr'] = self.xyz_scheduler(iter)

    def update_learning_rate_by_name(self, name: str, lr: float, optimizer: Optimizer, prefix: str = ''):
        for param_group in optimizer.param_groups:
            if param_group["name"] == f"{prefix}{name}":
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):  # Should be 3 for 3DGS
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(
        self,
        path: str,
        bounds: torch.Tensor = None
    ):
        from plyfile import PlyData, PlyElement
        os.makedirs(dirname(path), exist_ok=True)

        # Only save the points within the bounds
        # `bounds` is a tuple of two 3D points, representing the min and max bounds
        if bounds is not None: mask = ((self._xyz >= bounds[0]) & (self._xyz <= bounds[1])).all(dim=-1)
        else: mask = torch.ones((self._xyz.shape[0],), dtype=torch.bool, device=self._xyz.device)

        xyz = self._xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[mask].detach().cpu().numpy()
        scales = self._scaling[mask].detach().cpu().numpy()
        rotation = self._rotation[mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        transmission_coeffs = np.asarray(plydata.elements[0]["transmission_coeff"])[..., np.newaxis]
        speculars = np.asarray(plydata.elements[0]["specular"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._transmission_coeff = nn.Parameter(torch.tensor(transmission_coeffs, dtype=torch.float, device="cuda").requires_grad_(True))
        self._specular = nn.Parameter(torch.tensor(speculars, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = make_buffer(torch.full((1,), self.max_sh_degree, dtype=torch.long))

    def replace_with(self, other: 'GaussianModel', optimizer: torch.optim.Optimizer, prefix: str = ''):
        """
        Replaces the current model's parameters with another model's parameters.
        This handles resizing and optimizer state updates for both parameters and stats.
        """
        with torch.no_grad():
            # 1. Prune all existing points and their stats from the current model (`self`)
            prune_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device=self.device)
            if prune_mask.any():
                self.prune_points(prune_mask, optimizer, prefix)
                self.prune_stats(prune_mask)

            # 2. Prepare a dictionary of new tensors from the `other` model.
            # We clone and detach to avoid any gradient linkage.
            new_xyz = other._xyz.clone().detach()
            new_features_dc = other._features_dc.clone().detach()
            new_features_rest = other._features_rest.clone().detach()
            new_opacities = other._opacity.clone().detach()
            new_scaling = other._scaling.clone().detach()
            new_rotation = other._rotation.clone().detach()
            new_specular = other._specular.clone().detach()
            new_roughness = other._roughness.clone().detach()
            new_transmission_coeff = other._transmission_coeff.clone().detach()
            new_ior = other._ior.clone().detach()

            # 3. Use densification_postfix to add the new parameters and update the optimizer.
            self.densification_postfix(
                new_xyz, new_features_dc, new_features_rest, new_opacities,
                new_scaling, new_rotation, optimizer, prefix,
                new_specular, new_roughness, new_transmission_coeff, new_ior
            )
            
            # 4. Re-initialize the stats tensors with the correct size for the new points.
            # This is the inlined logic from the `create_stats` function.
            n_new = new_xyz.shape[0]
            self.xyz_gradient_accum = make_buffer(torch.zeros((n_new, 1), device=self.device))
            self.denom = make_buffer(torch.zeros((n_new, 1), device=self.device))
            self.max_radii2D = make_buffer(torch.zeros(n_new, device=self.device))
            self.xyz_weight_accum = make_buffer(torch.zeros((n_new, 1), device=self.device))

        log(green_slim(f"Replaced '{prefix}' with {other.get_xyz.shape[0]} points from source."))
        

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe: dotdict,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor = None,
    return_plane: bool = True,
    return_depth_normal: bool = True,
    wo_depth_normal_detach: bool = False,
    mlp_color: torch.Tensor = None,
    hybrid: bool = True,
    transparency_threshold: float = 0.15,
    device: str = 'cuda'
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity
    transparencies = pc.get_transmission_coeff
    speculars = pc.get_specular

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        rendered_image, radii, out_observe, out_all_map, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        rendered_alpha = out_all_map[3:4, ]
        output = dotdict()
        return_dict =  {"render": rendered_image,
                        "rend_alpha": rendered_alpha,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        output.update(return_dict)
        return output

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 7)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
    input_all_map[:, 5] = transparencies.squeeze()
    input_all_map[:, 6] = speculars.squeeze()

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    rendered_transparency = out_all_map[5:6, ]
    rendered_specular = out_all_map[6:7, ]

    rendered_normal = (rendered_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)  # (3, H, W)


    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rend_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rend_dist": rendered_distance,
                    "rend_alpha": rendered_alpha,
                    "transparency": rendered_transparency,
                    "specular": rendered_specular
                    }
    
    output = dotdict()  # (3, H, W)
    output.update(return_dict)
    
    if return_depth_normal:
        if wo_depth_normal_detach:
            depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha)
        else:
            depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        output.update({"surf_normal": depth_normal})
        
    return output

def render_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           return_plane = True, transparencies_map = None, speculars_map = None, transparency_threshold=0.15, start_threshold = 0.0, end_threshold = 0.2, window_size = 0.03):
    """
    Render the scene. Background tensor (bg_color) must be on GPU!
    Return only the depth map. No RGB or normal.
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    raster_settings = FirstSurfaceGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug,
            transparency_threshold=transparency_threshold,
            start_threshold=start_threshold,
            end_threshold=end_threshold,
            window_size=window_size
        )

    rasterizer = FirstSurfaceGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        rendered_image, radii, out_observe, _, _, _, _ = rasterizer.forward_depth(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
            ) # type: ignore
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    out_transparency_depth = rasterizer.forward_depth(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp,
        transparencies_map = transparencies_map,
        speculars_map = speculars_map,
        ) # type: ignore

    out_transparency_depth = out_transparency_depth[0]
    return_dict =  {
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "out_transparency_depth": out_transparency_depth
                    }
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict


def dpt2xyz(
    camera,
    dpt: torch.Tensor,
    device: str = 'cuda'
):
    # Get the camera extrinsic matrix
    c2w = (camera.world_view_transform.T).inverse()
    # Get the camera intrinsic matrix
    W, H = camera.image_width, camera.image_height
    fx = W / (2 * math.tan(camera.FoVx / 2.))
    fy = H / (2 * math.tan(camera.FoVy / 2.))
    K = torch.tensor([
        [fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]
    ]).float().to(device, non_blocking=True)  # (3, 3)

    # Backproject the depth map to 3D points
    u, v = torch.meshgrid(
        torch.arange(W).float().to(device, non_blocking=True),
        torch.arange(H).float().to(device, non_blocking=True),
        indexing='xy'
    )  # (H, W), (H, W)
    xyz = torch.stack(
        [u, v, torch.ones_like(u)], dim=-1
    ).reshape(-1, 3)  # (H * W, 3)
    ray_d = xyz @ K.inverse().mT @ c2w[:3, :3].mT  # (H * W, 3)
    ray_o = c2w[:3, 3]  # (3,)
    xyz = dpt.reshape(-1, 1) * ray_d + ray_o  # (H * W, 3)
    return xyz


def dpt2norm(
    camera,
    dpt: torch.Tensor,
    device: str = 'cuda'
):
    # Convert the depth map to 3D points
    xyz = dpt2xyz(
        camera, dpt, device
    ).reshape(*dpt.shape[1:], 3)  # (H, W, 3)

    out = torch.zeros_like(xyz)  # (H, W, 3)
    # Compute the normal map from the depth map
    dx = torch.cat([xyz[2:, 1:-1] - xyz[:-2, 1:-1]], dim=0)  # (H-2, W-2, 3)
    dy = torch.cat([xyz[1:-1, 2:] - xyz[1:-1, :-2]], dim=1)  # (H-2, W-2, 3)
    norm = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # (H-2, W-2, 3)
    out[1:-1, 1:-1, :] = norm  # (H, W, 3)
    return out
