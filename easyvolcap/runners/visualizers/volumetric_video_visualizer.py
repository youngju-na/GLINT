import torch
import numpy as np
from glob import glob
from torch import nn
from os.path import join, dirname, exists
from typing import List, Tuple, Union, Type
from multiprocessing.pool import ThreadPool

from easyvolcap.engine import cfg, args  # global
from easyvolcap.engine import VISUALIZERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import normalize, affine_padding, affine_inverse
from easyvolcap.utils.color_utils import colormap
from easyvolcap.utils.depth_utils import depth_curve_fn
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import save_image, generate_video, Visualization


@VISUALIZERS.register_module()
class VolumetricVideoVisualizer:  # this should act as a base class for other types of visualizations (need diff dataset)
    def __init__(self,
                 uncrop_output_images: bool = True,  # will try to find crop_h, crop_w...
                 store_alpha_channel: bool = True,  # store rendered acc in alpha channel
                 store_ground_truth: bool = True,  # store the ground truth rendered values
                 store_image_error: bool = True,  # render the error map (usually mse)
                 store_video_output: bool = False,  # whether to construct .mp4 from .png s
                 generate_video_using_cuda: bool = False,

                 vis_ext: str = '.png',  # faster saving, faster viewing, not good for evaluation (metrics)
                 result_dir: str = 'data/result',
                 img_pattern: str = f'{{type}}/frame{{frame:04d}}_camera{{camera:04d}}',  # the formatting of the output
                 save_tag: str = '',
                 types: List[str] = [
                     Visualization.RENDER.name,
                     Visualization.DEPTH.name,
                     Visualization.ALPHA.name,
                 ],

                 stream_delay: int = 2,  # after this number of pending copy, start synchronizing the stream and saving to disk
                 pool_limit: int = 10,  # maximum number of pending tasks in the thread pool, keep this small to avoid using too much resource
                 video_fps: int = 60,
                 verbose: bool = True,

                 dpt_curve: str = 'normalize',  # looks good
                 dpt_mult: float = 1.0,
                 dpt_cm: str = 'linear' if args.type != 'gui' else 'linear',  # looks good
                 
                 # PGSR-style TSDF Fusion parameters
                 tsdf_voxel_size: float = 0.01,
                 tsdf_sdf_trunc_multiplier: float = 4.0,  # sdf_trunc = tsdf_sdf_trunc_multiplier * voxel_size
                 tsdf_min_depth: float = 2.0,
                 tsdf_max_depth: float = 3.5,
                 tsdf_depth_scale: float = 1000.0,
                 use_depth_filter: bool = False,
                 depth_filter_angle: float = 80.0,

                 # PGSR-style post-processing
                 post_process_mesh: bool = True,
                 num_cluster: int = 2,
                 min_cluster_triangles: int = 50,
                 ):
        super().__init__()

        self.uncrop_output_images = uncrop_output_images
        self.store_alpha_channel = store_alpha_channel
        self.store_ground_truth = store_ground_truth
        self.store_video_output = store_video_output
        self.store_image_error = store_image_error
        self.generate_video_using_cuda = generate_video_using_cuda

        result_dir = join(result_dir, cfg.exp_name)  # MARK: global configuration # TODO: unify the global config, currently a hack for orbit.yaml here
        result_dir = join(result_dir, str(save_tag)) if save_tag != '' else result_dir  # could be a pure number
        
        self.vis_ext = vis_ext
        self.save_tag = save_tag
        self.result_dir = result_dir
        self.types = [Visualization[t] for t in types]  # types of visualization

        self.img_pattern = img_pattern + self.vis_ext
        self.img_gt_pattern = self.img_pattern.replace(self.vis_ext, f'_gt{self.vis_ext}')
        self.img_error_pattern = self.img_pattern.replace(self.vis_ext, f'_error{self.vis_ext}')

        self.thread_pools: List[ThreadPool] = []
        self.cuda_streams: List[torch.cuda.Stream] = []
        self.cpu_buffers: List[torch.Tensor] = []
        self.stream_delay = stream_delay
        self.pool_limit = pool_limit

        self.video_fps = video_fps
        self.verbose = verbose
        self.dpt_curve = dpt_curve
        self.dpt_mult = dpt_mult
        self.dpt_cm = dpt_cm
        
        # PGSR-style TSDF Fusion parameters
        self.tsdf_voxel_size = tsdf_voxel_size
        self.tsdf_sdf_trunc = tsdf_sdf_trunc_multiplier * tsdf_voxel_size
        self.tsdf_min_depth = tsdf_min_depth
        self.tsdf_max_depth = tsdf_max_depth
        self.tsdf_depth_scale = tsdf_depth_scale
        self.use_depth_filter = use_depth_filter
        self.depth_filter_angle = depth_filter_angle

        # PGSR-style post-processing
        self.post_process_mesh = post_process_mesh
        self.num_cluster = num_cluster
        self.min_cluster_triangles = min_cluster_triangles

        # TSDF Volume (initialized on first MESH visualization call)
        self.tsdf_volume = None

        if self.verbose:
            types = '{' + ','.join([t.name for t in self.types]) + '}'
            log(f'Visualization output: {blue(join(self.result_dir, dirname(self.img_pattern).format(type=types)))}')  # use yellow for output path

    def _post_process_mesh(self, mesh, num_cluster: int = 1, min_len: int = 50):
        """
        Post-process a mesh to filter out floaters and disconnected parts.
        Exactly follows PGSR's post_process_mesh implementation.

        Args:
            mesh: Open3D TriangleMesh
            num_cluster: Number of largest clusters to keep (PGSR default: 1)
            min_len: Minimum triangles per cluster (PGSR default: 50)
        """
        import copy
        import open3d as o3d

        log(f'Post-processing mesh to keep {num_cluster} clusters (PGSR style)')
        log(f'  Input: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles')

        mesh_0 = copy.deepcopy(mesh)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        if len(cluster_n_triangles) == 0:
            log(yellow('No clusters found in mesh, returning original'))
            return mesh

        # PGSR logic: keep top num_cluster largest clusters, but at least min_len triangles
        n_cluster = np.sort(cluster_n_triangles.copy())[-num_cluster] if len(cluster_n_triangles) >= num_cluster else 0
        n_cluster = max(n_cluster, min_len)  # filter meshes smaller than min_len (default 50)

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()

        log(f'  Output: {len(mesh_0.vertices)} vertices, {len(mesh_0.triangles)} triangles')
        return mesh_0

    def generate_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.RENDER):
        # Extract the renderable image from output and batch
        img: torch.Tensor = None
        img_gt: Union[torch.Tensor, None] = None
        img_error: Union[torch.Tensor, None] = None

        if type == Visualization.NORMAL:
            if 'norm_map' not in output: return None, None, None
            
            # save original normal map for evaluation
            norm_map = normalize(output.norm_map) # world coordinate
            norm_map = norm_map @ batch.R.mT  # range [-1, 1]
            
            norm_map_save = norm_map.detach().cpu().numpy()
            os.makedirs(join(self.result_dir, "geo_metric", "normals_pred"), exist_ok=True)
            np.save(join(self.result_dir, "geo_metric", "normals_pred", f'{int(batch.meta.camera_index[0]):04d}.npy'), norm_map_save) # local coordinate
            # also save camera rotation for evaluation
            cam_R = batch.R.detach().cpu().numpy() # 
            os.makedirs(join(self.result_dir, "geo_metric", "cams"), exist_ok=True)
            np.save(join(self.result_dir, "geo_metric", "cams", f'{int(batch.meta.camera_index[0]):04d}.npy'), cam_R)
            
            def norm_curve_fn(norm):
                norm = normalize(norm)
                norm = norm @ batch.R.mT
                norm[..., 1] *= -1
                norm[..., 2] *= -1
                norm = norm * 0.5 + 0.5
                norm = norm * output.acc_map  # norm is different when blending
                return norm

            img = norm_curve_fn(output.norm_map)
            if cfg.model_cfg.supervisor_cfg.use_normal_type == 'diffren' and 'normal' in batch:
                img_gt = norm_curve_fn(batch.normal)
            elif cfg.model_cfg.supervisor_cfg.use_normal_type == 'stable' and 'norm' in batch:
                img_gt = norm_curve_fn(batch.norm)
            
            if self.store_ground_truth and ('normal' in batch or 'norm' in batch):
                norm = batch.normal if (cfg.model_cfg.supervisor_cfg.use_normal_type == 'diffren' and 'normal' in batch) else batch.get('norm', None)
                if norm is None: return img, img_gt, img_error
                norm = norm * 2 - 1
                norm[..., 1] *= -1
                norm[..., 2] *= -1
                norm = norm * 0.5 + 0.5
                img_gt = norm

        elif type == Visualization.SURFACE_NORMAL:
            if 'surf_norm_map' not in output: return None, None, None
            def norm_curve_fn(norm):
                norm = normalize(norm)
                norm = norm @ batch.R.mT
                norm[..., 1] *= -1
                norm[..., 2] *= -1
                norm = norm * 0.5 + 0.5
                norm = norm * output.acc_map  # norm is different when blending
                return norm

            img = norm_curve_fn(output.surf_norm_map)

        elif type == Visualization.DEPTH:
            if 'dpt_map' not in output: return None, None, None
            
            # save original depth map for evaluation
            depth_map = output.dpt_map.detach().cpu().numpy()
            os.makedirs(join(self.result_dir, "geo_metric", "depths_pred"), exist_ok=True)
            np.save(join(self.result_dir, "geo_metric", "depths_pred", f'{int(batch.meta.camera_index[0]):04d}.npy'), depth_map)
            
            if self.dpt_curve == 'linear':
                img = output.dpt_map
            else:
                img = depth_curve_fn(output.dpt_map, cm=self.dpt_cm)
            # img = (img - 0.5) * self.dpt_mult + 0.5

            img = img * self.dpt_mult
            if self.store_ground_truth and 'depth' in batch: #! dpt 
                if self.dpt_curve == 'linear':
                    img_gt = batch.depth #! dpt
                else:
                    img_gt = depth_curve_fn(batch.depth, cm=self.dpt_cm)
                # img_gt = (img_gt - 0.5) * self.dpt_mult + 0.5
                img_gt = img_gt * self.dpt_mult

        elif type == Visualization.FEATURE:
            if 'feat_map' not in output: return None, None, None
            # This visualizes the xyzt + xyz feature output
            def feat_curve_fn(feat: torch.Tensor):
                B, P, C = feat.shape
                N = C // 3  # ignore last few feature channels
                feat = torch.stack(feat[..., :3 * N].chunk(3, dim=-1), dim=-1).mean(dim=-2)  # now in rgb
                return feat
            img = feat_curve_fn(output.feat_map)
            # No gt for this

        elif type == Visualization.SURFACE:
            if 'surf_map' not in output: return None, None, None
            img = output.surf_map  # rgb, maybe add multiplier

        elif type == Visualization.DEFORM:
            if 'def_map' not in output: return None, None, None
            img = output.resd_map  # rgb, maybe add multiplier

        elif type == Visualization.ALPHA:
            if 'acc_map' not in output: return None, None, None
            img = output.acc_map.expand(output.acc_map.shape[:-1] + (3,))
            if self.store_ground_truth and 'msk' in batch:
                img_gt = batch.msk.expand(batch.msk.shape[:-1] + (3,))

        elif type == Visualization.FLOW:
            if 'flo_map' not in output: return None, None, None
            from torchvision.utils import flow_to_image
            img = output.flo_map
            B, P, C = output.flo_map.shape
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            img = output.flo_map.view(B, H, W, C).float()
            img = img.permute(0, 3, 1, 2)  # B, 2, H, W
            img = flow_to_image(img).view(B, -1, H * W).permute(0, 2, 1)  # B, H*W, 3
            img = img.float() / 255.0
            if 'flow_weight' in batch:
                img = img * batch.flow_weight.view(B, H * W, 1)
            if self.store_ground_truth and 'flow' in batch:
                img_gt = batch.flow.view(B, H, W, C).float()
                img_gt = img_gt.permute(0, 3, 1, 2)  # B, 2, H, W
                img_gt = flow_to_image(img_gt).view(B, -1, H * W).permute(0, 2, 1)  # B, H*W, 3
                img_gt = img_gt.float() / 255.0
                if 'flow_weight' in batch:
                    img_gt = img_gt * batch.flow_weight.view(B, H * W, 1)

        elif type == Visualization.IMAGE_LOSS_WEIGHT:
            if 'img_loss_weight' not in output: return None, None, None
            if 'img_loss_wet' in batch:
                img = batch.img_loss_wet
                img = img / img.max()
                img = img.expand(-1, -1, 3)

        elif type == Visualization.MESH:
            import open3d as o3d

            if 'dpt_map' not in output: return None, None, None

            # Initialize TSDF volume on first call
            if self.tsdf_volume is None:
                log(f'Initializing TSDF volume: voxel_size={self.tsdf_voxel_size}, depth=[{self.tsdf_min_depth}, {self.tsdf_max_depth}]')
                self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=self.tsdf_voxel_size,
                    sdf_trunc=self.tsdf_sdf_trunc,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )

            # Get image dimensions
            dpt = output.dpt_map
            if dpt.ndim == 3:  # B, N, C - flattened
                if 'meta' in batch and 'H' in batch.meta:
                    H, W = int(batch.meta.H[0].item()), int(batch.meta.W[0].item())
                else:
                    raise ValueError(f"Could not determine H, W for dpt_map of shape {dpt.shape}")
                dpt = dpt.view(dpt.shape[0], H, W, -1)
            B, H, W, C = dpt.shape

            # Camera intrinsics and pose
            K = batch.K[0]
            Fx, Fy = K[0, 0].item(), K[1, 1].item()
            Cx, Cy = K[0, 2].item(), K[1, 2].item()
            w2c = torch.cat([batch.R[0], batch.T[0]], dim=-1)
            pose = np.eye(4)
            pose[:3, :] = w2c.cpu().numpy()

            # Depth map with range filtering
            ref_depth = dpt[0, :, :, 0].clone()
            ref_depth[ref_depth < self.tsdf_min_depth] = 0
            ref_depth[ref_depth > self.tsdf_max_depth] = 0
            depth_np = np.ascontiguousarray((ref_depth.detach().cpu().numpy() * self.tsdf_depth_scale).astype(np.uint16))

            # RGB
            if 'rgb_map' in output:
                rgb = output.rgb_map
                if rgb.ndim == 3:
                    rgb = rgb.view(B, H, W, 3)
                rgb_np = np.ascontiguousarray((rgb[0].clamp(0, 1) * 255).cpu().numpy().astype(np.uint8))
            else:
                rgb_np = np.ascontiguousarray(np.ones((H, W, 3), dtype=np.uint8) * 180)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_np),
                o3d.geometry.Image(depth_np),
                depth_scale=self.tsdf_depth_scale,
                depth_trunc=self.tsdf_max_depth,
                convert_rgb_to_intensity=False
            )
            self.tsdf_volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, Fx, Fy, Cx, Cy),
                pose
            )

            return None, None, None

        # ... implement more
        elif type == Visualization.RENDER:
            if 'rgb_map' not in output: return None, None, None
            img = output.rgb_map
            if self.store_ground_truth and 'rgb' in batch:
                img_gt = batch.rgb
                
        elif type == Visualization.ENV_RENDER:
            if 'env_rgb_direct' not in output: return None, None, None
            img = output.env_rgb_direct
            if self.store_ground_truth and 'rgb' in batch:
                img_gt = batch.rgb
        
        elif type == Visualization.TRANS_ENV_RENDER:
            if 'trans_env_rgb_direct' not in output: return None, None, None
            img = output.trans_env_rgb_direct
            if self.store_ground_truth and 'rgb' in batch:
                img_gt = batch.rgb
            

        elif type == Visualization.SRCINPS:
            # src_inps, only for per-command visualization
            img = batch.src_inps.permute(0, 1, 3, 4, 2)
            img = torch.cat([img[:, i] for i in range(img.shape[1])], dim=-2)
            return img, None, None

        elif type == Visualization.SPECULAR:
            if 'spec_map' not in output: return None, None, None
            img = output.spec_map
            if img.shape[-1] != 3: img = img.expand(img.shape[:-1] + (3,))
            if self.store_ground_truth and 'spec' in batch:
                img_gt = batch.spec

        elif type == Visualization.ROUGHNESS:
            if 'rough_map' not in output: return None, None, None
            img = output.rough_map.expand(output.rough_map.shape[:-1] + (3,))
            if self.store_ground_truth and 'rough' in batch:
                img_gt = batch.rough.expand(output.rough.shape[:-1] + (3,))
                
        elif type == Visualization.TRANSPARENCY:
            if 'trans_map' not in output: return None, None, None
            img = output.trans_map.expand(output.trans_map.shape[:-1] + (3,))
            if self.store_ground_truth and 'confident_transparent_mask' in batch:
                img_gt = batch.confident_transparent_mask.expand(output.trans_map.shape[:-1] + (3,))

        # color visualizations
        elif type == Visualization.DIFFUSE:
            if 'dif_rgb_map' not in output: return None, None, None
            img = output.dif_rgb_map

        elif type == Visualization.REFLECTION:
            if 'ref_rgb_map' not in output: return None, None, None
            img = output.ref_rgb_map # * 2.0 # for better visibility
            
        elif type == Visualization.TRANS_REFLECTION:
            if 'secondary_ref_rgb_map' not in output: return None, None, None
            img = output.secondary_ref_rgb_map # * 2.0 # for better visibility
            
        elif type == Visualization.TRANSMISSION:
            if 'trans_rgb_map' not in output: return None, None, None
            img = output.trans_rgb_map
            
        elif type == Visualization.TRANS_DEPTH:
            if 'trans_dpt_map' not in output: return None, None, None
            if self.dpt_curve == 'linear':
                img = output.trans_dpt_map * (output.trans_map > 0.5).float()
            else:
                img = depth_curve_fn(output.trans_dpt_map * (output.trans_map > 0.5).float(), cm=self.dpt_cm)
            # img = (img - 0.5) * self.dpt_mult + 0.5
            img = img * self.dpt_mult
            
        elif type == Visualization.TRANS_NORMAL:
            if 'trans_norm_map' not in output: return None, None, None
            def norm_curve_fn(norm):
                norm = normalize(norm)
                norm = norm @ batch.R.mT
                norm[..., 1] *= -1
                norm[..., 2] *= -1
                norm = norm * 0.5 + 0.5
                norm = norm * output.acc_map  # norm is different when blending
                return norm
            img = norm_curve_fn(output.trans_norm_map)
            if cfg.model_cfg.supervisor_cfg.use_normal_type == 'diffren' and 'normal' in batch:
                img_gt = norm_curve_fn(batch.normal)
            elif cfg.model_cfg.supervisor_cfg.use_normal_type == 'stable' and 'norm' in batch:
                img_gt = norm_curve_fn(batch.norm)
        
        elif type == Visualization.OPAQUE:
            if 'confident_opaque_mask' not in batch: return None, None, None
            img = batch.confident_opaque_mask
        
        elif type == Visualization.TRANS_GUIDANCE_ANGLE:
            if 'trans_guidance_angle_weight' not in output: return None, None, None
            img = output.trans_guidance_angle_weight.expand(output.trans_guidance_angle_weight.shape[:-1] + (3,))
        
        elif type == Visualization.TRANS_GUIDANCE_DEPTH:
            if 'trans_guidance_depth_weight' not in output: return None, None, None
            img = output.trans_guidance_depth_weight.expand(output.trans_guidance_depth_weight.shape[:-1] + (3,))
        
        elif type == Visualization.TRANS_GUIDANCE_WEIGHT:
            if 'trans_guidance_weight' not in output: return None, None, None
            img = output.trans_guidance_weight.expand(output.trans_guidance_weight.shape[:-1] + (3,))
            
            

        else:
            raise NotImplementedError(f'Unimplemented visualization type: {type}')


        if img_gt is not None and 'bg_color' in output and 'msk' in batch:
            # assert img_gt.shape[:2] == output.bg_color.shape[:2] == batch.msk.shape[:2], \
            #     "Shapes of img_gt, output.bg_color, and batch.msk must match"
    
            img_gt = img_gt + output.bg_color * (1 - batch.msk)

        if self.store_image_error and img_gt is not None:
            img_error = 3. * (img - img_gt).pow(2).sum(dim=-1).clip(0, 1)[..., None].expand(img.shape)

        if self.store_alpha_channel:
            msk = output.acc_map
            img = torch.cat([img, msk], dim=-1)
            if img_gt is not None:
                msk_gt = batch.msk
                img_gt = torch.cat([img_gt, msk_gt], dim=-1)
            if img_error is not None:
                msk_gt = batch.msk
                img_error = torch.cat([img_error, (msk_gt + msk).clip(0, 1)], dim=-1)

        B, P, C = img.shape
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        img = img.view(B, H, W, C).float()
        if img_gt is not None: img_gt = img_gt.view(B, H, W, C).float()
        if img_error is not None: img_error = img_error.view(B, H, W, C).float()

        if self.uncrop_output_images:  # necessary for GUI applications
            if 'orig_h' in batch.meta:  # !: BATCH: Removed
                x, y, w, h = batch.meta.crop_x[0].item(), batch.meta.crop_y[0].item(), batch.meta.W[0].item(), batch.meta.H[0].item()
                H, W = batch.meta.orig_h[0].item(), batch.meta.orig_w[0].item()
                img_full = img.new_zeros(B, H, W, C)  # original size
                img_full[:, y:y + h, x:x + w, :] = img
                img = img_full
                if img_gt is not None:
                    img_gt_full = img_gt.new_zeros(B, H, W, C)  # original size
                    img_gt_full[:, y:y + h, x:x + w, :] = img_gt
                    img_gt = img_gt_full
                if img_error is not None:
                    img_error_full = img_error.new_zeros(B, H, W, C)  # original size
                    img_error_full[:, y:y + h, x:x + w, :] = img_error
                    img_error = img_error_full

        return img, img_gt, img_error

    def visualize_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.RENDER):
        # Return a dictionary of image paths and arrays
        imgs, img_gts, img_errors = self.generate_type(output, batch, type)  # can be batched
        if imgs is None: return dotdict()

        # Starting a new stream is a small overhead for the GPU, but it allows us to run the visualization in parallel
        dft_stream: torch.cuda.Stream = torch.cuda.current_stream()  # default stream
        vis_stream: torch.cuda.Stream = torch.cuda.Stream()
        vis_stream.wait_stream(dft_stream)
        torch.cuda.set_stream(vis_stream)

        # Prepare for recoreder and storing some stuff to disk
        image_stats = dotdict()
        camera_index: torch.Tensor = batch.meta.camera_index
        frame_index: torch.Tensor = batch.meta.frame_index
        img_paths = []
        img_arrays = []

        for i in range(len(imgs)):
            frame = frame_index[i].item()
            camera = camera_index[i].item()

            # For shared values # TODO: fix this hacky implementation
            self.camera = camera  # for generating video
            self.frame = frame  # for generating video
            img_path = self.img_pattern.format(type=type.name, camera=camera, frame=frame)
            img_gt_path = self.img_gt_pattern.format(type=type.name, camera=camera, frame=frame)
            img_error_path = self.img_error_pattern.format(type=type.name, camera=camera, frame=frame)

            # Images
            img = imgs[i]
            if img_gts is not None: img_gt = img_gts[i]
            if img_errors is not None: img_error = img_errors[i]

            # For recorder
            image_stats[img_path] = img
            if img_gts is not None: image_stats[img_gt_path] = img_gt
            if img_errors is not None: image_stats[img_error_path] = img_error

            # Saving images to disk
            img_paths.append(join(self.result_dir, img_path))
            img_arrays.append(img.detach().to('cpu', non_blocking=True))  # start moving
            if img_gts is not None:
                img_paths.append(join(self.result_dir, img_gt_path))
                img_arrays.append(img_gt.detach().to('cpu', non_blocking=True))  # start moving
            if img_errors is not None:
                img_paths.append(join(self.result_dir, img_error_path))
                img_arrays.append(img_error.detach().to('cpu', non_blocking=True))  # start moving

        self.cuda_streams.append(vis_stream)
        self.cpu_buffers.append((img_paths, img_arrays))
        self.limit_cuda_streams()
        self.limit_thread_pools()  # maybe clear some of the taskes in the thread pool

        dft_stream.wait_stream(vis_stream)  # wait for the copy in this stream to finish before any other cuda operations on the default stream begins
        torch.cuda.set_stream(dft_stream)  # restore the original state
        return image_stats  # it's OK to return this

    # We need an interface for constructing final output paths
    # Along with paths (keys) for tensorboard logging system
    # GT values are stored separatedly in another entry
    # Same for `error` values

    def visualize(self, output: dotdict, batch: dotdict):
        image_stats = dotdict()
        for type in self.types:
            image_stats.update(self.visualize_type(output, batch, type))
        return image_stats

    def limit_cuda_streams(self):
        stream_cnt = len(self.cuda_streams)
        if stream_cnt > self.stream_delay:
            excess_streams = self.cuda_streams[:stream_cnt - self.stream_delay]
            excess_buffers = self.cpu_buffers[:stream_cnt - self.stream_delay]
            for stream, buffer in zip(excess_streams, excess_buffers):
                stream.synchronize()  # wait for the copy in this stream to finish
                img_paths, img_arrays = buffer
                img_arrays = [im.numpy() for im in img_arrays]
                pool = parallel_execution(img_paths, img_arrays, action=save_image, async_return=True, num_workers=3)  # actual writing to disk (async)
                self.thread_pools.append(pool)
            self.cpu_buffers = self.cpu_buffers[stream_cnt - self.stream_delay:]
            self.cuda_streams = self.cuda_streams[stream_cnt - self.stream_delay:]

    def limit_thread_pools(self):
        pool_length = len(self.thread_pools)
        if pool_length > self.pool_limit:
            for pool in self.thread_pools[:pool_length - self.pool_limit]:
                pool.close()
                pool.join()
            self.thread_pools = self.thread_pools[pool_length - self.pool_limit:]

    def summarize(self):
        for stream, buffer in zip(self.cuda_streams, self.cpu_buffers):
            stream.synchronize()  # wait for the copy in this stream to finish
            img_paths, img_arrays = buffer
            img_arrays = [im.numpy() for im in img_arrays]
            pool = parallel_execution(img_paths, img_arrays, action=save_image, async_return=True, num_workers=3)  # actual writing to disk (async)
            self.thread_pools.append(pool)

        for pool in self.thread_pools:  # finish all pending taskes before generating videos
            pool.close()
            pool.join()
        self.thread_pools.clear()  # remove all pools for this evaluation

        if self.store_video_output:
            for type in self.types:
                result_dir = dirname(join(self.result_dir, self.img_pattern)).format(type=type.name, camera=self.camera, frame=self.frame)
                frame_paths = sorted(glob(join(result_dir, f'*{self.vis_ext}')))
                if not frame_paths:
                    log(yellow(f'Skipping video generation for {type.name}: no {self.vis_ext} frames found in {blue(result_dir)}'))
                    continue
                result_str = f'"{result_dir}/*{self.vis_ext}"'
                output_path = result_str[1:].split('*')[0][:-1] + '.mp4'
                if self.generate_video_using_cuda:
                    try:
                        generate_video(result_str, output_path, fps=self.video_fps)  # one video for one type?
                    except RuntimeError:
                        log(yellow('Error encountered during video composition, will retry without hardware encoding'))
                        generate_video(result_str, output_path, fps=self.video_fps, hwaccel='none', vcodec='libx265')  # one video for one type?
                else:
                    generate_video(result_str, output_path, fps=self.video_fps, hwaccel='none', vcodec='libx265')  # one video for one type?
                log(f'Video generated: {blue(output_path)}')
                # TODO: use timg/tiv to visaulize the video / image on disk to the commandline

        if self.verbose:
            types = '{' + ','.join([t.name for t in self.types]) + '}'
            log(yellow(f'Visualization output: {blue(join(self.result_dir, dirname(self.img_pattern).format(type=types)))}'))  # use yellow for output path
        
        # Mesh extraction from TSDF volume
        if self.tsdf_volume is not None:
            import open3d as o3d

            mesh_dir = join(self.result_dir, "mesh")
            os.makedirs(mesh_dir, exist_ok=True)

            log(green('Extracting triangle mesh from TSDF volume...'))
            mesh = self.tsdf_volume.extract_triangle_mesh()

            raw_mesh_path = join(mesh_dir, "tsdf_fusion.ply")
            o3d.io.write_triangle_mesh(raw_mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            log(f'Raw mesh saved to {blue(raw_mesh_path)} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)')

            if self.post_process_mesh:
                mesh = self._post_process_mesh(mesh, num_cluster=self.num_cluster, min_len=self.min_cluster_triangles)
                post_mesh_path = join(mesh_dir, "tsdf_fusion_post.ply")
                o3d.io.write_triangle_mesh(post_mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                log(f'Post-processed mesh saved to {blue(post_mesh_path)} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)')

            self.tsdf_volume = None

        return dotdict()
