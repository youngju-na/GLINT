import torch
import torch.nn.functional as F





@torch.no_grad()
def check_geometric_consistency_batch(batch: dotdict, output: dotdict) -> torch.Tensor:
    B, P, _ = output.pcd_dpt_map.shape
    H = W = int(np.sqrt(P))
    
    # Use view 0 as source and view 1 as target
    src_idx, tgt_idx = 0, 1

    # 1. Get camera parameters and geometry for the view pair
    K_src, K_tgt = batch.K[src_idx], batch.K[tgt_idx]
    T_src_c2w, T_tgt_c2w = batch.T[src_idx], batch.T[tgt_idx]
    T_tgt_w2c = torch.inverse(T_tgt_c2w)

    dpt_src = output.pcd_dpt_map[src_idx].view(H, W, 1)
    dpt_tgt = output.pcd_dpt_map[tgt_idx].view(H, W, 1)
    norm_src = output.norm_map[src_idx].view(H, W, 3)
    norm_tgt = output.norm_map[tgt_idx].view(H, W, 3)

    # 2. Unproject source view pixels to 3D world coordinates
    y, x = torch.meshgrid(torch.arange(H, device=dpt_src.device), torch.arange(W, device=dpt_src.device), indexing='ij')
    p_src_cam = torch.stack([(x - K_src[0, 2]) / K_src[0, 0], (y - K_src[1, 2]) / K_src[1, 1], torch.ones_like(x)], dim=-1)
    p_src_cam = p_src_cam * dpt_src

    p_world_h = torch.cat([p_src_cam, torch.ones_like(p_src_cam[..., :1])], dim=-1)
    p_world = (T_src_c2w @ p_world_h.view(-1, 4).T).T[..., :3].view(H, W, 3)

    # 3. Reproject 3D points to the target view
    p_tgt_cam_h = (T_tgt_w2c @ torch.cat([p_world, torch.ones_like(p_world[..., :1])], dim=-1).view(-1, 4).T).T
    
    warped_dpt = p_tgt_cam_h[..., 2].view(H, W, 1)
    p_tgt_uv = p_tgt_cam_h[..., :2] / p_tgt_cam_h[..., 2:3].clamp(min=1e-6)
    
    p_tgt_norm_uv = torch.stack([(p_tgt_uv[..., 0] / (W - 1)) * 2 - 1,
                                    (p_tgt_uv[..., 1] / (H - 1)) * 2 - 1], dim=-1)

    # 4. Consistency check
    sampled_dpt_tgt = F.grid_sample(dpt_tgt.permute(2,0,1).unsqueeze(0), p_tgt_norm_uv.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)
    sampled_norm_tgt = F.grid_sample(norm_tgt.permute(2,0,1).unsqueeze(0), p_tgt_norm_uv.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0).permute(1,2,0)

    depth_consistent = torch.abs(warped_dpt - sampled_dpt_tgt) < self.consistency_depth_thresh

    R_src_c2w = T_src_c2w[:3, :3]
    R_tgt_c2w = T_tgt_c2w[:3, :3]
    norm_src_world = (R_src_c2w @ norm_src.view(-1, 3).T).T.view(H, W, 3)
    norm_tgt_world = (R_tgt_c2w @ sampled_norm_tgt.view(-1, 3).T).T.view(H, W, 3)
    normal_consistent = (torch.sum(norm_src_world * norm_tgt_world, dim=-1, keepdim=True) > self.consistency_normal_thresh)

    in_bounds_mask = ((p_tgt_norm_uv[..., 0] >= -1) & (p_tgt_norm_uv[..., 0] <= 1) &
                        (p_tgt_norm_uv[..., 1] >= -1) & (p_tgt_norm_uv[..., 1] <= 1)).unsqueeze(-1)

    consistency_mask_single_view = (depth_consistent & normal_consistent & in_bounds_mask).view(1, P, 1)
    
    full_consistency_mask = torch.ones(B, P, 1, dtype=torch.bool, device=dpt_src.device)
    full_consistency_mask[src_idx] = consistency_mask_single_view

    return full_consistency_mask