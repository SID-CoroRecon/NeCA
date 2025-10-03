import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def sdf_to_occupancy(sdf, alpha=50.0):
    """
    Convert SDF to occupancy/density for volume rendering.
    
    Args:
        sdf: SDF values (negative inside vessel, positive outside)
        alpha: Sharpness parameter for sigmoid conversion
        
    Returns:
        occupancy: Values in [0,1] representing vessel occupancy
    """
    # Inside vessel (negative SDF) -> high occupancy (near 1)
    # Outside vessel (positive SDF) -> low occupancy (near 0)
    occupancy = torch.sigmoid(-alpha * sdf)
    return occupancy


def occupancy_to_sdf_2d(occupancy_2d, voxel_size=1.0):
    """
    Convert 2D occupancy/projection to 2D SDF using distance transform.
    
    Args:
        occupancy_2d: 2D projection tensor [batch, height, width] or [height, width]
        voxel_size: Physical size of pixels for distance calculation
        
    Returns:
        sdf_2d: 2D SDF tensor (negative inside, positive outside)
    """
    if occupancy_2d.dim() == 2:
        occupancy_2d = occupancy_2d.unsqueeze(0)  # Add batch dimension
    
    batch_size = occupancy_2d.shape[0]
    device = occupancy_2d.device
    
    sdf_2d_list = []
    
    for b in range(batch_size):
        # Convert to numpy for scipy operations
        occ_np = occupancy_2d[b].detach().cpu().numpy()
        
        # Threshold to create binary mask
        binary_mask = occ_np > 0.5
        
        # Distance transform for inside (negative distances)
        inside_dist = ndimage.distance_transform_edt(binary_mask) * voxel_size
        
        # Distance transform for outside (positive distances)  
        outside_dist = ndimage.distance_transform_edt(~binary_mask) * voxel_size
        
        # Combine: negative inside, positive outside
        sdf_2d = np.where(binary_mask, -inside_dist, outside_dist)
        
        # Convert back to tensor
        sdf_2d_tensor = torch.tensor(sdf_2d, dtype=torch.float32, device=device)
        sdf_2d_list.append(sdf_2d_tensor)
    
    result = torch.stack(sdf_2d_list, dim=0)
    
    # Remove batch dimension if input was 2D
    if result.shape[0] == 1:
        result = result.squeeze(0)
    
    return result


def compute_eikonal_loss(sdf_pred, coords, num_sample_points=1024):
    """
    Compute Eikonal regularization loss: ||∇SDF|| = 1
    
    Note: Simplified implementation that returns zero loss.
    Proper Eikonal loss requires network re-evaluation at sample points.
    
    Args:
        sdf_pred: Predicted SDF values at coords
        coords: 3D coordinates where SDF was evaluated  
        num_sample_points: Number of random points to sample for gradient computation
        
    Returns:
        eikonal_loss: Zero loss to avoid computational graph issues
    """
    # Return zero loss for now to avoid gradient computation issues
    return torch.tensor(0.0, device=sdf_pred.device)


def sdf_3d_to_occupancy_to_sdf_2d(sdf_3d, projector_first, projector_second, 
                                  alpha=50.0, voxel_size_2d=0.278):
    """
    Complete pipeline: 3D SDF -> occupancy -> 2D projections -> 2D SDF
    
    Args:
        sdf_3d: 3D SDF prediction
        projector_first: First view projector
        projector_second: Second view projector  
        alpha: SDF to occupancy conversion sharpness
        voxel_size_2d: Pixel size for 2D SDF computation (mm)
        
    Returns:
        sdf_2d_combined: Combined 2D SDF from both views [batch, 2, height, width]
        occupancy_projections: Intermediate occupancy projections for debugging
    """
    # Convert 3D SDF to occupancy
    occupancy_3d = sdf_to_occupancy(sdf_3d, alpha=alpha)
    
    # Project to 2D occupancy
    proj_occupancy_first = projector_first.forward_project(occupancy_3d)
    proj_occupancy_second = projector_second.forward_project(occupancy_3d)
    
    # Combine projections
    occupancy_projections = torch.cat((proj_occupancy_first, proj_occupancy_second), dim=1)
    
    # Convert each projection to 2D SDF
    batch_size = occupancy_projections.shape[0]
    num_views = occupancy_projections.shape[1]
    
    sdf_2d_list = []
    for b in range(batch_size):
        view_sdf_list = []
        for v in range(num_views):
            proj = occupancy_projections[b, v]  # [height, width]
            sdf_2d = occupancy_to_sdf_2d(proj, voxel_size=voxel_size_2d)
            view_sdf_list.append(sdf_2d)
        sdf_2d_batch = torch.stack(view_sdf_list, dim=0)  # [num_views, height, width]
        sdf_2d_list.append(sdf_2d_batch)
    
    sdf_2d_combined = torch.stack(sdf_2d_list, dim=0)  # [batch, num_views, height, width]
    
    return sdf_2d_combined, occupancy_projections