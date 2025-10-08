#!/usr/bin/env python3
import os
import sys
import yaml
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.configloading import load_config
from src.render.ct_geometry_projector import ConeBeam3DProjector
from src.render.sdf_utils import sdf_to_occupancy, occupancy_to_sdf_2d 
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to

alpha = 50.0  # Use same alpha as training (from train.py)


def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0,0] + m[1,1] + m[2,2] - 1)/2)
    x = (m[2,1] - m[1,2])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2] - m[2,0])**2 + (m[1,0] -m[0,1])**2)
    y = (m[0,2] - m[2,0])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    z = (m[1,0] - m[0,1])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    return (x,y,z), angle


def load_real_artery(artery_path):
    """Load a real 3D artery from dataset."""
    print(f"Loading real artery from: {artery_path}")
    
    # Load the numpy array
    artery_3d = np.load(artery_path)
    print(f"Original artery shape: {artery_3d.shape}")
    print(f"Original artery range: [{artery_3d.min():.3f}, {artery_3d.max():.3f}]")
    
    # Apply the same preprocessing as in trainer.py
    artery_3d = np.transpose(artery_3d, (1,2,0))[::,::-1,::-1]
    artery_3d = np.transpose(artery_3d, (2,1,0))[::-1,::,::].copy()
    
    print(f"Preprocessed artery shape: {artery_3d.shape}")
    print(f"Preprocessed artery range: [{artery_3d.min():.3f}, {artery_3d.max():.3f}]")
    
    return artery_3d


def save_training_pipeline_results(proj_one, proj_two, sdf_one, sdf_two):
    """Save results from the exact training pipeline."""
    print("Saving training pipeline results...")
    
    output_dir = "./test_sdf_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy for visualization
    proj_one_np = proj_one.squeeze().cpu().numpy()
    proj_two_np = proj_two.squeeze().cpu().numpy()
    sdf_one_np = sdf_one.cpu().numpy()
    sdf_two_np = sdf_two.cpu().numpy()
    
    # Create comparison figure showing TRAINING pipeline results
    plt.figure(figsize=(16, 8))
    
    # View 1: Occupancy projection
    plt.subplot(2, 4, 1)
    plt.imshow(proj_one_np, cmap='gray')
    plt.title('ODL Projection View 1\n(Training Pipeline)')
    plt.colorbar()
    
    # View 1: SDF (this is what training actually uses!)
    plt.subplot(2, 4, 2)
    plt.imshow(sdf_one_np, cmap='RdBu_r')
    plt.title('2D SDF View 1\n(Training Ground Truth)')
    plt.colorbar()
    
    # View 1: 2D occupancy from SDF
    plt.subplot(2, 4, 3)
    sdf_to_occ_1 = sdf_to_occupancy(sdf_one, alpha=alpha)
    plt.imshow(sdf_to_occ_1.cpu().numpy(), cmap='gray')
    plt.title('2D Occupancy from SDF View 1\n(SDF→Occupancy)')
    plt.colorbar()
    
    # View 1: Back-conversion test
    plt.subplot(2, 4, 4)
    sdf_to_occ_back = sdf_to_occupancy(sdf_one, alpha=alpha)
    proj_one_normalized = (proj_one_np - proj_one_np.min()) / (proj_one_np.max() - proj_one_np.min())
    diff = np.abs(proj_one_normalized - sdf_to_occ_back.cpu().numpy())
    plt.imshow(diff, cmap='hot')
    plt.title('Reconstruction Error View 1\n|Proj - SDF→Occ|')
    plt.colorbar()
    
    # View 2: Occupancy projection
    plt.subplot(2, 4, 5)
    plt.imshow(proj_two_np, cmap='gray')
    plt.title('ODL Projection View 2\n(Training Pipeline)')
    plt.colorbar()
    
    # View 2: SDF (this is what training actually uses!)
    plt.subplot(2, 4, 6)
    plt.imshow(sdf_two_np, cmap='RdBu_r')
    plt.title('2D SDF View 2\n(Training Ground Truth)')
    plt.colorbar()
    
    # View 2: 2D occupancy from SDF
    plt.subplot(2, 4, 7)
    sdf_to_occ_2 = sdf_to_occupancy(sdf_two, alpha=alpha)
    plt.imshow(sdf_to_occ_2.cpu().numpy(), cmap='gray')
    plt.title('2D Occupancy from SDF View 2\n(SDF→Occupancy)')
    plt.colorbar()
    
    # View 2: Back-conversion test
    plt.subplot(2, 4, 8)
    sdf_to_occ_back_2 = sdf_to_occupancy(sdf_two, alpha=alpha)
    proj_two_normalized = (proj_two_np - proj_two_np.min()) / (proj_two_np.max() - proj_two_np.min())
    diff_2 = np.abs(proj_two_normalized - sdf_to_occ_back_2.cpu().numpy())
    plt.imshow(diff_2, cmap='hot')
    plt.title('Reconstruction Error View 2\n|Proj - SDF→Occ|')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_pipeline_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {output_dir}")


def main(artery_path=None):
    """Simulate exact training pipeline: 3D occupancy → ODL projector → 2D SDF"""
    print("=" * 60)
    print("Training Pipeline Simulation Test")
    print("=" * 60)
    
    # Load config and setup projectors exactly like training
    cfg = load_config("./config/CCTA.yaml")
    configPath = cfg['exp']['dataconfig']
    with open(configPath, "r") as handle:
        data = yaml.safe_load(handle)
    
    # Setup projector parameters (exact copy from trainer.py)
    dso = data["DSO"]
    dde = data["DDE"]
    proj_size = np.array(data["nDetector"])
    proj_reso = np.array(data["dDetector"])
    image_size = np.array(data["nVoxel"])
    image_reso = np.array(data["dVoxel"])
    first_proj_angle = [-data["first_projection_angle"][1], data["first_projection_angle"][0]]
    second_proj_angle = [-data["second_projection_angle"][1], data["second_projection_angle"][0]]
    
    # Create first projector (exact copy from trainer.py)
    from_source_vec= (0,-dso[0],0)
    from_rot_vec = (-1,0,0)
    to_source_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_source_vec)
    to_rot_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
    to_source_vec = axis_rotation(to_rot_vec[0], angle=first_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])
    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
    ct_projector_first = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[0], dso[0])
    
    # Create second projector (exact copy from trainer.py)
    from_source_vec= (0,-dso[1],0)
    from_rot_vec = (-1,0,0)
    to_source_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_source_vec)
    to_rot_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
    to_source_vec = axis_rotation(to_rot_vec[0], angle=second_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])
    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
    ct_projector_second = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[1], dso[1])
    
    # Load and preprocess 3D artery exactly like training
    phantom = np.load(artery_path)
    phantom = np.transpose(phantom, (1,2,0))[::,::-1,::-1]
    phantom = np.transpose(phantom, (2,1,0))[::-1,::,::].copy()
    phantom_tensor = torch.tensor(phantom, dtype=torch.float32)[None, ...]
    
    print("Running EXACT training pipeline...")
    print("Step 1: 3D occupancy → ODL projections")
    
    # Generate projections exactly like training
    train_projs_one = ct_projector_first.forward_project(phantom_tensor)
    train_projs_two = ct_projector_second.forward_project(phantom_tensor)
    
    print("Step 2: 2D projections → 2D SDF (distance transform)")
    
    # Convert to 2D SDF exactly like training  
    proj_sdf_one = occupancy_to_sdf_2d(train_projs_one.squeeze(0).squeeze(0), voxel_size=proj_reso[0])
    proj_sdf_two = occupancy_to_sdf_2d(train_projs_two.squeeze(0).squeeze(0), voxel_size=proj_reso[1])
    
    print(f"2D Projection 1 range: [{train_projs_one.min():.3f}, {train_projs_one.max():.3f}]")
    print(f"2D Projection 2 range: [{train_projs_two.min():.3f}, {train_projs_two.max():.3f}]")
    print(f"2D SDF 1 range: [{proj_sdf_one.min():.3f}, {proj_sdf_one.max():.3f}]")
    print(f"2D SDF 2 range: [{proj_sdf_two.min():.3f}, {proj_sdf_two.max():.3f}]")
    
    # Save results showing the TRAINING pipeline results
    save_training_pipeline_results(train_projs_one, train_projs_two, proj_sdf_one, proj_sdf_two)
    
    print("=" * 60)
    print("Training pipeline simulation completed! ✓")
    print("Check ./test_sdf_results/ for results.")
    print("=" * 60)


if __name__ == "__main__":
    # You can specify the path to your real artery data here
    artery_path = "data\gt_volume_900.npy"  # Change this to your artery file path
    
    # Run tests
    main(artery_path)