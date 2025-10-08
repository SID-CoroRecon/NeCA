#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.render.sdf_utils import sdf_to_occupancy, occupancy_to_sdf_2d 

alpha = 100.0  # Sharpness parameter for SDF to occupancy conversion

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


def occupancy_to_sdf_3d(occupancy_3d, voxel_size=1.0):
    """Convert 3D occupancy to 3D SDF using distance transform."""
    
    print("Converting 3D occupancy to 3D SDF...")
    
    # Threshold to create binary mask
    binary_mask = occupancy_3d > 0.5
    
    # Distance transform for inside (negative distances)
    inside_dist = ndimage.distance_transform_edt(binary_mask) * voxel_size
    
    # Distance transform for outside (positive distances)  
    outside_dist = ndimage.distance_transform_edt(~binary_mask) * voxel_size
    
    # Combine: negative inside, positive outside
    sdf_3d = np.where(binary_mask, -inside_dist, outside_dist)
    
    print(f"3D SDF range: [{sdf_3d.min():.3f}, {sdf_3d.max():.3f}]")
    
    return sdf_3d.astype(np.float32)


def test_sdf_to_occupancy_conversion(artery_3d=None):
    """Test SDF to occupancy conversion."""
    print("Testing SDF to occupancy conversion...")
    
    # Use real artery data - convert occupancy to SDF first
    print("Using real artery data...")
    sdf_3d = occupancy_to_sdf_3d(artery_3d, voxel_size=0.8)  # Use actual voxel size from config
    sdf_tensor = torch.tensor(sdf_3d, dtype=torch.float32)
    
    # Convert to occupancy
    occupancy = sdf_to_occupancy(sdf_tensor, alpha=alpha)
    
    # Check properties
    print(f"SDF range: [{sdf_tensor.min():.3f}, {sdf_tensor.max():.3f}]")
    print(f"Occupancy range: [{occupancy.min():.3f}, {occupancy.max():.3f}]")
    
    # Verify that negative SDF -> high occupancy, positive SDF -> low occupancy
    inside_mask = sdf_tensor < 0
    outside_mask = sdf_tensor > 0
    
    avg_inside_occupancy = occupancy[inside_mask].mean()
    avg_outside_occupancy = occupancy[outside_mask].mean()
    
    print(f"Average occupancy inside (SDF < 0): {avg_inside_occupancy:.3f}")
    print(f"Average occupancy outside (SDF > 0): {avg_outside_occupancy:.3f}")
    
    assert avg_inside_occupancy > avg_outside_occupancy, "Inside should have higher occupancy than outside"
    assert occupancy.min() >= 0 and occupancy.max() <= 1, "Occupancy should be in [0,1]"
    
    print("✓ SDF to occupancy conversion test passed!\n")
    return sdf_tensor, occupancy


def save_test_results(occupancy_3d):
    """Save test results for visual inspection - focus on 2D projections."""
    print("Saving test results...")
    
    output_dir = "./test_sdf_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 2D projections from the 3D model for comparison
    print("Generating 2D projections from 3D model...")
    
    # Create a simple orthogonal projection (max along z-axis to preserve vessel structure)
    if occupancy_3d.dim() == 3:
        # Use max projection instead of sum to preserve vessel structure
        proj_2d_occupancy = torch.max(occupancy_3d, dim=0)[0].cpu().numpy()
        # Apply a lower threshold to capture more vessel parts
        proj_2d_occupancy = np.where(proj_2d_occupancy > 0.1, proj_2d_occupancy, 0.0)
        # Normalize to [0,1] range
        proj_2d_occupancy = proj_2d_occupancy / proj_2d_occupancy.max() if proj_2d_occupancy.max() > 0 else proj_2d_occupancy
        
        # Convert to tensor and generate 2D SDF
        proj_2d_occupancy_tensor = torch.tensor(proj_2d_occupancy, dtype=torch.float32)
        proj_2d_sdf = occupancy_to_sdf_2d(proj_2d_occupancy_tensor, voxel_size=0.28)
        
        print(f"2D Projection occupancy range: [{proj_2d_occupancy.min():.3f}, {proj_2d_occupancy.max():.3f}]")
        print(f"2D Projection SDF range: [{proj_2d_sdf.min():.3f}, {proj_2d_sdf.max():.3f}]")
        
        # Create comparison figure
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(proj_2d_occupancy, cmap='gray')
        plt.title('2D Occupancy Projection\n(from 3D model)')
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.imshow(proj_2d_sdf.cpu().numpy(), cmap='RdBu_r')
        plt.title('2D SDF from Projection\n(from 3D model)')
        plt.colorbar()
        
        # Create overlay: occupancy in gray, SDF contours in red/blue
        plt.subplot(2, 2, 2)
        plt.imshow(proj_2d_occupancy, cmap='gray', alpha=0.7)
        plt.contour(proj_2d_sdf.cpu().numpy(), levels=[0], colors='red', linewidths=2)
        plt.contour(proj_2d_sdf.cpu().numpy(), levels=[-3, -2, -1], colors='blue', alpha=0.5)
        plt.contour(proj_2d_sdf.cpu().numpy(), levels=[1, 2, 3], colors='red', alpha=0.5)
        plt.title('Occupancy + SDF Contours\n(0-level in red)')
        
        plt.subplot(2, 2, 4)
        # Show difference between actual occupancy and SDF-derived occupancy
        sdf_to_occ_back = sdf_to_occupancy(proj_2d_sdf, alpha=alpha)
        diff = np.abs(proj_2d_occupancy - sdf_to_occ_back.cpu().numpy())
        plt.imshow(diff, cmap='hot')
        plt.title('Reconstruction Error\n|Orig - SDF→Occ|')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2d_projection_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 2D projection arrays to {output_dir}")
        
        # Print validation metrics
        print("\n=== 2D Projection Validation ===")
        inside_points = proj_2d_sdf < 0
        outside_points = proj_2d_sdf > 0
        avg_occ_inside = proj_2d_occupancy[inside_points.cpu().numpy()].mean() if inside_points.sum() > 0 else 0
        avg_occ_outside = proj_2d_occupancy[outside_points.cpu().numpy()].mean() if outside_points.sum() > 0 else 0
        print(f"Average occupancy inside vessels (SDF < 0): {avg_occ_inside:.3f}")
        print(f"Average occupancy outside vessels (SDF > 0): {avg_occ_outside:.3f}")
        print(f"Ratio (should be > 1): {avg_occ_inside / avg_occ_outside if avg_occ_outside > 0 else 'inf'}")
    
    print(f"Results saved to {output_dir}")


def main(artery_path=None):
    """Run all tests.
    
    Args:
        artery_path (str): Path to real 3D artery numpy file. If None, uses synthetic data.
    """
    print("=" * 60)
    print("SDF Conversion Functions Test Suite")
    print("=" * 60)
    
    # Load real artery data if path provided
    artery_3d = load_real_artery(artery_path)
    print(f"Successfully loaded real artery data!")    
    print("\n" + "=" * 60)
    
    # Test 1: SDF to occupancy conversion
    _, occupancy_3d = test_sdf_to_occupancy_conversion(artery_3d)
    
    # Save results for visual inspection
    save_test_results(occupancy_3d)
    
    print("=" * 60)
    print("All tests completed successfully! ✓")
    print("Check ./test_sdf_results/ for visual results.")
    print("=" * 60)


if __name__ == "__main__":
    # You can specify the path to your real artery data here
    artery_path = "data\gt_volume_900.npy"  # Change this to your artery file path
    
    # Run tests
    main(artery_path)