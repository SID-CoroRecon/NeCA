import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
import os
import argparse
import pandas as pd
import json
from datetime import datetime
from src.config.configloading import load_config


def compute_overlap_metric(
    label: np.ndarray,
    output: np.ndarray,
    d: float = 0,
    voxel_spacing: tuple = (0.37695312, 0.37695312, 0.5),
    threshold_label: float = 0.5,
    threshold_output: float = 0.5
):
    """
    Computes the Overlap Metric (Ot(d)) between a binary label and a predicted output.
    
    Args:
        label (numpy.ndarray): Ground truth binary array (2D or 3D).
        output (numpy.ndarray): Predicted binary array (same shape as label).
        d (float): Distance threshold in mm (0 for Dice, 1 or 2 for relaxed overlap).
        voxel_spacing (tuple): Physical voxel spacing in mm (x,y,z). Default matches CCTA data.
        threshold_label (float): Binarization threshold for label (default: 0.5).
        threshold_output (float): Binarization threshold for output (default: 0.5).
    
    Returns:
        float: Overlap score Ot(d) ∈ [0, 1].
    """
    # Binarize inputs
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    # If d=0, compute Dice score (Ot(0))
    if d == 0:
        intersection = np.sum(label_bin & output_bin)
        union = np.sum(label_bin) + np.sum(output_bin)
        return (2 * intersection) / union if union != 0 else 0.0
    
    # Convert mm distance to voxel units using the smallest spacing dimension
    d_voxels = d / min(voxel_spacing)
    
    # Get coordinates of foreground points in label and output
    label_points = np.argwhere(label_bin > 0)
    output_points = np.argwhere(output_bin > 0)
    
    # If either set is empty, return 0 (no overlap)
    if len(label_points) == 0 or len(output_points) == 0:
        return 0.0
    
    # Build KDTree for the output points (prediction)
    kdtree = KDTree(output_points)
    
    # Find TPR(d): Label points within distance d_voxels of any output point
    dist_label_to_output, _ = kdtree.query(label_points, distance_upper_bound=d_voxels)
    tpr_d = np.sum(dist_label_to_output <= d_voxels)
    fn_d = len(label_points) - tpr_d
    
    # Build KDTree for the label points (ground truth)
    kdtree_label = KDTree(label_points)
    
    # Find TPM(d): Output points within distance d_voxels of any label point
    dist_output_to_label, _ = kdtree_label.query(output_points, distance_upper_bound=d_voxels)
    tpm_d = np.sum(dist_output_to_label <= d_voxels)
    fp_d = len(output_points) - tpm_d
    
    # Compute Ot(d)
    ot_d = (tpm_d + tpr_d) / (tpm_d + tpr_d + fn_d + fp_d)
    return ot_d


def compute_chamfer_distance(
    label: np.ndarray,
    output: np.ndarray,
    voxel_spacing: tuple = (0.37695312, 0.37695312, 0.5),
    threshold_label: float = 0.5,
    threshold_output: float = 0.5,
    physical_units: bool = True
) -> float:
    """
    Computes the symmetric Chamfer Distance (L2) between two binary volumes.
    
    Args:
        label (np.ndarray): Ground truth binary array (2D/3D).
        output (np.ndarray): Predicted binary array (same shape as label).
        voxel_spacing (tuple): Physical voxel spacing in mm (x,y,z). Default matches CCTA data.
        threshold_label (float): Binarization threshold for label. Default: 0.5.
        threshold_output (float): Binarization threshold for output. Default: 0.5.
        physical_units (bool): If True, returns distance in mm. If False, in voxels.
    
    Returns:
        float: Chamfer Distance (L2) in mm or voxels.
    """

    # Binarize inputs
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    # Get coordinates of foreground points
    label_points = np.argwhere(label_bin > 0)
    output_points = np.argwhere(output_bin > 0)
    
    # Handle empty cases
    if len(label_points) == 0 or len(output_points) == 0:
        return np.inf  # No overlap
    
    # Scale coordinates to physical units if requested
    if physical_units:
        label_points = label_points * np.array(voxel_spacing)
        output_points = output_points * np.array(voxel_spacing)
    
    # Build KDTrees
    kdtree_label = KDTree(label_points)
    kdtree_output = KDTree(output_points)
    
    # Label → Output distances
    dist_label_to_output, _ = kdtree_output.query(label_points)
    term1 = np.mean(dist_label_to_output ** 2)
    
    # Output → Label distances
    dist_output_to_label, _ = kdtree_label.query(output_points)
    term2 = np.mean(dist_output_to_label ** 2)
    
    # Symmetric Chamfer Distance (ℓ2)
    chamfer_distance = (term1 + term2) ** 0.5
    return chamfer_distance


def compute_all_metrics(
    label: np.ndarray,
    output: np.ndarray,
    voxel_spacing: tuple = (0.8, 0.8, 0.8),
    threshold_label: float = 0.8,
    threshold_output: float = 0.8,
    apply_rotation: bool = True
) -> dict:
    """
    Compute all evaluation metrics for a pair of volumes.
    
    Args:
        label (np.ndarray): Ground truth volume
        output (np.ndarray): Predicted volume
        voxel_spacing (tuple): Physical voxel spacing in mm
        threshold_label (float): Binarization threshold for ground truth
        threshold_output (float): Binarization threshold for prediction
        apply_rotation (bool): Whether to apply rotation to output for alignment
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Apply rotation if needed (for alignment)
    if apply_rotation:
        output = ndimage.rotate(output, angle=-90, axes=(2, 1), reshape=True, order=0)
    
    # Compute all metrics
    metrics = {}
    
    try:
        metrics['dice'] = compute_overlap_metric(
            label, output, d=0.0, 
            threshold_output=threshold_output, 
            threshold_label=threshold_label, 
            voxel_spacing=voxel_spacing
        )
        
        metrics['ot_1mm'] = compute_overlap_metric(
            label, output, d=1.0, 
            threshold_output=threshold_output, 
            threshold_label=threshold_label, 
            voxel_spacing=voxel_spacing
        )
        
        metrics['ot_2mm'] = compute_overlap_metric(
            label, output, d=2.0, 
            threshold_output=threshold_output, 
            threshold_label=threshold_label, 
            voxel_spacing=voxel_spacing
        )
        
        metrics['chamfer_distance'] = compute_chamfer_distance(
            label, output, 
            physical_units=True, 
            voxel_spacing=voxel_spacing, 
            threshold_output=threshold_output, 
            threshold_label=threshold_label
        )
        # Add additional useful metrics
        label_bin = (label >= threshold_label).astype(np.uint8)
        output_bin = (output >= threshold_output).astype(np.uint8)
        
        metrics['label_volume'] = np.sum(label_bin) * np.prod(voxel_spacing)  # mm³
        metrics['output_volume'] = np.sum(output_bin) * np.prod(voxel_spacing)  # mm³
        metrics['volume_ratio'] = metrics['output_volume'] / metrics['label_volume'] if metrics['label_volume'] > 0 else np.inf
        
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        metrics = {
            'dice': np.nan,
            'ot_1mm': np.nan,
            'ot_2mm': np.nan,
            'chamfer_distance': np.nan,
            'label_volume': np.nan,
            'output_volume': np.nan,
            'volume_ratio': np.nan
        }
    
    return metrics


def evaluate_single_model(
    model_id: int,
    gt_volume_path: str,
    recon_volume_path: str,
    voxel_spacing: tuple = (0.8, 0.8, 0.8),
    threshold_label: float = 0.8,
    threshold_output: float = 0.8,
    apply_rotation: bool = True
) -> dict:
    """
    Evaluate a single model by comparing ground truth and reconstruction.
    
    Args:
        model_id (int): Model identifier
        gt_volume_path (str): Path to ground truth volume
        recon_volume_path (str): Path to reconstructed volume
        voxel_spacing (tuple): Physical voxel spacing in mm
        threshold_label (float): Binarization threshold for ground truth
        threshold_output (float): Binarization threshold for prediction
        apply_rotation (bool): Whether to apply rotation for alignment
    
    Returns:
        dict: Evaluation results for this model
    """
    
    print(f"Evaluating Model {model_id}")
    print(f"  GT: {gt_volume_path}")
    print(f"  Recon: {recon_volume_path}")
    
    try:
        # Load volumes
        if not os.path.exists(gt_volume_path):
            raise FileNotFoundError(f"Ground truth not found: {gt_volume_path}")
        if not os.path.exists(recon_volume_path):
            raise FileNotFoundError(f"Reconstruction not found: {recon_volume_path}")
        
        label = np.load(gt_volume_path).astype(np.float32)
        output = np.load(recon_volume_path).astype(np.float32)
        
        print(f"  GT shape: {label.shape}, Recon shape: {output.shape}")
        
        # Compute metrics
        metrics = compute_all_metrics(
            label, output, 
            voxel_spacing=voxel_spacing,
            threshold_label=threshold_label,
            threshold_output=threshold_output,
            apply_rotation=apply_rotation
        )
        
        # Add metadata
        result = {
            'model_id': model_id,
            'gt_path': gt_volume_path,
            'recon_path': recon_volume_path,
            'gt_shape': label.shape,
            'recon_shape': output.shape,
            'voxel_spacing': voxel_spacing,
            'threshold_label': threshold_label,
            'threshold_output': threshold_output,
            'evaluation_success': True,
            **metrics
        }
        
        print(f"  ✅ Success: Dice={metrics['dice']:.4f}, Ot(1mm)={metrics['ot_1mm']:.4f}")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        result = {
            'model_id': model_id,
            'gt_path': gt_volume_path,
            'recon_path': recon_volume_path,
            'evaluation_success': False,
            'error_message': str(e),
            'dice': np.nan,
            'ot_1mm': np.nan,
            'ot_2mm': np.nan,
            'chamfer_distance': np.nan,
            'label_volume': np.nan,
            'output_volume': np.nan,
            'volume_ratio': np.nan
        }
    
    return result


def batch_evaluate_models(
    config_path: str = "./config/CCTA.yaml",
    model_numbers: list = None,
    voxel_spacing: tuple = (0.8, 0.8, 0.8),
    threshold_label: float = 0.8,
    threshold_output: float = 0.8,
    apply_rotation: bool = True,
    output_dir: str = "./logs/evaluation/"
) -> pd.DataFrame:
    """
    Batch evaluate multiple models and save comprehensive results.
    
    Args:
        config_path (str): Path to configuration file
        model_numbers (list): List of model IDs to evaluate (None = use config)
        voxel_spacing (tuple): Physical voxel spacing in mm
        threshold_label (float): Binarization threshold for ground truth
        threshold_output (float): Binarization threshold for prediction
        apply_rotation (bool): Whether to apply rotation for alignment
        output_dir (str): Directory to save evaluation results
    
    Returns:
        pd.DataFrame: DataFrame containing all evaluation results
    """
    
    print("🔍 BATCH EVALUATION OF NECA MODELS")
    print("=" * 60)
    
    # Load configuration
    try:
        cfg = load_config(config_path)
        print(f"✅ Loaded configuration: {config_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return pd.DataFrame()
    
    # Get paths and model numbers
    input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
    output_recon_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
    
    if model_numbers is None:
        model_numbers = cfg["exp"].get("model_numbers", [1])
    
    # Get experiment parameters for generating filenames
    lrates = cfg["train"]["lrate"] if isinstance(cfg["train"]["lrate"], list) else [cfg["train"]["lrate"]]
    loss_weight_experiments = cfg["train"].get("loss_weight_experiments", [[1.0, 1.0, 0.1]])
    
    print(f"Input GT directory: {input_data_dir}")
    print(f"Reconstructions directory: {output_recon_dir}")
    print(f"Models to evaluate: {model_numbers}")
    print(f"Learning rates: {lrates}")
    print(f"Loss weight experiments: {loss_weight_experiments}")
    print(f"Evaluation parameters:")
    print(f"  Voxel spacing: {voxel_spacing} mm")
    print(f"  GT threshold: {threshold_label}")
    print(f"  Prediction threshold: {threshold_output}")
    print(f"  Apply rotation: {apply_rotation}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each model and experiment combination
    results = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for model_id in model_numbers:
        print(f"\n📊 Evaluating Model {model_id}")
        print("-" * 40)
        
        # Define ground truth path (same for all experiments)
        gt_path = os.path.join(input_data_dir, f"{model_id}.npy")
        
        # Evaluate each experiment combination
        for lr in lrates:
            for loss_weights in loss_weight_experiments:
                proj_w, sdf_w = loss_weights
                experiment_name = f"{model_id}_lr{lr}_proj{proj_w}_sdf{sdf_w}"
                # Updated path structure: output_recon_dir/model_id/experiment_name/recon_occupancy_experiment_name.npy
                recon_path = os.path.join(output_recon_dir, str(model_id), experiment_name, f"recon_occupancy_{experiment_name}.npy")
                
                print(f"  Experiment: {experiment_name}")
                
                # Evaluate single experiment
                result = evaluate_single_model(
                    model_id=experiment_name,  # Use experiment name as identifier
                    gt_volume_path=gt_path,
                    recon_volume_path=recon_path,
                    voxel_spacing=voxel_spacing,
                    threshold_label=threshold_label,
                    threshold_output=threshold_output,
                    apply_rotation=apply_rotation
                )
                
                # Add experiment metadata
                result['base_model_id'] = model_id
                result['learning_rate'] = lr
                result['projection_weight'] = proj_w
                result['sdf_weight'] = sdf_w
                result['experiment_name'] = experiment_name
                
                results.append(result)
                
                if result['evaluation_success']:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results in multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV for easy viewing/analysis
    csv_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Saved CSV results: {csv_path}")
    
    # JSON for programmatic access
    json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    df_results.to_json(json_path, orient='records', indent=2)
    print(f"💾 Saved JSON results: {json_path}")
    
    # Summary statistics - only for successful evaluations
    if successful_evaluations > 0:
        # Filter only successful evaluations for statistics
        successful_results = df_results[df_results['evaluation_success'] == True]
        
        summary_stats = {
            'timestamp': timestamp,
            'total_models': len(model_numbers),
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'parameters': {
                'voxel_spacing': voxel_spacing,
                'threshold_label': threshold_label,
                'threshold_output': threshold_output,
                'apply_rotation': apply_rotation
            },
            'statistics': {
                'dice_mean': float(successful_results['dice'].mean()),
                'dice_std': float(successful_results['dice'].std()),
                'ot_1mm_mean': float(successful_results['ot_1mm'].mean()),
                'ot_1mm_std': float(successful_results['ot_1mm'].std()),
                'ot_2mm_mean': float(successful_results['ot_2mm'].mean()),
                'ot_2mm_std': float(successful_results['ot_2mm'].std()),
                'chamfer_distance_mean': float(successful_results['chamfer_distance'].mean()),
                'chamfer_distance_std': float(successful_results['chamfer_distance'].std())
            }
        }
        
        summary_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"📈 Saved summary statistics: {summary_path}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total models: {len(model_numbers)}")
    print(f"Successful: {successful_evaluations}")
    print(f"Failed: {failed_evaluations}")
    
    if successful_evaluations > 0:
        # Filter only successful evaluations for statistics display
        successful_results = df_results[df_results['evaluation_success'] == True]
        
        print(f"\n📊 Performance Statistics (based on {successful_evaluations} successful evaluations):")
        print(f"Dice Score:       {successful_results['dice'].mean():.4f} ± {successful_results['dice'].std():.4f}")
        print(f"Ot(1mm):          {successful_results['ot_1mm'].mean():.4f} ± {successful_results['ot_1mm'].std():.4f}")
        print(f"Ot(2mm):          {successful_results['ot_2mm'].mean():.4f} ± {successful_results['ot_2mm'].std():.4f}")
        print(f"Chamfer Dist:     {successful_results['chamfer_distance'].mean():.4f} ± {successful_results['chamfer_distance'].std():.4f} mm")
    
    if failed_evaluations > 0:
        print(f"\n❌ Failed models: {df_results[~df_results['evaluation_success']]['model_id'].tolist()}")
    
    return df_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate NeCA reconstructions against ground truth")
    
    parser.add_argument("--config", default="./config/CCTA.yaml",
                       help="Path to configuration file")
    parser.add_argument("--models", nargs='+', type=int, default=None,
                       help="Specific model IDs to evaluate (default: use config)")
    parser.add_argument("--single", type=int, default=None,
                       help="Evaluate only a single model")
    parser.add_argument("--output-dir", default="./logs/evaluation/",
                       help="Directory to save evaluation results")
    parser.add_argument("--threshold-gt", type=float, default=0.8,
                       help="Binarization threshold for ground truth")
    parser.add_argument("--threshold-pred", type=float, default=0.8,
                       help="Binarization threshold for predictions")
    parser.add_argument("--no-rotation", action='store_true',
                       help="Skip rotation alignment")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8],
                       help="Voxel spacing in mm (x y z)")
    
    args = parser.parse_args()
    
    # Handle single model evaluation
    if args.single is not None:
        args.models = [args.single]
    
    # Run batch evaluation
    results_df = batch_evaluate_models(
        config_path=args.config,
        model_numbers=args.models,
        voxel_spacing=tuple(args.voxel_spacing),
        threshold_label=args.threshold_gt,
        threshold_output=args.threshold_pred,
        apply_rotation=not args.no_rotation,
        output_dir=args.output_dir
    )
    
    return results_df


if __name__ == "__main__":
    main()