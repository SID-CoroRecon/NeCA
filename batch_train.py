#!/usr/bin/env python3
"""
Batch processing script for training NeCA on multiple 3D models.
This script processes all models specified in the configuration file.
"""
import os
import yaml
import argparse
import torch
import numpy as np
from src.config.configloading import load_config
from train import BasicTrainer, print_memory_stats

def batch_process_models(config_path):
    """Process multiple models as specified in the configuration."""
    
    # Load base configuration
    cfg = load_config(config_path)
    
    # Get model numbers to process
    model_numbers = cfg["exp"].get("model_numbers", [1])
    input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
    output_recon_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
    
    print(f"Batch processing {len(model_numbers)} models: {model_numbers}")
    print(f"Input directory: {input_data_dir}")
    print(f"Output directory: {output_recon_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_recon_dir, exist_ok=True)
    
    successful_models = []
    failed_models = []
    
    # Get experiment parameters
    lrates = cfg["train"]["lrate"] if isinstance(cfg["train"]["lrate"], list) else [cfg["train"]["lrate"]]
    loss_weight_experiments = cfg["train"].get("loss_weight_experiments", [[1.0, 1.0, 0.1]])
    
    for i, model_id in enumerate(model_numbers):
        print(f"\n{'='*60}")
        print(f"Processing Model {model_id} ({i+1}/{len(model_numbers)})")
        print(f"{'='*60}")
        
        try:
            # Check if input file exists
            input_file = os.path.join(input_data_dir, f"{model_id}.npy")
            if not os.path.exists(input_file):
                print(f"WARNING: Input file not found: {input_file}")
                print(f"Skipping model {model_id}")
                failed_models.append(model_id)
                continue
            
            # Run experiments for each learning rate and loss weight combination
            for exp_idx, (lr, loss_weights) in enumerate([(lr, weights) for lr in lrates for weights in loss_weight_experiments]):
                proj_w, sdf_w, eik_w = loss_weights
                experiment_name = f"{model_id}_lr{lr}_proj{proj_w}_sdf{sdf_w}_eik{eik_w}"
                print(f"\n--- Experiment {exp_idx+1}: LR={lr}, Weights=[proj:{proj_w}, sdf:{sdf_w}, eik:{eik_w}] ---")
                
                # Update configuration for current experiment
                cfg["exp"]["current_model_id"] = experiment_name
                cfg["train"]["lrate"] = lr
                cfg["train"]["current_loss_weights"] = loss_weights
                
                # Initialize device and set seeds for reproducibility
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Set same seeds for all experiments (fair comparison)
                torch.manual_seed(42)
                np.random.seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)
                    torch.cuda.manual_seed_all(42)
                
                print(f"Starting training for experiment {experiment_name}...")
                print(f"Input file: {input_file}")
                print(f"Expected output: {os.path.join(output_recon_dir, f'recon_{experiment_name}.npy')}")
                
                # Create trainer and start training
                trainer = BasicTrainer.__new__(BasicTrainer)
                trainer.__init__(cfg, device)
                
                print("Initial memory stats:")
                print_memory_stats()
                
                # Start training
                trainer.start()
                
                print(f"✅ Successfully completed experiment {experiment_name}")
                
                # Clear memory between experiments
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            successful_models.append(model_id)
            
            # Memory is already cleared after each experiment
                
        except Exception as e:
            print(f"❌ Error processing model {model_id}: {str(e)}")
            failed_models.append(model_id)
            
            # Clear memory even on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(model_numbers)}")
    print(f"Successful: {len(successful_models)} - {successful_models}")
    print(f"Failed: {len(failed_models)} - {failed_models}")
    
    if successful_models:
        print(f"\n✅ Successfully generated reconstructions:")
        for model_id in successful_models:
            recon_file = os.path.join(output_recon_dir, f"recon_{model_id}.npy")
            network_file = os.path.join(output_recon_dir, f"network_{model_id}.pth")
            print(f"  Model {model_id}: {recon_file}")
            print(f"  Network {model_id}: {network_file}")
    
    if failed_models:
        print(f"\n❌ Failed models: {failed_models}")
        print("Please check the error messages above and ensure input files exist.")

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple 3D models with NeCA")
    parser.add_argument("--config", default="./config/CCTA.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--single", type=int, default=None,
                       help="Process only a single model with the specified ID")
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Process single model
        cfg = load_config(args.config)
        cfg["exp"]["model_numbers"] = [args.single]
        cfg["exp"]["current_model_id"] = args.single
        
        # Save temporary config
        temp_config_path = "./config/temp_single_model.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        batch_process_models(temp_config_path)
        
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    else:
        # Process all models in config
        batch_process_models(args.config)

if __name__ == "__main__":
    main()
