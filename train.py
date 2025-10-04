import torch
import argparse
import os

from src.config.configloading import load_config
from src.render import run_network
from src.trainer import Trainer

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/CCTA.yaml",
                        help="configs file path")
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable memory efficient settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def print_memory_stats():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

class BasicTrainer(Trainer):
    def __init__(self, config=None, device_override=None):
        """
        Basic network trainer with memory optimizations.
        """
        # Use provided config or default global config
        train_cfg = config if config is not None else cfg
        train_device = device_override if device_override is not None else device
        
        super().__init__(train_cfg, train_device)
        print(f"[Start] exp: {train_cfg['exp']['expname']}, net: Basic network")
        print(f"Model ID: {train_cfg['exp'].get('current_model_id', 'default')}")

        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        
        # SDF-related parameters
        self.use_sdf = train_cfg.get("train", {}).get("use_sdf", True)
        self.sdf_alpha = train_cfg.get("train", {}).get("sdf_alpha", 50.0)
        
        # Loss weights from experiment configuration
        self.loss_weights = train_cfg.get("train", {}).get("current_loss_weights", [1.0, 1.0, 0.1])
        self.projection_weight, self.sdf_loss_weight, self.eikonal_weight = self.loss_weights
        
        print(f"SDF Mode: {self.use_sdf}, Alpha: {self.sdf_alpha}")
        print(f"Loss Weights - Projection: {self.projection_weight}, SDF: {self.sdf_loss_weight}, Eikonal: {self.eikonal_weight}")
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.net, 'gradient_checkpointing_enable'):
            self.net.gradient_checkpointing_enable()
            
        print("Initial memory stats:")
        print_memory_stats()

    def compute_loss(self, data, global_step, idx_epoch):
        loss = {"loss": 0.}

        projs = data.projs
        
        # Use autocast for mixed precision to save memory
        with torch.amp.autocast(enabled=self.use_mixed_precision, dtype=torch.float16, device_type='cuda'):
            # Process network in chunks to save memory - now outputs SDF
            sdf_pred = run_network(self.voxels, self.net, self.netchunk)
            train_output_sdf = sdf_pred.squeeze()[None, ...]

            # Convert SDF to occupancy for projection
            from src.render.sdf_utils import sdf_to_occupancy
            train_output_occupancy = sdf_to_occupancy(train_output_sdf, alpha=self.sdf_alpha)

            # Process projections sequentially to reduce peak memory usage
            train_projs_one = self.ct_projector_first.forward_project(train_output_occupancy)
            
            # Clear intermediate tensors to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            train_projs_two = self.ct_projector_second.forward_project(train_output_occupancy)
            
            # Concatenate projections
            train_projs = torch.cat((train_projs_one, train_projs_two), 1)
            
            # Main projection loss (occupancy-based)
            projection_loss = self.l2_loss(train_projs, projs.float())
            
            # Add 2D SDF loss if available and weight > 0
            sdf_2d_loss = torch.tensor(0.0, device=projs.device, requires_grad=True)
            if (self.sdf_loss_weight > 0 and 
                hasattr(data, 'sdf_projs') and data.sdf_projs is not None):
                from src.render.sdf_utils import sdf_3d_to_occupancy_to_sdf_2d
                # Use actual detector resolution from config
                detector_pixel_size = self.dataconfig["dDetector"][0]  # Use first detector resolution
                pred_sdf_2d, _ = sdf_3d_to_occupancy_to_sdf_2d(
                    train_output_sdf, self.ct_projector_first, self.ct_projector_second,
                    alpha=self.sdf_alpha, voxel_size_2d=detector_pixel_size
                )
                sdf_2d_loss = self.l2_loss(pred_sdf_2d, data.sdf_projs.float())
            
            # Add Eikonal regularization if weight > 0
            eikonal_loss = torch.tensor(0.0, device=projs.device, requires_grad=True)
            if self.eikonal_weight > 0:
                from src.render.sdf_utils import compute_eikonal_loss
                eikonal_loss = compute_eikonal_loss(sdf_pred, self.voxels, num_sample_points=4096)
            
            # Combine losses with weights
            total_loss = (self.projection_weight * projection_loss + 
                         self.sdf_loss_weight * sdf_2d_loss + 
                         self.eikonal_weight * eikonal_loss)
            
            loss["loss"] = total_loss
            loss["projection_loss"] = projection_loss
            loss["sdf_2d_loss"] = sdf_2d_loss
            loss["eikonal_loss"] = eikonal_loss

        return loss

if __name__ == "__main__":
    print("Setting up trainer...")
    trainer = BasicTrainer()
    print("Starting training...")
    trainer.start()
