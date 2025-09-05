import torch
import numpy as np
import argparse
import gc
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
    def __init__(self):
        """
        Basic network trainer with memory optimizations.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.net, 'gradient_checkpointing_enable'):
            self.net.gradient_checkpointing_enable()
            
        print("Initial memory stats:")
        print_memory_stats()

    def compute_loss(self, data, global_step, idx_epoch):
        loss = {"loss": 0.}

        projs = data.projs
        
        # Use autocast for mixed precision to save memory
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision, dtype=torch.float16):
            # Process network in chunks to save memory
            image_pred = run_network(self.voxels, self.net, self.netchunk)
            train_output = image_pred.squeeze()[None, ...]

            # Process projections sequentially to reduce peak memory usage
            train_projs_one = self.ct_projector_first.forward_project(train_output)
            
            # Clear intermediate tensors to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            train_projs_two = self.ct_projector_second.forward_project(train_output)
            
            # Concatenate projections
            train_projs = torch.cat((train_projs_one, train_projs_two), 1)
            
            # Compute loss
            loss["loss"] = self.l2_loss(train_projs, projs.float())

        return loss

print("Setting up trainer...")
trainer = BasicTrainer()
print("Starting training...")
trainer.start()
