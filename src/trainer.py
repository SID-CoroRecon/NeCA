import os
import math
import yaml
import json
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt

from .network import get_network
from .encoder import get_encoder
from src.render import run_network
from .dataset import TIGREDataset as Dataset

from src.render.ct_geometry_projector import ConeBeam3DProjector
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to

def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0,0] + m[1,1] + m[2,2] - 1)/2)

    x = (m[2,1] - m[1,2])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2] - m[2,0])**2 + (m[1,0] -m[0,1])**2)
    y = (m[0,2] - m[2,0])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    z = (m[1,0] - m[0,1])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    axis=(x,y,z)

    return axis, angle

class Trainer:
    def __init__(self, cfg, device="cuda"):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        
        # Memory optimization settings
        self.use_mixed_precision = cfg.get("train", {}).get("mixed_precision", True)
        self.gradient_accumulation_steps = cfg.get("train", {}).get("gradient_accumulation_steps", 1)
        self.memory_efficient_eval = cfg.get("train", {}).get("memory_efficient_eval", True)
        
        # Initialize AMP scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_mixed_precision)
  
        # Log directory and output paths
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        
        # Setup output directories for batch processing
        self.output_recon_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
        self.current_model_id = cfg["exp"].get("current_model_id", 1)
        os.makedirs(self.evaldir, exist_ok=True)
        os.makedirs(self.output_recon_dir, exist_ok=True)

        #######################################
        # Load CT geometry configuration
        configPath = cfg['exp']['dataconfig']
        with open(configPath, "r") as handle:
            data = yaml.safe_load(handle)

        # Setup data paths from main config (CCTA.yaml) - much cleaner!
        input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
        
        # Extract original model ID from experiment name (format: {model_id}_lr{lr}_loss{type})
        original_model_id = str(self.current_model_id).split('_')[0]
        gt_volume_filename = f"{original_model_id}.npy"
        gt_volume_path = osp.join(input_data_dir, gt_volume_filename)
        
        print(f"Processing experiment {self.current_model_id}")
        print(f"Original model ID: {original_model_id}")
        print(f"GT volume path: {gt_volume_path}")

        # Check if ground truth volume exists
        if not os.path.exists(gt_volume_path):
            raise FileNotFoundError(f"Ground truth volume not found: {gt_volume_path}")

        # No need for datadir anymore - we generate projections from GT volume directly
        print("Generating projections from 3D ground truth volume...")

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        dsd = data["DSD"] # Distance Source Detector   mm   
        dso = data["DSO"] # Distance Source Origin      mm 
        dde = data["DDE"]

        # Detector parameters
        proj_size = np.array(data["nDetector"])  # number of pixels              (px)
        proj_reso = np.array(data["dDetector"]) 
        # Image parameters
        image_size = np.array(data["nVoxel"])  # number of voxels              (vx)
        image_reso = np.array(data["dVoxel"])  # size of each voxel            (mm)
   
        first_proj_angle = [-data["first_projection_angle"][1], data["first_projection_angle"][0]]
        second_proj_angle = [-data["second_projection_angle"][1], data["second_projection_angle"][0]]

        #############
        #### first_projection
        from_source_vec= (0,-dso[0],0)
        from_rot_vec = (-1,0,0)
        to_source_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_source_vec)
        to_rot_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
        to_source_vec = axis_rotation(to_rot_vec[0], angle=first_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

        rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
        proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)

        self.ct_projector_first = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[0], dso[0])
        # proj_first = ct_projector.forward_project(phantom.squeeze(4))  # [bs, x, y, z] -> [bs, n, h, w]

        ### second projection
        from_source_vec= (0,-dso[1],0)
        from_rot_vec = (-1,0,0)
        to_source_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_source_vec)
        to_rot_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
        to_source_vec = axis_rotation(to_rot_vec[0], angle=second_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

        rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
        proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)

        self.ct_projector_second = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[1], dso[1])
        # proj_second = ct_projector.forward_project(phantom.squeeze(4))  # [bs, x, y, z] -> [bs, n, h, w]
        
        #####
        # Load 3D ground truth volume and generate projections (simplified)
        phantom = np.load(gt_volume_path)
        phantom = np.transpose(phantom, (1,2,0))[::,::-1,::-1]
        phantom = np.transpose(phantom, (2,1,0))[::-1,::,::].copy()
        phantom = torch.tensor(phantom, dtype=torch.float32)[None, ...]

        train_projs_one = self.ct_projector_first.forward_project(phantom)
        train_projs_two = self.ct_projector_second.forward_project(phantom)

        data["projections"] = torch.cat((train_projs_one,train_projs_two), 1)
        print(f"Generated projections from 3D volume: {data['projections'].shape}")
        
        # Generate 2D SDF targets from projections for SDF loss
        from src.render.sdf_utils import occupancy_to_sdf_2d
        proj_sdf_one = occupancy_to_sdf_2d(train_projs_one.squeeze(0).squeeze(0), voxel_size=proj_reso[0])  # [512, 512]
        proj_sdf_two = occupancy_to_sdf_2d(train_projs_two.squeeze(0).squeeze(0), voxel_size=proj_reso[1])  # [512, 512]
        data["sdf_projections"] = torch.cat((proj_sdf_one[None, None, :], proj_sdf_two[None, None, :]), 1)  # [1, 2, 512, 512]
        print(f"Generated 2D SDF targets: {data['sdf_projections'].shape}")

        # Dataset
        self.dataconfig = data
        self.train_dset = Dataset(data, device)
        self.voxels = self.train_dset.voxels

        # Network
        network = get_network(cfg["network"]["net_type"])
        net_type = cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        self.grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            self.grad_vars += list(self.net_fine.parameters())
        cfg["network"]["net_type"] = net_type

        # Optimizer with memory-efficient settings
        weight_decay_val = cfg["train"].get("weight_decay", 1e-6)
        if isinstance(weight_decay_val, str):
            weight_decay_val = float(weight_decay_val)
            
        self.optimizer = torch.optim.AdamW(
            params=self.grad_vars, 
            lr=cfg["train"]["lrate"], 
            betas=(0.9, 0.999),
            weight_decay=weight_decay_val,  # Ensure proper type conversion
            eps=1e-8
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            try:
                ckpt = torch.load(self.ckptdir, map_location=device)  # Load to specific device
                self.epoch_start = ckpt["epoch"] + 1
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if "scaler" in ckpt and self.use_mixed_precision:
                    self.scaler.load_state_dict(ckpt["scaler"])
                self.global_step = self.epoch_start #* len(self.train_dloader)
                self.net.load_state_dict(ckpt["network"])
                if self.n_fine > 0 and "network_fine" in ckpt and ckpt["network_fine"] is not None:
                    self.net_fine.load_state_dict(ckpt["network_fine"])
                print(f"Successfully loaded checkpoint from epoch {ckpt['epoch']}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting training from scratch.")
                self.epoch_start = 0

        # Loss tracking for plotting
        self.training_losses = []

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))
    
    def save_loss_plot(self):
        """
        Save training loss plot for current model.
        """
        if len(self.training_losses) == 0:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, 'b-', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss - Model {self.current_model_id}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        loss_plot_path = osp.join(self.output_recon_dir, f"loss_plot_{self.current_model_id}.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved loss plot: {loss_plot_path}")

    def start(self):
        """
        Main loop with memory optimizations.
        """
        iter_per_epoch = 1 #len(self.train_dloader)
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs+1):
            
            # Clear cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Evaluate and save final results
            save_final = self.conf.get("log", {}).get("save_final_model", True)
            save_intermediate = self.conf.get("log", {}).get("save_intermediate", False)
            
            should_evaluate = False
            if idx_epoch == self.epochs and save_final:
                should_evaluate = True  # Always evaluate at final epoch if save_final is True
            elif save_intermediate and (idx_epoch % self.i_eval == 0) and self.i_eval > 0:
                should_evaluate = True  # Evaluate intermediate epochs if save_intermediate is True
                
            if should_evaluate:  
                self.net.eval()
                with torch.no_grad():
                    if self.memory_efficient_eval:
                        # Process voxels in smaller chunks during evaluation
                        eval_chunk_size = self.netchunk // 4  # Use smaller chunks for evaluation
                        image_pred = self.run_network_chunked(
                            self.voxels, 
                            self.net_fine if self.net_fine is not None else self.net, 
                            eval_chunk_size
                        )
                    else:
                        image_pred = run_network(self.voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
                    
                    image_pred = (image_pred.squeeze()).detach().cpu().numpy()
                    
                    # Save with custom naming
                    if idx_epoch == self.epochs and save_final:
                        # Final epoch: save with custom reconstruction name
                        recon_filename = f"recon_{self.current_model_id}.npy"
                        recon_path = osp.join(self.output_recon_dir, recon_filename)
                        np.save(recon_path, image_pred)
                        print(f"Saved final reconstruction: {recon_path}")
                        
                        # Save ground truth 3D model
                        gt_volume_filename = f"gt_volume_{self.current_model_id}.npy"
                        gt_volume_path = osp.join(self.output_recon_dir, gt_volume_filename)
                        input_data_dir = self.conf["exp"].get("input_data_dir", "./data/GT_volumes/")
                        original_model_id = str(self.current_model_id).split('_')[0]
                        original_gt_path = osp.join(input_data_dir, f"{original_model_id}.npy")
                        gt_volume = np.load(original_gt_path)
                        np.save(gt_volume_path, gt_volume)
                        print(f"Saved ground truth 3D model: {gt_volume_path}")
                        
                        # Save ground truth projections
                        gt_projs_filename = f"gt_projections_{self.current_model_id}.npy"
                        gt_projs_path = osp.join(self.output_recon_dir, gt_projs_filename)
                        gt_projs_data = self.train_dset.projs.detach().cpu().numpy()
                        np.save(gt_projs_path, gt_projs_data)
                        print(f"Saved ground truth projections: {gt_projs_path}")
                        
                        # Save ground truth projections as images
                        for i in range(gt_projs_data.shape[1]):
                            proj_img = gt_projs_data[0, i]  # Get projection i
                            # Normalize to 0-255 range
                            proj_img_norm = ((proj_img - proj_img.min()) / (proj_img.max() - proj_img.min()) * 255).astype(np.uint8)
                            gt_proj_img_path = osp.join(self.output_recon_dir, f"gt_projection_{self.current_model_id}_view_{i+1}.png")
                            plt.imsave(gt_proj_img_path, proj_img_norm, cmap='gray')
                            print(f"Saved ground truth projection image {i+1}: {gt_proj_img_path}")
                        
                        # Generate and save predicted projections from final reconstruction
                        # Note: image_pred is now SDF values, need to handle differently
                        sdf_pred_tensor = torch.tensor(image_pred, dtype=torch.float32, device=self.train_dset.projs.device)[None, ...]
                        
                        # Save 3D SDF prediction
                        sdf_3d_filename = f"sdf_3d_{self.current_model_id}.npy"
                        sdf_3d_path = osp.join(self.output_recon_dir, sdf_3d_filename)
                        np.save(sdf_3d_path, image_pred)
                        print(f"Saved 3D SDF prediction: {sdf_3d_path}")
                        
                        # Convert SDF to occupancy for projection
                        from src.render.sdf_utils import sdf_to_occupancy, sdf_3d_to_occupancy_to_sdf_2d
                        occupancy_pred = sdf_to_occupancy(sdf_pred_tensor, alpha=50.0)
                        
                        # Save 3D occupancy prediction (final output like before)
                        occupancy_3d_filename = f"recon_occupancy_{self.current_model_id}.npy"
                        occupancy_3d_path = osp.join(self.output_recon_dir, occupancy_3d_filename)
                        occupancy_3d_data = occupancy_pred.squeeze().detach().cpu().numpy()
                        np.save(occupancy_3d_path, occupancy_3d_data)
                        print(f"Saved 3D occupancy prediction: {occupancy_3d_path}")
                        
                        pred_projs_one = self.ct_projector_first.forward_project(occupancy_pred)
                        pred_projs_two = self.ct_projector_second.forward_project(occupancy_pred)
                        pred_projs = torch.cat((pred_projs_one, pred_projs_two), 1)
                        
                        pred_projs_filename = f"pred_projections_{self.current_model_id}.npy"
                        pred_projs_path = osp.join(self.output_recon_dir, pred_projs_filename)
                        pred_projs_data = pred_projs.detach().cpu().numpy()
                        np.save(pred_projs_path, pred_projs_data)
                        print(f"Saved predicted projections: {pred_projs_path}")
                        
                        # Generate and save 2D SDF predictions
                        detector_pixel_size = self.dataconfig["dDetector"][0]  # Use first detector resolution  
                        pred_sdf_2d, _ = sdf_3d_to_occupancy_to_sdf_2d(
                            sdf_pred_tensor, self.ct_projector_first, self.ct_projector_second,
                            alpha=50.0, voxel_size_2d=detector_pixel_size
                        )
                        
                        pred_sdf_2d_filename = f"sdf_2d_pred_{self.current_model_id}.npy"
                        pred_sdf_2d_path = osp.join(self.output_recon_dir, pred_sdf_2d_filename)
                        pred_sdf_2d_data = pred_sdf_2d.detach().cpu().numpy()
                        np.save(pred_sdf_2d_path, pred_sdf_2d_data)
                        print(f"Saved 2D SDF predictions: {pred_sdf_2d_path}")
                        
                        # Save predicted 2D SDF as images
                        for i in range(pred_sdf_2d_data.shape[1]):
                            sdf_img = pred_sdf_2d_data[0, i]  # Get SDF view i
                            # Use RdBu_r colormap for SDF (blue=negative/inside, red=positive/outside)
                            pred_sdf_img_path = osp.join(self.output_recon_dir, f"sdf_2d_pred_{self.current_model_id}_view_{i+1}.png")
                            plt.figure(figsize=(8, 8))
                            plt.imshow(sdf_img, cmap='RdBu_r', vmin=-5, vmax=5)  # Fixed range for better visualization
                            plt.colorbar(label='SDF Value (mm)')
                            plt.title(f'Predicted 2D SDF - View {i+1}')
                            plt.savefig(pred_sdf_img_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            print(f"Saved predicted 2D SDF image {i+1}: {pred_sdf_img_path}")
                        
                        # Save ground truth 2D SDF targets 
                        if "sdf_projections" in self.dataconfig:
                            gt_sdf_2d_filename = f"sdf_2d_gt_{self.current_model_id}.npy"
                            gt_sdf_2d_path = osp.join(self.output_recon_dir, gt_sdf_2d_filename)
                            gt_sdf_2d_data = self.dataconfig["sdf_projections"].detach().cpu().numpy()
                            np.save(gt_sdf_2d_path, gt_sdf_2d_data)
                            print(f"Saved 2D SDF ground truth: {gt_sdf_2d_path}")
                            
                            # Save ground truth 2D SDF as images
                            for i in range(gt_sdf_2d_data.shape[1]):
                                sdf_img = gt_sdf_2d_data[0, i]  # Get SDF view i
                                # Use RdBu_r colormap for SDF (blue=negative/inside, red=positive/outside)
                                gt_sdf_img_path = osp.join(self.output_recon_dir, f"sdf_2d_gt_{self.current_model_id}_view_{i+1}.png")
                                plt.figure(figsize=(8, 8))
                                plt.imshow(sdf_img, cmap='RdBu_r', vmin=-5, vmax=5)  # Fixed range for better visualization
                                plt.colorbar(label='SDF Value (mm)')
                                plt.title(f'Ground Truth 2D SDF - View {i+1}')
                                plt.savefig(gt_sdf_img_path, dpi=150, bbox_inches='tight')
                                plt.close()
                                print(f"Saved ground truth 2D SDF image {i+1}: {gt_sdf_img_path}")
                        else:
                            print("Warning: No 2D SDF ground truth found in dataconfig")
                        
                        # Save predicted projections as images
                        for i in range(pred_projs_data.shape[1]):
                            proj_img = pred_projs_data[0, i]  # Get projection i
                            # Normalize to 0-255 range
                            proj_img_norm = ((proj_img - proj_img.min()) / (proj_img.max() - proj_img.min()) * 255).astype(np.uint8)
                            pred_proj_img_path = osp.join(self.output_recon_dir, f"pred_projection_{self.current_model_id}_view_{i+1}.png")
                            plt.imsave(pred_proj_img_path, proj_img_norm, cmap='gray')
                            print(f"Saved predicted projection image {i+1}: {pred_proj_img_path}")
                        
                        # Create comparison images showing GT vs Predicted side by side
                        comparison_path = osp.join(self.output_recon_dir, f"comparison_{self.current_model_id}.png")
                        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                        
                        for i in range(2):  # Two views
                            # Ground truth occupancy projection
                            axes[i, 0].imshow(gt_projs_data[0, i], cmap='gray')
                            axes[i, 0].set_title(f'GT Occupancy View {i+1}')
                            axes[i, 0].axis('off')
                            
                            # Predicted occupancy projection
                            axes[i, 1].imshow(pred_projs_data[0, i], cmap='gray')
                            axes[i, 1].set_title(f'Pred Occupancy View {i+1}')
                            axes[i, 1].axis('off')
                            
                            # Ground truth SDF (if available)
                            if "sdf_projections" in self.dataconfig:
                                im3 = axes[i, 2].imshow(gt_sdf_2d_data[0, i], cmap='RdBu_r', vmin=-5, vmax=5)
                                axes[i, 2].set_title(f'GT SDF View {i+1}')
                                axes[i, 2].axis('off')
                            else:
                                axes[i, 2].text(0.5, 0.5, 'No GT SDF', ha='center', va='center')
                                axes[i, 2].axis('off')
                            
                            # Predicted SDF
                            im4 = axes[i, 3].imshow(pred_sdf_2d_data[0, i], cmap='RdBu_r', vmin=-5, vmax=5)
                            axes[i, 3].set_title(f'Pred SDF View {i+1}')
                            axes[i, 3].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"Saved comparison image: {comparison_path}")
                        
                        # Also save network weights with custom name
                        network_filename = f"network_{self.current_model_id}.pth"
                        network_path = osp.join(self.output_recon_dir, network_filename)
                        torch.save({
                            'network': self.net.state_dict(),
                            'network_fine': self.net_fine.state_dict() if self.n_fine > 0 else None,
                            'model_id': self.current_model_id,
                            'epoch': idx_epoch,
                            'config': self.conf
                        }, network_path)
                        print(f"Saved final network: {network_path}")
                    else:
                        # Intermediate epochs: save in eval directory
                        np.save(self.evaldir + "/" + str(idx_epoch), image_pred)
                    
                    # Free memory immediately after evaluation
                    del image_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Train
            self.global_step += 1
            self.net.train()
            
            # Memory-efficient training step
            loss_train = self.train_step_memory_efficient(self.train_dset, global_step=self.global_step, idx_epoch=idx_epoch)
            
            # Track loss for plotting
            self.training_losses.append(loss_train['loss'])
            
            # Update learning rate after optimizer step
            self.lr_scheduler.step()
            
            pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, {loss_train['loss']:.6f}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
            pbar.update(1)
            
            # Save checkpoints (only if save_intermediate is True or at final epoch)
            should_save_checkpoint = False
            if idx_epoch == self.epochs and save_final:
                should_save_checkpoint = True  # Always save at final epoch if save_final is True
            elif save_intermediate and (idx_epoch % self.i_save == 0) and self.i_save > 0 and idx_epoch > 0:
                should_save_checkpoint = True  # Save intermediate checkpoints if save_intermediate is True
                
            if should_save_checkpoint:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                
                # Save with memory-efficient settings
                save_dict = {
                    "epoch": idx_epoch,
                    "network": self.net.state_dict(),
                    "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                    "optimizer": self.optimizer.state_dict(),
                }
                
                if self.use_mixed_precision:
                    save_dict["scaler"] = self.scaler.state_dict()
                    
                torch.save(save_dict, self.ckptdir)
                
                # Clear cache after saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save loss plot after training completion
        self.save_loss_plot()
        
        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def run_network_chunked(self, inputs, fn, chunk_size):
        """
        Memory-efficient network inference with smaller chunks.
        """
        uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        output_chunks = []
        
        for i in range(0, uvt_flat.shape[0], chunk_size):
            chunk = uvt_flat[i:i + chunk_size]
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision, dtype=torch.float16):
                chunk_output = fn(chunk)
            output_chunks.append(chunk_output)
            
            # Clear intermediate tensors
            del chunk, chunk_output
            if torch.cuda.is_available() and i % (chunk_size * 4) == 0:  # Periodic cleanup
                torch.cuda.empty_cache()
        
        out_flat = torch.cat(output_chunks, 0)
        out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
        
        # Clean up
        del output_chunks, out_flat
        
        return out

    def train_step_memory_efficient(self, data, global_step, idx_epoch):
        """
        Memory-efficient training step with gradient accumulation and mixed precision.
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        
        # Gradient accumulation loop
        for acc_step in range(self.gradient_accumulation_steps):
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision, dtype=torch.float16):
                loss = self.compute_loss(data, global_step, idx_epoch)
                scaled_loss = loss["loss"] / self.gradient_accumulation_steps
                total_loss += loss["loss"].item()
            
            # Backward pass with gradient scaling
            self.scaler.scale(scaled_loss).backward()
            
            # Clear intermediate tensors
            del loss, scaled_loss
            
        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Clear gradients and cache
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"loss": total_loss / self.gradient_accumulation_steps}

    def train_step(self, data, global_step, idx_epoch):
        """
        Legacy training step - kept for compatibility
        """
        return self.train_step_memory_efficient(data, global_step, idx_epoch)
        
    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()
        