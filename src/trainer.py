import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from shutil import copyfile
import numpy as np
import math
import yaml

from .dataset import TIGREDataset as Dataset
from .network import get_network
from .encoder import get_encoder
from src.render import run_network

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
  
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        #######################################
        configPath = cfg['exp']['dataconfig']
        with open(configPath, "r") as handle:
            data = yaml.safe_load(handle)

        # data["projections"] = np.load(data["datadir"] + '_projs.npy')
        # Load projections if they exist, otherwise they will be generated from GT volume
        if os.path.exists(data['datadir']):
            data["projections"] = np.load(data['datadir'])
            print(f"Loaded existing projections: {data['projections'].shape}")
        else:
            print("No existing projections found. Will generate from 3D volume.")

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
        # Load 3D ground truth volume and generate projections
        if "GT_volume_path" in data:
            phantom = np.load(data["GT_volume_path"])
            phantom = np.transpose(phantom, (1,2,0))[::,::-1,::-1]
            phantom = np.transpose(phantom, (2,1,0))[::-1,::,::].copy()
            phantom = torch.tensor(phantom, dtype=torch.float32)[None, ...]

            train_projs_one = self.ct_projector_first.forward_project(phantom)
            train_projs_two = self.ct_projector_second.forward_project(phantom)

            data["projections"] = torch.cat((train_projs_one,train_projs_two), 1)
            print(f"Generated projections from 3D volume: {data['projections'].shape}")

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

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

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
                
            # Evaluate with memory efficiency
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0:  
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
            
            # Update learning rate after optimizer step
            self.lr_scheduler.step()
            
            pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, {loss_train['loss']:.6f}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
            pbar.update(1)
            
            # Save checkpoints
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
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

            # Log learning rate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

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
        