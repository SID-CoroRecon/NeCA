import torch
import numpy as np

from torch.utils.data import Dataset

class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data, index=0):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"][index]/1000 # Distance Source Detector      (m)
        self.DSO = data["DSO"][index]/1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)


class TIGREDataset(Dataset):
    """
    TIGRE dataset with memory optimizations.
    """
    def __init__(self, data, device="cuda"):    
        super().__init__()

        # Store projections with memory-efficient data type
        if isinstance(data["projections"], torch.Tensor):
            self.projs = data["projections"].to(device=device, dtype=torch.float32)
        else:
            # Convert to half precision if possible to save memory
            proj_data = torch.tensor(data["projections"], dtype=torch.float32, device=device)
            self.projs = proj_data
            
        # Store SDF projections if available
        self.sdf_projs = None
        if "sdf_projections" in data:
            if isinstance(data["sdf_projections"], torch.Tensor):
                self.sdf_projs = data["sdf_projections"].to(device=device, dtype=torch.float32)
            else:
                sdf_proj_data = torch.tensor(data["sdf_projections"], dtype=torch.float32, device=device)
                self.sdf_projs = sdf_proj_data

        self.geo_one = ConeGeometry(data,index=0)
        self.geo_two = ConeGeometry(data, index=1)
        self.n_samples = data["numTrain"]
        geo = self.geo_one
        
        # Create coordinates more memory efficiently
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, geo.nDetector[1] - 1, geo.nDetector[1], device=device, dtype=torch.float32),
            torch.linspace(0, geo.nDetector[0] - 1, geo.nDetector[0], device=device, dtype=torch.float32),
            indexing='ij'
        ), -1)
        self.coords = torch.reshape(coords, [-1, 2])
        
        # Store voxels with memory-efficient precision
        voxel_data = self.get_voxels(geo)
        self.voxels = torch.tensor(voxel_data, dtype=torch.float32, device=device)
        
        # Clear intermediate variables to free memory
        del voxel_data, coords

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        select_coords = self.coords.long() 
        projs = self.projs[index]
        out = {
            "projs": projs,
        }
        # Add SDF projections if available
        if self.sdf_projs is not None:
            out["sdf_projs"] = self.sdf_projs[index]
        return out

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels with memory optimization.
        """
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        # Use float32 instead of float64 to save memory
        xyz = np.meshgrid(np.linspace(-s1, s1, n1, dtype=np.float32),
                        np.linspace(-s2, s2, n2, dtype=np.float32),
                        np.linspace(-s3, s3, n3, dtype=np.float32), indexing="ij")
        voxel = np.asarray(xyz, dtype=np.float32).transpose([1, 2, 3, 0])
        return voxel
