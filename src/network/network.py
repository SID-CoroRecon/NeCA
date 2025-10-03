import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid", use_gradient_checkpointing=True):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU(inplace=True) for i in range(0, num_layers-1, 1)])  # Use inplace for memory
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "tanh":
            self.activations.append(nn.Tanh())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU(inplace=True))
        elif last_activation == "linear" or last_activation == "none":
            self.activations.append(nn.Identity())  # No activation - pure regression
        else:
            raise NotImplementedError("Unknown last activation")

    def _forward_layers(self, x, input_pts, start_layer=0, end_layer=None):
        """
        Forward pass through a subset of layers for gradient checkpointing.
        """
        if end_layer is None:
            end_layer = len(self.layers)
            
        for i in range(start_layer, end_layer):
            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x

    def forward(self, x):
        # Encode input
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            # Split the network into checkpointed segments
            mid_layer = len(self.layers) // 2
            
            # First half of layers
            x = checkpoint(self._forward_layers, x, input_pts, 0, mid_layer, use_reentrant=False)
            
            # Second half of layers  
            x = checkpoint(self._forward_layers, x, input_pts, mid_layer, len(self.layers), use_reentrant=False)
            
        else:
            # Standard forward pass for inference
            x = self._forward_layers(x, input_pts, 0, len(self.layers))
        
        return x
    