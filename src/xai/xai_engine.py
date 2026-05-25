import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import os

class GradCAM:
    """3D Grad-CAM implementation for Aneurysm Classification Model."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_heatmap(self, input_dict: Dict[str, torch.Tensor], target_class_idx: int, target_head: str = 'ap'):
        """
        Generate a 3D Grad-CAM heatmap.
        target_head: 'ap' for ap_head, 'loc' for cls_head
        target_class_idx: class index within the head
        """
        self.model.eval()
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        # Note: AneurysmVesselSegROILitModule forward handles dict or individual args
        output = self.model(**input_dict)
        
        if target_head == 'ap':
            logit = output['logit_ap'] # (B,)
            score = logit[0] if logit.dim() > 0 else logit
        else:
            logit = output['logits_loc'] # (B, 13)
            score = logit[0, target_class_idx]

        # Backward pass
        score.backward()

        # Compute weights
        # gradients: (B, C, D, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Generate heatmap
        # activations: (B, C, D, H, W)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and upsample
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to input size
        # Input image is (B, 1, D, H, W)
        target_size = input_dict['x'].shape[2:]
        cam_upsampled = F.interpolate(cam, size=target_size, mode='trilinear', align_corners=False)
        
        return cam_upsampled[0, 0].detach().cpu().numpy()

class OcclusionSensitivity:
    """3D Occlusion Sensitivity implementation."""
    def __init__(self, model, patch_size: int = 24, stride: int = 12):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    @torch.no_grad()
    def generate_heatmap(self, input_dict: Dict[str, torch.Tensor], target_class_idx: int, target_head: str = 'ap'):
        self.model.eval()
        image = input_dict['x'] # (B, 1, D, H, W)
        vessel_seg = input_dict.get('vessel_seg')
        vessel_union = input_dict.get('vessel_union')
        
        B, C, D, H, W = image.shape
        
        # Baseline prediction
        output = self.model(**input_dict)
        if target_head == 'ap':
            baseline_prob = torch.sigmoid(output['logit_ap']).item()
        else:
            baseline_prob = torch.sigmoid(output['logits_loc'][0, target_class_idx]).item()
            
        sensitivity_map = np.zeros((D, H, W), dtype=np.float32)
        count_map = np.zeros((D, H, W), dtype=np.float32)
        
        # Sliding window
        for z in range(0, D - self.patch_size + 1, self.stride):
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    # Create occluded input
                    image_occ = image.clone()
                    image_occ[:, :, z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size] = 0
                    
                    # Note: We keep vessel_seg as it is or should we occlude it?
                    # Typically, we just occlude the image signal.
                    
                    occ_dict = {
                        'x': image_occ,
                        'vessel_seg': vessel_seg,
                        'vessel_union': vessel_union,
                        **{k: v for k, v in input_dict.items() if k not in ['x', 'vessel_seg', 'vessel_union']}
                    }
                    
                    output_occ = self.model(**occ_dict)
                    if target_head == 'ap':
                        occ_prob = torch.sigmoid(output_occ['logit_ap']).item()
                    else:
                        occ_prob = torch.sigmoid(output_occ['logits_loc'][0, target_class_idx]).item()
                    
                    # Sensitivity = Baseline - Occluded (higher drop means region is important)
                    drop = baseline_prob - occ_prob
                    sensitivity_map[z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size] += drop
                    count_map[z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Average
        mask = count_map > 0
        sensitivity_map[mask] /= count_map[mask]
        
        # Normalize to [0, 1]
        sensitivity_map = np.maximum(sensitivity_map, 0)
        max_val = sensitivity_map.max()
        if max_val > 0:
            sensitivity_map /= max_val
            
        return sensitivity_map

class XAIPlotter:
    """Utilities for plotting XAI results."""
    def __init__(self):
        # Create light blue colormap: White -> Cyan -> Blue
        colors = [(1, 1, 1, 0), (0, 1, 1, 0.5), (0.4, 0.7, 1, 0.8), (0.2, 0.5, 1, 1)]
        self.cmap = LinearSegmentedColormap.from_list('light_blue', colors, N=256)

    def plot_3view(self, image: np.ndarray, heatmap: np.ndarray, title: str, save_path: str):
        """
        Generate axial, coronal, and sagittal view of image with heatmap overlay.
        image and heatmap should be (D, H, W).
        """
        D, H, W = image.shape
        # Center of heatmap for slicing
        z_idx, y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Better fallback: use center of volume if max is vague
        if heatmap[z_idx, y_idx, x_idx] < 0.1:
            z_idx, y_idx, x_idx = D // 2, H // 2, W // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        # Normalize image for display
        img_disp = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Axial
        axes[0].imshow(img_disp[z_idx], cmap='gray')
        axes[0].imshow(heatmap[z_idx], cmap=self.cmap, alpha=0.6)
        axes[0].set_title(f'Axial (z={z_idx})')
        axes[0].axis('off')

        # Coronal
        axes[1].imshow(img_disp[:, y_idx, :], cmap='gray', origin='lower')
        axes[1].imshow(heatmap[:, y_idx, :], cmap=self.cmap, alpha=0.6, origin='lower')
        axes[1].set_title(f'Coronal (y={y_idx})')
        axes[1].axis('off')

        # Sagittal
        axes[2].imshow(img_disp[:, :, x_idx], cmap='gray', origin='lower')
        axes[2].imshow(heatmap[:, :, x_idx], cmap=self.cmap, alpha=0.6, origin='lower')
        axes[2].set_title(f'Sagittal (x={x_idx})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_original(self, image: np.ndarray, save_path: str):
        D, H, W = image.shape
        z_idx, y_idx, x_idx = D // 2, H // 2, W // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Original Input ROI', fontsize=16)

        img_disp = (image - image.min()) / (image.max() - image.min() + 1e-8)

        axes[0].imshow(img_disp[z_idx], cmap='gray')
        axes[0].set_title(f'Axial (z={z_idx})')
        axes[0].axis('off')

        axes[1].imshow(img_disp[:, y_idx, :], cmap='gray', origin='lower')
        axes[1].set_title(f'Coronal (y={y_idx})')
        axes[1].axis('off')

        axes[2].imshow(img_disp[:, :, x_idx], cmap='gray', origin='lower')
        axes[2].set_title(f'Sagittal (x={x_idx})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
