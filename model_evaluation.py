import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from model import get_model, ModelType
from preprocess import get_test_loader

def init_device():
    """Initialize device and check for multi-GPU setup"""
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"GPUs detected: {n_gpu}")
        return torch.device("cuda:0"), n_gpu > 1
    return torch.device("cpu"), False

class EnhancedMetrics:
    """Enhanced metrics calculation including PSNR, SSIM, NDVI, and FLIP"""
    
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    def calculate_flip(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """NVIDIA FLIP-inspired perceptual metric"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        batch_scores = []
        for i in range(pred_np.shape[0]):
            p = np.transpose(pred_np[i], (1, 2, 0))
            t = np.transpose(target_np[i], (1, 2, 0))
            
            if p.shape[2] >= 3:
                p_lab = cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
                t_lab = cv2.cvtColor((t * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
                color_diff = np.sqrt(np.sum((p_lab - t_lab) ** 2, axis=2))
                flip_score = np.mean(color_diff) / 255.0
            else:
                flip_score = np.mean(np.abs(p - t))
            
            batch_scores.append(flip_score)
        
        return torch.tensor(np.mean(batch_scores), device=pred.device)
    
    def calculate_ndvi(self, pred: torch.Tensor, target: torch.Tensor, config: str) -> Dict[str, torch.Tensor]:
        """Calculate NDVI metrics based on configuration"""
        results = {'ndvi_applicable': False}
        
        if config == 'a':
            # RGB only - no NDVI possible
            results['ndvi_error'] = torch.tensor(0.0, device=pred.device)
            return results
        
        elif config == 'b':
            # NIR-SWIR-RedEdge configuration
            nir_pred = pred[:, 0:1]
            red_edge_pred = pred[:, 2:3]
            nir_target = target[:, 0:1]
            red_edge_target = target[:, 2:3]
            
            ndvi_pred = (nir_pred - red_edge_pred) / (nir_pred + red_edge_pred + 1e-8)
            ndvi_target = (nir_target - red_edge_target) / (nir_target + red_edge_target + 1e-8)
            
        elif config == 'c':
            # RGB+NIR configuration - true NDVI
            red_pred = pred[:, 0:1]
            nir_pred = pred[:, 3:4]
            red_target = target[:, 0:1]
            nir_target = target[:, 3:4]
            
            ndvi_pred = (nir_pred - red_pred) / (nir_pred + red_pred + 1e-8)
            ndvi_target = (nir_target - red_target) / (nir_target + red_target + 1e-8)
        
        if config in ['b', 'c']:
            results['ndvi_applicable'] = True
            results['ndvi_mae'] = torch.mean(torch.abs(ndvi_pred - ndvi_target))
            results['ndvi_rmse'] = torch.sqrt(torch.mean((ndvi_pred - ndvi_target) ** 2))
            
            # Calculate correlation
            pred_flat = ndvi_pred.view(-1)
            target_flat = ndvi_target.view(-1)
            pred_centered = pred_flat - torch.mean(pred_flat)
            target_centered = target_flat - torch.mean(target_flat)
            correlation = torch.sum(pred_centered * target_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(target_centered ** 2)) + 1e-8
            )
            results['ndvi_correlation'] = correlation
        
        return results

class FeatureVisualizer:
    """Feature visualization and analysis capabilities"""
    
    def __init__(self, device, output_dir='./generated_samples'):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Feature storage
        self.feature_maps = {}
        self.hooks = []
    
    def register_hooks(self, model):
        """Register forward hooks for feature extraction"""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        if hasattr(model.gen_EO, 'initial'):
            model.gen_EO.initial[0].register_forward_hook(get_activation('encoder_input'))
            
        if hasattr(model.gen_EO, 'res_blocks'):
            for i, block in enumerate(model.gen_EO.res_blocks):
                block.register_forward_hook(get_activation(f'residual_block_{i}'))
        
        if hasattr(model.gen_EO, 'last'):
            model.gen_EO.last.register_forward_hook(get_activation('decoder_output'))
    
    def generate_feature_maps(self, model, sar_input, config, batch_idx=0):
        """Generate and save feature map visualizations"""
        self.register_hooks(model)
        
        with torch.no_grad():
            # Forward pass to collect features
            _ = model.gen_EO(sar_input[:1])  # Use only first sample
            
            # Visualize key feature maps
            for layer_name, features in self.feature_maps.items():
                self._save_feature_map(features, layer_name, config, batch_idx)
            
            self.feature_maps.clear()
    
    def _save_feature_map(self, features, layer_name, config, batch_idx):
        """Save individual feature map visualization"""
        if features.dim() != 4:
            return
        
        # Take first sample and first 16 channels
        feat = features[0]
        n_channels = min(16, feat.size(0))
        
        # Create grid of feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Config {config} - {layer_name} - Batch {batch_idx}', fontsize=16)
        
        for i in range(16):
            row, col = i // 4, i % 4
            if i < n_channels:
                feat_map = feat[i].cpu().numpy()
                axes[row, col].imshow(feat_map, cmap='viridis')
                axes[row, col].set_title(f'Channel {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f'features_{config}_{layer_name}_batch_{batch_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_activation_heatmaps(self, model, sar_input, pred_output, config, batch_idx=0):
        """Generate activation-based attention maps without gradients"""
        
        # Hook to capture intermediate activations
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hook on a key layer
        hook_handle = None
        try:
            if hasattr(model.gen_EO, 'res_blocks') and len(model.gen_EO.res_blocks) > 0:
                hook_handle = model.gen_EO.res_blocks[-1].register_forward_hook(get_activation('last_residual'))
            
            with torch.no_grad():
                # Forward pass to collect activations
                _ = model.gen_EO(sar_input[:1])
                
                if 'last_residual' in activations:
                    # Create attention map from activation magnitudes
                    activation_map = activations['last_residual'][0]  # First sample
                    
                    # Average across channels and normalize
                    attention = torch.mean(activation_map, dim=0)
                    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
                    
                    # Resize to input dimensions if needed
                    if attention.shape != sar_input.shape[-2:]:
                        attention = F.interpolate(
                            attention.unsqueeze(0).unsqueeze(0), 
                            size=sar_input.shape[-2:], 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze()
                    
                    # Save visualization
                    plt.figure(figsize=(10, 5))
                    
                    plt.subplot(1, 2, 1)
                    sar_display = torch.mean(sar_input[0], dim=0).cpu().numpy()
                    plt.imshow(sar_display, cmap='gray')
                    plt.title('SAR Input')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(attention.cpu().numpy(), cmap='hot', alpha=0.7)
                    plt.title('Activation-based Attention')
                    plt.axis('off')
                    
                    save_path = self.output_dir / f'attention_{config}_batch_{batch_idx}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
        except Exception as e:
            print(f"Warning: Could not generate attention map for batch {batch_idx}: {str(e)}")
        finally:
            # Clean up hook
            if hook_handle:
                hook_handle.remove()
        
    def generate_error_maps(self, pred, target, sar, config, batch_idx=0):
        """Generate comprehensive error analysis maps"""
        n_samples = min(4, pred.size(0))
        
        # Calculate different error types
        l1_error = torch.abs(pred - target)[:n_samples]
        l2_error = (pred - target) ** 2 * 10  # Scaled for visibility
        
        if config == 'c' and target.size(1) == 4:
            # Handle RGB+NIR separately
            rgb_pred, nir_pred = pred[:n_samples, :3], pred[:n_samples, 3:4]
            rgb_target, nir_target = target[:n_samples, :3], target[:n_samples, 3:4]
            
            # RGB error maps
            rgb_grid = torch.cat([
                sar[:n_samples], rgb_pred, rgb_target, 
                l1_error[:, :3], l2_error[:, :3]
            ], dim=0)
            save_image(rgb_grid, self.output_dir / f'error_analysis_{config}_rgb_batch_{batch_idx}.png', nrow=n_samples)
            
            # NIR error maps
            nir_expanded = nir_pred.expand(-1, 3, -1, -1)
            nir_target_expanded = nir_target.expand(-1, 3, -1, -1)
            nir_error_expanded = l1_error[:, 3:4].expand(-1, 3, -1, -1)
            
            nir_grid = torch.cat([
                torch.zeros_like(sar[:n_samples]), nir_expanded, nir_target_expanded, nir_error_expanded
            ], dim=0)
            save_image(nir_grid, self.output_dir / f'error_analysis_{config}_nir_batch_{batch_idx}.png', nrow=n_samples)
        else:
            # Standard error visualization
            error_grid = torch.cat([
                sar[:n_samples], pred[:n_samples], target[:n_samples], 
                l1_error, l2_error
            ], dim=0)
            save_image(error_grid, self.output_dir / f'error_analysis_{config}_batch_{batch_idx}.png', nrow=n_samples)

class EnhancedModelEvaluator:
    """Comprehensive SAR-to-EO model evaluator with feature visualization"""
    
    def __init__(self, model_config, checkpoint_paths, output_dir='./test_results'):
        self.device, self.dp = init_device()
        self.model_config = model_config
        self.checkpoint_paths = checkpoint_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metrics and visualizer
        self.metrics = EnhancedMetrics(self.device)
        self.visualizer = FeatureVisualizer(self.device, self.output_dir / 'visualizations')
        
        # Load model
        self.cycleGAN = self._load_trained_model()
    
    def _load_trained_model(self):
        """Load pre-trained CycleGAN model"""
        print(f"Loading model configuration: {self.model_config}")
        
        cycleGAN = get_model(self.model_config)
        
        # Load checkpoints
        for component, path in self.checkpoint_paths.items():
            if 'gen_sar' in component:
                self._load_checkpoint_simple(path, cycleGAN.gen_SAR, cycleGAN.opt_gen)
            elif 'gen_eo' in component:
                self._load_checkpoint_simple(path, cycleGAN.gen_EO, cycleGAN.opt_gen)
            elif 'disc_sar' in component:
                self._load_checkpoint_simple(path, cycleGAN.disc_SAR, cycleGAN.opt_disc)
            elif 'disc_eo' in component:
                self._load_checkpoint_simple(path, cycleGAN.disc_EO, cycleGAN.opt_disc)
        
        # Multi-GPU setup
        if self.dp:
            cycleGAN.gen_EO = nn.DataParallel(cycleGAN.gen_EO)
            cycleGAN.gen_SAR = nn.DataParallel(cycleGAN.gen_SAR)
            cycleGAN.disc_EO = nn.DataParallel(cycleGAN.disc_EO)
            cycleGAN.disc_SAR = nn.DataParallel(cycleGAN.disc_SAR)
        
        # Move to device and set eval mode
        cycleGAN.gen_EO = cycleGAN.gen_EO.to(self.device).eval()
        cycleGAN.gen_SAR = cycleGAN.gen_SAR.to(self.device).eval()
        cycleGAN.disc_EO = cycleGAN.disc_EO.to(self.device).eval()
        cycleGAN.disc_SAR = cycleGAN.disc_SAR.to(self.device).eval()
        
        return cycleGAN
    
    def _load_checkpoint_simple(self, checkpoint_file, model, optimizer):
        """Load checkpoint with error handling"""
        print(f"=> Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(checkpoint["state_dict"])
        
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        print("✓ Checkpoint loaded successfully")
    
    def evaluate_on_test_set(self, test_loader, config_name):
        """Comprehensive evaluation with feature visualization"""
        print(f"\n=== ENHANCED EVALUATION - CONFIG {config_name.upper()} ===")
        print(f"Test dataset size: {len(test_loader.dataset)} samples")
        
        # Metrics accumulators
        metrics_accumulator = {
            'psnr': [], 'ssim': [], 'flip': [],
            'ndvi_mae': [], 'ndvi_rmse': [], 'ndvi_correlation': []
        }
        
        batch_results = []
        visualization_count = 0
        
        with torch.no_grad():
            for batch_idx, (sar, eo) in enumerate(tqdm(test_loader, desc=f"Evaluating Config {config_name.upper()}")):
                sar = sar.to(self.device)
                eo = eo.to(self.device)
                
                # Generate predictions
                fake_eo = self.cycleGAN.gen_EO(sar)
                
                # Denormalize
                fake_eo_norm = fake_eo * 0.5 + 0.5
                eo_norm = eo * 0.5 + 0.5
                sar_norm = sar * 0.5 + 0.5
                
                # Calculate core metrics
                batch_metrics = self._calculate_batch_metrics(fake_eo_norm, eo_norm, config_name)
                
                # Store results
                batch_results.append({
                    'batch_idx': batch_idx,
                    'batch_size': sar.size(0),
                    **batch_metrics
                })
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in metrics_accumulator and value is not None:
                        metrics_accumulator[key].append(value)
                
                # Generate visualizations for first few batches
                if visualization_count < 3:
                    self._generate_batch_visualizations(
                        fake_eo_norm, eo_norm, sar_norm, config_name, batch_idx
                    )
                    visualization_count += 1
                
                # Progress update
                if batch_idx % 25 == 0:
                    current_psnr = batch_metrics.get('psnr', 0)
                    current_ssim = batch_metrics.get('ssim', 0)
                    print(f"Batch {batch_idx}: PSNR={current_psnr:.3f}, SSIM={current_ssim:.3f}")
        
        # Calculate final statistics
        results = self._compile_final_results(metrics_accumulator, batch_results, config_name)
        
        # Generate summary visualizations
        self._generate_summary_visualizations(results, config_name)
        
        return results
    
    def _calculate_batch_metrics(self, pred, target, config):
        """Calculate all metrics for a batch"""
        # Handle channel-specific evaluation
        if config == 'c' and target.size(1) == 4:
            # Use RGB channels for PSNR/SSIM, full for NDVI
            pred_rgb = pred[:, :3]
            target_rgb = target[:, :3]
            psnr_val = self.metrics.psnr(pred_rgb, target_rgb).item()
            ssim_val = self.metrics.ssim(pred_rgb, target_rgb).item()
            flip_val = self.metrics.calculate_flip(pred_rgb, target_rgb).item()
        else:
            psnr_val = self.metrics.psnr(pred, target).item()
            ssim_val = self.metrics.ssim(pred, target).item()
            flip_val = self.metrics.calculate_flip(pred, target).item()
        
        # NDVI metrics
        ndvi_results = self.metrics.calculate_ndvi(pred, target, config)
        
        return {
            'psnr': psnr_val,
            'ssim': ssim_val,
            'flip': flip_val,
            'ndvi_mae': ndvi_results.get('ndvi_mae', torch.tensor(0.0)).item() if ndvi_results.get('ndvi_applicable') else None,
            'ndvi_rmse': ndvi_results.get('ndvi_rmse', torch.tensor(0.0)).item() if ndvi_results.get('ndvi_applicable') else None,
            'ndvi_correlation': ndvi_results.get('ndvi_correlation', torch.tensor(0.0)).item() if ndvi_results.get('ndvi_applicable') else None,
            'ndvi_applicable': ndvi_results.get('ndvi_applicable', False)
        }
    
    def _generate_batch_visualizations(self, pred, target, sar, config, batch_idx):
        """Generate comprehensive visualizations for a batch"""
        # Feature maps (this works fine)
        self.visualizer.generate_feature_maps(self.cycleGAN, sar, config, batch_idx)
        
        # TEMPORARILY DISABLE: Comment out the problematic attention heatmaps
        # self.visualizer.generate_activation_heatmaps(self.cycleGAN, sar, pred, config, batch_idx)
        
        # Error analysis (this works fine)
        self.visualizer.generate_error_maps(pred, target, sar, config, batch_idx)
        
        # NDVI visualization for applicable configs
        if config in ['b', 'c']:
            self._generate_ndvi_visualization(pred, target, config, batch_idx)

    
    def _generate_ndvi_visualization(self, pred, target, config, batch_idx):
        """Generate NDVI-specific visualizations"""
        n_samples = min(4, pred.size(0))
        
        if config == 'b':
            # NIR-SWIR-RedEdge
            nir_pred = pred[:n_samples, 0:1]
            red_edge_pred = pred[:n_samples, 2:3]
            nir_target = target[:n_samples, 0:1]
            red_edge_target = target[:n_samples, 2:3]
            
            ndvi_pred = (nir_pred - red_edge_pred) / (nir_pred + red_edge_pred + 1e-8)
            ndvi_target = (nir_target - red_edge_target) / (nir_target + red_edge_target + 1e-8)
            
        elif config == 'c':
            # RGB+NIR
            red_pred = pred[:n_samples, 0:1]
            nir_pred = pred[:n_samples, 3:4]
            red_target = target[:n_samples, 0:1]
            nir_target = target[:n_samples, 3:4]
            
            ndvi_pred = (nir_pred - red_pred) / (nir_pred + red_pred + 1e-8)
            ndvi_target = (nir_target - red_target) / (nir_target + red_target + 1e-8)
        
        # Normalize NDVI for visualization
        ndvi_pred_vis = (ndvi_pred + 1) / 2
        ndvi_target_vis = (ndvi_target + 1) / 2
        ndvi_error = torch.abs(ndvi_pred_vis - ndvi_target_vis)
        
        # Create NDVI comparison
        ndvi_grid = torch.cat([
            ndvi_pred_vis.expand(-1, 3, -1, -1),
            ndvi_target_vis.expand(-1, 3, -1, -1),
            ndvi_error.expand(-1, 3, -1, -1)
        ], dim=0)
        
        save_image(ndvi_grid, 
                  self.output_dir / 'visualizations' / f'ndvi_analysis_{config}_batch_{batch_idx}.png', 
                  nrow=n_samples)
    
    def _compile_final_results(self, metrics_accumulator, batch_results, config):
        """Compile final evaluation results"""
        def safe_stats(values):
            if not values or all(v is None for v in values):
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            clean_values = [v for v in values if v is not None]
            return {
                'mean': np.mean(clean_values),
                'std': np.std(clean_values),
                'min': np.min(clean_values),
                'max': np.max(clean_values)
            }
        
        results = {
            'config': config,
            'total_samples': sum(b['batch_size'] for b in batch_results),
            'total_batches': len(batch_results),
            'model_architecture': str(self.model_config),
        }
        
        # Add statistics for each metric
        for metric_name, values in metrics_accumulator.items():
            stats = safe_stats(values)
            results[f'{metric_name}_mean'] = stats['mean']
            results[f'{metric_name}_std'] = stats['std']
            results[f'{metric_name}_min'] = stats['min']
            results[f'{metric_name}_max'] = stats['max']
        
        # NDVI applicability
        results['ndvi_applicable'] = config in ['b', 'c']
        
        return results
    
    def _generate_summary_visualizations(self, results, config):
        """Generate summary plots and analysis"""
        # Metric distribution plots
        metrics_to_plot = ['psnr', 'ssim', 'flip']
        if results.get('ndvi_applicable', False):
            metrics_to_plot.extend(['ndvi_mae', 'ndvi_rmse', 'ndvi_correlation'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Configuration {config.upper()} - Metric Summary', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            if i >= 6:
                break
            row, col = i // 3, i % 3
            
            mean_val = results.get(f'{metric}_mean', 0)
            std_val = results.get(f'{metric}_std', 0)
            
            axes[row, col].bar(['Mean'], [mean_val], yerr=[std_val], capsize=5)
            axes[row, col].set_title(f'{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}')
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(metrics_to_plot), 6):
            row, col = i // 3, i % 3
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'summary_metrics_{config}.png', dpi=300, bbox_inches='tight')
        plt.close()

def get_test_loader_wrapper(base_dir: str, config: str, batch_size: int):
    """Wrapper for test loader initialization"""
    return get_test_loader(base_dir, config, [8,49,81,115,146],batch_size)

def main():
    """Enhanced evaluation pipeline with comprehensive analysis"""
    
    # Configuration
    BASE_DIR = "/kaggle/input/satellitessubset"  # Update to your data path
    BATCH_SIZE = 16
    OUTPUT_DIR = "./enhanced_test_results"
    
    # Checkpoint configurations
    checkpoint_configs = {
        'a': {
            'model_type': ModelType.SAR_TO_EORGB,
            'paths': {
                'gen_sar': "/kaggle/input/newcheckpoints/rgb/eorgb_gensar.pth.tar",
                'gen_eo': "/kaggle/input/newcheckpoints/rgb/eorgb_geneo.pth.tar",
                'disc_sar': "/kaggle/input/newcheckpoints/rgb/eorgb_criticsar.pth.tar",
                'disc_eo': "/kaggle/input/newcheckpoints/rgb/eorgb_criticeo.pth.tar",
            }
        },
        'b': {
            'model_type': ModelType.SAR_TO_EONIRSWIR,
            'paths': {
                'gen_sar': "/kaggle/input/newcheckpoints/nirswir/eonirswir_gensar.pth.tar",
                'gen_eo': "/kaggle/input/newcheckpoints/nirswir/eonirswir_geneo.pth.tar",
                'disc_sar': "/kaggle/input/newcheckpoints/nirswir/eonirswir_criticsar.pth.tar",
                'disc_eo': "/kaggle/input/newcheckpoints/nirswir/eonirswir_criticeo.pth.tar",
            }
        },
        'c': {
            'model_type': ModelType.SAR_TO_EORGBNIR,
            'paths': {
                'gen_sar': "/kaggle/input/newcheckpoints/rgbnir/eorgbnir_gensar.pth.tar",
                'gen_eo': "/kaggle/input/newcheckpoints/rgbnir/eorgbnir_geneo.pth.tar",
                'disc_sar': "/kaggle/input/newcheckpoints/rgbnir/eorgbnir_criticsar.pth.tar",
                'disc_eo': "/kaggle/input/newcheckpoints/rgbnir/eorgbnir_criticeo.pth.tar",
            }
        }
    }
    
    configs_to_test = ['a', 'b', 'c']
    all_results = {}
    
    for config in configs_to_test:
        print(f"\n{'='*60}")
        print(f"ENHANCED EVALUATION - CONFIGURATION {config.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load test data
            test_loader = get_test_loader_wrapper(BASE_DIR, config, BATCH_SIZE)
            
            # Initialize enhanced evaluator
            evaluator = EnhancedModelEvaluator(
                model_config=checkpoint_configs[config]['model_type'],
                checkpoint_paths=checkpoint_configs[config]['paths'],
                output_dir=OUTPUT_DIR
            )
            
            # Run comprehensive evaluation
            results = evaluator.evaluate_on_test_set(test_loader, config)
            all_results[config] = results
            
            # Print detailed results
            print(f"\n=== FINAL RESULTS - CONFIG {config.upper()} ===")
            print(f"Total samples: {results['total_samples']}")
            print(f"PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f} dB")
            print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
            print(f"FLIP: {results['flip_mean']:.4f} ± {results['flip_std']:.4f}")
            
            if results.get('ndvi_applicable', False):
                print(f"NDVI MAE: {results['ndvi_mae_mean']:.4f} ± {results['ndvi_mae_std']:.4f}")
                print(f"NDVI RMSE: {results['ndvi_rmse_mean']:.4f} ± {results['ndvi_rmse_std']:.4f}")
                print(f"NDVI Correlation: {results['ndvi_correlation_mean']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating configuration {config}: {str(e)}")
            continue
    
    # Save comprehensive results
    if all_results:
        print(f"\n{'='*60}")
        print("COMPREHENSIVE EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        # Cross-configuration comparison
        comparison_metrics = ['psnr_mean', 'ssim_mean', 'flip_mean']
        
        print("\nCross-Configuration Comparison:")
        for metric in comparison_metrics:
            print(f"\n{metric.upper()}:")
            for config, results in all_results.items():
                value = results.get(metric, 0)
                print(f"  Config {config.upper()}: {value:.4f}")
        
        # Save results to JSON
        results_file = Path(OUTPUT_DIR) / "enhanced_evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for config, results in all_results.items():
                json_results[config] = {
                    key: float(value) if isinstance(value, (np.float32, np.float64, np.integer)) else value 
                    for key, value in results.items()
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Visualizations saved to: {OUTPUT_DIR}/visualizations/")

if __name__ == "__main__":
    main()
