import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CycleGANEvaluator:
    def __init__(self):
        self.config_names = {
            'config_a': 'RGB (B4, B3, B2)',
            'config_b': 'RGB+NIR (B4, B3, B2, B8)',
            'config_c': 'NIR-SWIR-RedEdge (B8, B11, B5)', 
        }
        self.colors = ['#2E86C1', '#E74C3C', '#28B463']
        
    def load_training_logs(self, config_a_path=None, config_b_path=None, config_c_path=None):
        self.data = {}  # Reset data
        
        paths = {
            'config_a': config_a_path,
            'config_b': config_b_path, 
            'config_c': config_c_path
        }
        
        print(" Loading Training Logs...")
        loaded_count = 0
        
        for config, path in paths.items():
            if path and os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        self.data[config] = df
                        loaded_count += 1
                        print(f"✓ Loaded {self.config_names[config]}: {len(df)} epochs")
                    else:
                        print(f" Empty file: {path}")
                except Exception as e:
                    print(f"✗ Error loading {config}: {e}")
            elif path:
                print(f" File not found: {path}")
            else:
                print(f" No path provided for {config}")
        
        if loaded_count == 0:
            print(" No training logs were successfully loaded")
        elif loaded_count < 3:
            print(f" Only {loaded_count}/3 configurations loaded successfully")
        else:
            print(f"All {loaded_count} configurations loaded successfully")
        
        return loaded_count    
    

    def plot_losses(self, figsize=(15, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Generator Loss
        for i, (config, df) in enumerate(self.data.items()):
            ax1.plot(df['epoch'], df['generator_loss'], 
                    color=self.colors[i], linewidth=2, marker='o', markersize=4,
                    label=self.config_names[config])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Generator Loss')
        ax1.set_title('Generator Loss Over Training', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Discriminator Loss  
        for i, (config, df) in enumerate(self.data.items()):
            ax2.plot(df['epoch'], df['discriminator_loss'], 
                    color=self.colors[i], linewidth=2, marker='s', markersize=4,
                    label=self.config_names[config])
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Discriminator Loss')
        ax2.set_title('Discriminator Loss Over Training', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_validation_metrics(self, figsize=(15, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # PSNR
        for i, (config, df) in enumerate(self.data.items()):
            ax1.plot(df['epoch'], df['validation_psnr'], 
                    color=self.colors[i], linewidth=2, marker='o', markersize=4,
                    label=self.config_names[config])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Validation PSNR Over Training', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SSIM
        for i, (config, df) in enumerate(self.data.items()):
            ax2.plot(df['epoch'], df['validation_ssim'], 
                    color=self.colors[i], linewidth=2, marker='s', markersize=4,
                    label=self.config_names[config])
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Validation SSIM Over Training', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_learning_rates(self, figsize=(15, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Generator Learning Rate
        for i, (config, df) in enumerate(self.data.items()):
            ax1.semilogy(df['epoch'], df['learning_rate_gen'], 
                        color=self.colors[i], linewidth=2, marker='o', markersize=4,
                        label=self.config_names[config])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate (log scale)')
        ax1.set_title('Generator Learning Rate Schedule', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Discriminator Learning Rate
        for i, (config, df) in enumerate(self.data.items()):
            ax2.semilogy(df['epoch'], df['learning_rate_disc'], 
                        color=self.colors[i], linewidth=2, marker='s', markersize=4,
                        label=self.config_names[config])
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate (log scale)')
        ax2.set_title('Discriminator Learning Rate Schedule', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_discriminator_scores(self, figsize=(15, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Real Scores
        for i, (config, df) in enumerate(self.data.items()):
            ax1.plot(df['epoch'], df['discriminator_eo_real_score'], 
                    color=self.colors[i], linewidth=2, marker='o', markersize=4,
                    label=self.config_names[config])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Real Score')
        ax1.set_title('Discriminator Real Scores Over Training', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 0.8)
        
        # Fake Scores
        for i, (config, df) in enumerate(self.data.items()):
            ax2.plot(df['epoch'], df['discriminator_eo_fake_score'], 
                    color=self.colors[i], linewidth=2, marker='s', markersize=4,
                    label=self.config_names[config])
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Fake Score')
        ax2.set_title('Discriminator Fake Scores Over Training', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.2, 0.4)
        
        plt.tight_layout()
        plt.show()
        

    def create_comparison_table(self):
        comparison_data = []
        
        # Check if data is loaded for multiple configurations
        if not self.data:
            print(" No training data loaded. Please load training logs first.")
            return None
        
        if len(self.data) == 1:
            print(f" Only {len(self.data)} configuration loaded. Expected 3 configurations.")
            print(f"Loaded: {list(self.data.keys())}")
        
        for config, df in self.data.items():
            if df.empty:
                print(f" No data found for {config}")
                continue
                
            final_epoch = df.iloc[-1]
            comparison_data.append({
                'Configuration': self.config_names[config],
                'Final PSNR': f"{final_epoch['validation_psnr']:.2f} dB",
                'Final SSIM': f"{final_epoch['validation_ssim']:.4f}",
                'Generator Loss': f"{final_epoch['generator_loss']:.3f}",
                'Discriminator Loss': f"{final_epoch['discriminator_loss']:.3f}",
                'Real Score': f"{final_epoch['discriminator_eo_real_score']:.3f}",
                'Fake Score': f"{final_epoch['discriminator_eo_fake_score']:.3f}"
            })
        
        if not comparison_data:
            print(" No valid comparison data to display")
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create a styled table plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=comparison_df.values,
                        colLabels=comparison_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style the rows
        for i in range(1, len(comparison_df) + 1):
            for j in range(len(comparison_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
        
        plt.title('Final Epoch Performance Comparison', fontweight='bold', pad=20, fontsize=14)
        plt.show()
        
        return comparison_df

    def plot_comprehensive_dashboard(self, figsize=(20, 12)):
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Generator Loss
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (config, df) in enumerate(self.data.items()):
            ax1.plot(df['epoch'], df['generator_loss'], 
                    color=self.colors[i], linewidth=2, label=self.config_names[config])
        ax1.set_title('Generator Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Discriminator Loss
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (config, df) in enumerate(self.data.items()):
            ax2.plot(df['epoch'], df['discriminator_loss'], 
                    color=self.colors[i], linewidth=2, label=self.config_names[config])
        ax2.set_title('Discriminator Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # PSNR
        ax3 = fig.add_subplot(gs[0, 2])
        for i, (config, df) in enumerate(self.data.items()):
            ax3.plot(df['epoch'], df['validation_psnr'], 
                    color=self.colors[i], linewidth=2, label=self.config_names[config])
        ax3.set_title('Validation PSNR', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('PSNR (dB)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # SSIM
        ax4 = fig.add_subplot(gs[1, 0])
        for i, (config, df) in enumerate(self.data.items()):
            ax4.plot(df['epoch'], df['validation_ssim'], 
                    color=self.colors[i], linewidth=2, label=self.config_names[config])
        ax4.set_title('Validation SSIM', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('SSIM')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        
        # Real Scores
        ax5 = fig.add_subplot(gs[1, 1])
        for i, (config, df) in enumerate(self.data.items()):
            ax5.plot(df['epoch'], df['discriminator_eo_real_score'], 
                    color=self.colors[i], linewidth=2, label=self.config_names[config])
        ax5.set_title('Discriminator Real Scores', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Real Score')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)
        
        # Fake Scores
        ax6 = fig.add_subplot(gs[1, 2])
        for i, (config, df) in enumerate(self.data.items()):
            ax6.plot(df['epoch'], df['discriminator_eo_fake_score'], 
                    color=self.colors[i], linewidth=2, label=self.config_names[config])
        ax6.set_title('Discriminator Fake Scores', fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Fake Score')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=8)
        
        # Learning Rates (Combined)
        ax7 = fig.add_subplot(gs[2, :2])
        for i, (config, df) in enumerate(self.data.items()):
            ax7.semilogy(df['epoch'], df['learning_rate_gen'], 
                        color=self.colors[i], linewidth=2, linestyle='-', 
                        label=f"{self.config_names[config]} (Gen)")
            ax7.semilogy(df['epoch'], df['learning_rate_disc'], 
                        color=self.colors[i], linewidth=2, linestyle='--', 
                        label=f"{self.config_names[config]} (Disc)")
        ax7.set_title('Learning Rate Schedules', fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Learning Rate (log scale)')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8, ncol=2)
        
        # PSNR vs SSIM Scatter
        ax8 = fig.add_subplot(gs[2, 2])
        for i, (config, df) in enumerate(self.data.items()):
            ax8.scatter(df['validation_psnr'], df['validation_ssim'], 
                       color=self.colors[i], alpha=0.7, s=50,
                       label=self.config_names[config])
        ax8.set_title('PSNR vs SSIM', fontweight='bold')
        ax8.set_xlabel('PSNR (dB)')
        ax8.set_ylabel('SSIM')
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        
        plt.suptitle('CycleGAN SAR-to-EO Translation: Training Performance Dashboard', 
                     fontweight='bold', fontsize=16, y=0.98)
        plt.show()
    
    def analyze_best_configuration(self):
        print("="*60)
        print("CONFIGURATION ANALYSIS")
        print("="*60)
        
        best_psnr = {'config': None, 'value': 0, 'epoch': 0}
        best_ssim = {'config': None, 'value': 0, 'epoch': 0}
        most_stable = {'config': None, 'psnr_std': float('inf'), 'ssim_std': float('inf')}
        
        for config, df in self.data.items():
            # Find best PSNR
            max_psnr_idx = df['validation_psnr'].idxmax()
            if df.loc[max_psnr_idx, 'validation_psnr'] > best_psnr['value']:
                best_psnr = {
                    'config': config,
                    'value': df.loc[max_psnr_idx, 'validation_psnr'],
                    'epoch': df.loc[max_psnr_idx, 'epoch']
                }
            
            # Find best SSIM
            max_ssim_idx = df['validation_ssim'].idxmax()
            if df.loc[max_ssim_idx, 'validation_ssim'] > best_ssim['value']:
                best_ssim = {
                    'config': config,
                    'value': df.loc[max_ssim_idx, 'validation_ssim'],
                    'epoch': df.loc[max_ssim_idx, 'epoch']
                }
            
            # Calculate stability (last 10 epochs)
            last_10 = df.tail(10)
            psnr_std = last_10['validation_psnr'].std()
            ssim_std = last_10['validation_ssim'].std()
            
            if psnr_std < most_stable['psnr_std']:
                most_stable = {
                    'config': config,
                    'psnr_std': psnr_std,
                    'ssim_std': ssim_std
                }
        
        print(f"Best PSNR: {self.config_names[best_psnr['config']]}")
        print(f"   Value: {best_psnr['value']:.2f} dB at epoch {best_psnr['epoch']}")
        print()
        print(f" Best SSIM: {self.config_names[best_ssim['config']]}")
        print(f"   Value: {best_ssim['value']:.4f} at epoch {best_ssim['epoch']}")
        print()
        print(f"Most Stable: {self.config_names[most_stable['config']]}")
        print(f"   PSNR std: {most_stable['psnr_std']:.3f}, SSIM std: {most_stable['ssim_std']:.4f}")
        print()
        
        # Overall recommendation
        print("RECOMMENDATIONS:")
        if best_psnr['config'] == best_ssim['config']:
            print(f"• {self.config_names[best_psnr['config']]} performs best in both PSNR and SSIM")
        else:
            print(f"• For highest PSNR: Use {self.config_names[best_psnr['config']]}")
            print(f"• For highest SSIM: Use {self.config_names[best_ssim['config']]}")
        
        print(f"• For stable training: Consider {self.config_names[most_stable['config']]}")
        print("="*60)

# Usage example
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = CycleGANEvaluator()
    
    # Load training logs (update paths as needed)
    evaluator.load_training_logs(
        config_a_path="./training_log/Sar to EO RGB/training_log_config_a.csv",
        config_b_path="./training_log/Sar to EO SWIR NIR/training_log_config_b.csv", 
        config_c_path="./training_log/Sar to EO RGB NIR/training_log_config_c.csv"
    )
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print(" Starting CycleGAN Training Analysis...")
    print()
    
    # Generate all visualizations
    print("Plotting Loss Curves...")
    evaluator.plot_losses()
    
    print(" Plotting Validation Metrics...")
    evaluator.plot_validation_metrics()
    
    print("Plotting Learning Rate Schedules...")
    evaluator.plot_learning_rates()
    
    print(" Plotting Discriminator Scores...")
    evaluator.plot_discriminator_scores()
    
    print("Creating Performance Comparison Table...")
    comparison_table = evaluator.create_comparison_table()
    
    print(" Creating Comprehensive Dashboard...")
    evaluator.plot_comprehensive_dashboard()
    
    print(" Analyzing Best Configuration...")
    evaluator.analyze_best_configuration()
    
    print("Analysis Complete!")
