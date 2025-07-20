import torch
import torch.nn as nn
from tqdm import tqdm
import model
import config
from torchvision.utils import save_image
from utils import load_checkpoint,save_checkpoint,seed_everything
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
import csv
import time


psnr = PeakSignalNoiseRatio().to(config.DEVICE)
ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE)

import csv
import os

class TrainingLogger:
    """
    A simple logger to save training and validation metrics to a CSV file for analysis.
    """
    def __init__(self, filepath="training_log.csv"):
        self.filepath = filepath
        self.file_exists = os.path.isfile(filepath)
        
        self.file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        
        if not self.file_exists:
            self.writer.writerow([
                "epoch",
                "learning_rate_gen",
                "learning_rate_disc", 
                "generator_loss",
                "discriminator_loss",
                "validation_psnr",
                "validation_ssim",
                "discriminator_eo_real_score",
                "discriminator_eo_fake_score"
            ])

    def log_epoch(self, epoch_data):
        """
        Writes a new row of data for a completed epoch.
        """
        self.writer.writerow([
            epoch_data.get("epoch", ""),
            epoch_data.get("learning_rate_gen", ""),
            epoch_data.get("learning_rate_disc", ""),
            epoch_data.get("generator_loss", ""),
            epoch_data.get("discriminator_loss", ""),
            epoch_data.get("validation_psnr", ""),
            epoch_data.get("validation_ssim", ""),
            epoch_data.get("discriminator_eo_real_score", ""),
            epoch_data.get("discriminator_eo_fake_score", "")
        ])
        self.file.flush()

    def close(self):
        self.file.close()




def validate_fn(model, epoch):
    """
    Performs validation at the end of an epoch.
    """
    print("\n--- Running Validation ---")
    
    gen_EO = model.gen_EO
    loader = model.val_dataloader

    gen_EO.eval()

    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for idx, (sar, eo) in enumerate(loader):
            sar = sar.to(config.DEVICE)
            eo = eo.to(config.DEVICE)

            fake_eo = gen_EO(sar)
            
            # Denormalize images from [-1, 1] to [0, 1] range for metric calculation
            fake_eo_norm = fake_eo * 0.5 + 0.5
            eo_norm = eo * 0.5 + 0.5
            sar_norm = sar * 0.5 + 0.5

            total_psnr += psnr(fake_eo_norm, eo_norm)
            total_ssim += ssim(fake_eo_norm, eo_norm)
            num_samples += 1

            # Save a grid of images for the first batch of each validation run
            if idx == 0:
                # Handle channel mismatch for visualization
                if sar.shape[1] != eo.shape[1]: 
                    if eo.shape[1] == 4:  # EO has 4 channels (RGB+NIR)
                        fake_eo_rgb = fake_eo_norm[:, :3]  # RGB only
                        eo_rgb = eo_norm[:, :3]           # RGB only
                        
                        comparison_grid = torch.cat([
                            sar_norm, 
                            fake_eo_rgb,
                            eo_rgb      
                        ], dim=0)
                        
                        nir_comparison = torch.cat([
                            torch.zeros_like(sar_norm[:, :1]).expand(-1, 3, -1, -1),  
                            fake_eo_norm[:, 3:4].expand(-1, 3, -1, -1),  
                            eo_norm[:, 3:4].expand(-1, 3, -1, -1)      
                        ], dim=0)
                        
                        save_image(comparison_grid, f"./saved_images/validation_epoch_{epoch}_rgb.png", nrow=len(sar))
                        save_image(nir_comparison, f"./saved_images/validation_epoch_{epoch}_nir.png", nrow=len(sar))
                        
                else:  
                    comparison_grid = torch.cat([sar_norm, fake_eo_norm, eo_norm], dim=0)
                    save_image(comparison_grid, f"./saved_images/validation_epoch_{epoch}.png", nrow=len(sar))
                print(f"Saved validation image grid to saved_images/validation_epoch_{epoch}.png")

    if num_samples == 0:
        print("Warning: Validation set is empty. Skipping metric calculation for this epoch.")
        # Set models back to train mode before exiting
        gen_EO.train()
        return {
        "validation_psnr": -1,
        "validation_ssim": -1
    }
    # Calculate average metrics for the epoch
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print(f"Validation Metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    print("--- Validation Complete ---\n")

    # Set the generator back to training mode
    gen_EO.train()
    print(f"validation_psnr: {avg_psnr.item()},validation_ssim: {avg_ssim.item()}")
    return {
        "validation_psnr": avg_psnr.item(),
        "validation_ssim": avg_ssim.item()
    }

def train(cycleGAN: model.Model):
    disc_SAR, disc_EO = cycleGAN.disc_SAR, cycleGAN.disc_EO
    gen_SAR, gen_EO = cycleGAN.gen_SAR, cycleGAN.gen_EO
    opt_disc, opt_gen = cycleGAN.opt_disc, cycleGAN.opt_gen
    d_scaler, g_scaler = cycleGAN.d_scaler, cycleGAN.g_scaler
    l1, mse = cycleGAN.L1, cycleGAN.mse
    loader = cycleGAN.train_dataloader
    ssim_loss_fn = model.SSIMLoss().to(config.DEVICE)

    eo_real, eo_fake = 0.0, 0.0
    total_g_loss, total_d_loss = 0.0, 0.0
    loop = tqdm(loader, leave=False)

    for idx, (sar, eo) in enumerate(loop):
        sar, eo = sar.to(config.DEVICE), eo.to(config.DEVICE)
        
        # Train Discriminators
        opt_disc.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # Generate fake images
            fake_eo = gen_EO(sar).detach() 
            fake_sar = gen_SAR(eo).detach()
            
            real_label = torch.ones_like(torch.empty(1)).to(config.DEVICE) * (1.0 - config.LABEL_SMOOTHING)
            fake_label = torch.zeros_like(torch.empty(1)).to(config.DEVICE) + config.LABEL_SMOOTHING
            
            D_EO_real = disc_EO(eo)
            D_EO_fake = disc_EO(fake_eo)
            D_EO_loss = (mse(D_EO_real, real_label.expand_as(D_EO_real)) + 
                        mse(D_EO_fake, fake_label.expand_as(D_EO_fake))) * 0.5
            
            D_SAR_real = disc_SAR(sar)
            D_SAR_fake = disc_SAR(fake_sar)
            D_SAR_loss = (mse(D_SAR_real, real_label.expand_as(D_SAR_real)) + 
                         mse(D_SAR_fake, fake_label.expand_as(D_SAR_fake))) * 0.5
            
            D_loss = D_EO_loss + D_SAR_loss
        
        # Update discriminators
        d_scaler.scale(D_loss).backward()
        d_scaler.unscale_(opt_disc)
        torch.nn.utils.clip_grad_norm_(
            list(disc_SAR.parameters()) + list(disc_EO.parameters()), 
            max_norm=config.GRADIENT_CLIP
        )
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        # Train Generators  
        opt_gen.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            fake_eo = gen_EO(sar)
            fake_sar = gen_SAR(eo)
            
            # Adversarial losses
            D_EO_fake_gen = disc_EO(fake_eo)
            D_SAR_fake_gen = disc_SAR(fake_sar)
            
            real_label = torch.ones_like(D_EO_fake_gen).to(config.DEVICE)
            loss_G_EO = mse(D_EO_fake_gen, real_label)
            loss_G_SAR = mse(D_SAR_fake_gen, torch.ones_like(D_SAR_fake_gen))
            
            # Cycle consistency losses
            cycle_sar = gen_SAR(fake_eo)
            cycle_eo = gen_EO(fake_sar)
            cycle_loss = l1(sar, cycle_sar) + l1(eo, cycle_eo)
            
            # Identity losses
            if sar.shape[1] == eo.shape[1]:  # Only if channels match
                identity_sar = gen_SAR(sar)  
                identity_eo = gen_EO(eo)     
                identity_loss = l1(sar, identity_sar) + l1(eo, identity_eo)
            else:
                identity_loss = torch.tensor(0.0, device=sar.device)
            
            # SSIM loss for perceptual quality
            ssim_loss = ssim_loss_fn(fake_eo * 0.5 + 0.5, eo * 0.5 + 0.5)
            
            # Total generator loss
            G_loss = (loss_G_EO + loss_G_SAR + 
                     cycle_loss * config.LAMBDA_CYCLE + 
                     identity_loss * config.LAMBDA_IDENTITY +
                     ssim_loss * config.LAMBDA_SSIM)
        
        # Update generators
        g_scaler.scale(G_loss).backward()
        g_scaler.unscale_(opt_gen)
        torch.nn.utils.clip_grad_norm_(
            list(gen_SAR.parameters()) + list(gen_EO.parameters()), 
            max_norm=config.GRADIENT_CLIP
        )
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Track metrics
        eo_real += D_EO_real.mean().item()
        eo_fake += D_EO_fake.mean().item()
        total_g_loss += G_loss.item()
        total_d_loss += D_loss.item()
        
        loop.set_postfix(
            EO_real=eo_real/(idx+1), 
            EO_fake=eo_fake/(idx+1),
            G_loss=total_g_loss/(idx+1),
            D_loss=total_d_loss/(idx+1)
        )

    # Return fixed variable names
    avg_g_loss = total_g_loss / len(loader)
    avg_d_loss = total_d_loss / len(loader)
    
    return {
        "generator_loss": avg_g_loss,
        "discriminator_loss": avg_d_loss,
        "discriminator_eo_real_score": eo_real / len(loader),
        "discriminator_eo_fake_score": eo_fake / len(loader)
    }

    
    

def main():
    seed_everything() 
    cycleGAN = model.get_model(model.ModelType.SAR_TO_EORGB)
    best_val_metric = -float('inf')
    best_epoch = 0
    patience_counter = 0
    MIN_IMPROVEMENT = 0.01  # Minimum improvement to consider as better

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        
        # Wrap each model component in nn.DataParallel
        cycleGAN.gen_SAR = nn.DataParallel(cycleGAN.gen_SAR)
        cycleGAN.gen_EO = nn.DataParallel(cycleGAN.gen_EO)
        cycleGAN.disc_SAR = nn.DataParallel(cycleGAN.disc_SAR)
        cycleGAN.disc_EO = nn.DataParallel(cycleGAN.disc_EO)

    logger = TrainingLogger("training_log_config_a.csv")

    for epoch in range(config.NUM_EPOCHS):
        t = time.time()
        print(f"Epoch: {epoch+1}")
        train_metrics = train(cycleGAN)
        validation_metrics = validate_fn(cycleGAN, epoch)
        print(f"Time per epoch: {time.time()-t}")

        current_lr_gen = cycleGAN.opt_gen.param_groups[0]['lr']
        current_lr_disc = cycleGAN.opt_disc.param_groups[0]['lr']
        print(f"learning_rate_gen: {current_lr_gen}\nlearning_rate_disc: {current_lr_disc}")
        epoch_log_data = {
            "epoch": epoch + 1,
            "learning_rate_gen": current_lr_gen, 
            "learning_rate_disc": current_lr_disc,
            **train_metrics,
            **validation_metrics
        }

        cycleGAN.scheduler_gen.step()
        cycleGAN.scheduler_disc.step()

        logger.log_epoch(epoch_log_data)
    
        val_metric_to_monitor = validation_metrics.get('validation_psnr', None)
        
        if val_metric_to_monitor is not None:
            improvement = val_metric_to_monitor - best_val_metric
            
            if improvement > MIN_IMPROVEMENT:
                best_val_metric = val_metric_to_monitor
                best_epoch = epoch + 1
                patience_counter = 0
                
                print(f"New best validation PSNR: {best_val_metric:.4f} (improvement: +{improvement:.4f})")
                print("Saving checkpoints...")
                
                save_checkpoint(cycleGAN.gen_SAR, cycleGAN.opt_gen, filename=config.EORGB_CHECKPOINT_GEN_SAR)
                save_checkpoint(cycleGAN.gen_EO, cycleGAN.opt_gen, filename=config.EORGB_CHECKPOINT_GEN_EO)
                save_checkpoint(cycleGAN.disc_SAR, cycleGAN.opt_disc, filename=config.EORGB_CHECKPOINT_CRITIC_SAR)
                save_checkpoint(cycleGAN.disc_EO, cycleGAN.opt_disc, filename=config.EORGB_CHECKPOINT_CRITIC_EO)
            else:
                patience_counter += 1
                print(f"Current PSNR: {val_metric_to_monitor:.4f} (Best: {best_val_metric:.4f} at epoch {best_epoch})")

if __name__=='__main__':
    main()