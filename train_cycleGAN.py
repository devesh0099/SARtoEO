import torch
import torch.nn as nn
from tqdm import tqdm
import model
import config
from torchvision.utils import save_image
from utils import load_checkpoint,save_checkpoint
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
        """
        Initializes the logger and creates the CSV file with a header.
        
        Args:
            filepath (str): The path to the log file.
        """
        self.filepath = filepath
        self.file_exists = os.path.isfile(filepath)
        
        # Open the file in append mode, which creates it if it doesn't exist
        self.file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        
        # Write the header row only if the file is new
        if not self.file_exists:
            self.writer.writerow([
                "epoch",
                "learning_rate", 
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
        
        Args:
            epoch_data (dict): A dictionary containing all metrics for the epoch.
        """
        self.writer.writerow([
            epoch_data.get("epoch", ""),
            epoch_data.get("learning_rate", ""),
            epoch_data.get("generator_loss", ""),
            epoch_data.get("discriminator_loss", ""),
            epoch_data.get("validation_psnr", ""),
            epoch_data.get("validation_ssim", ""),
            epoch_data.get("discriminator_eo_real_score", ""),
            epoch_data.get("discriminator_eo_fake_score", "")
        ])
        # Flush the writer's buffer to ensure data is written to disk immediately
        self.file.flush()

    def close(self):
        """
        Closes the log file.
        """
        self.file.close()




def validate_fn(model, epoch):
    """
    Performs validation at the end of an epoch.

    Args:
        model: An instance of the Model class.
        epoch (int): The current epoch number, used for naming saved images.
    """
    print("\n--- Running Validation ---")
    
    # Unpack the necessary components from the model
    gen_EO = model.gen_EO
    loader = model.val_dataloader

    # Set models to evaluation mode
    gen_EO.eval()

    # Placeholders for accumulating metric scores
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    # Use torch.no_grad() for efficiency as we don't need to calculate gradients
    with torch.no_grad():
        # Loop through the validation data
        for idx, (sar, eo) in enumerate(loader):
            sar = sar.to(config.DEVICE)
            eo = eo.to(config.DEVICE)

            # Generate fake EO images from the SAR input
            fake_eo = gen_EO(sar)
            
            # --- Quantitative Validation: Calculate Metrics ---
            # Denormalize images from [-1, 1] to [0, 1] range for metric calculation
            fake_eo_norm = fake_eo * 0.5 + 0.5
            eo_norm = eo * 0.5 + 0.5
            
            total_psnr += psnr(fake_eo_norm, eo_norm)
            total_ssim += ssim(fake_eo_norm, eo_norm)
            num_samples += 1

            # --- Qualitative Validation: Save Sample Images ---
            # Save a grid of images for the first batch of each validation run
            if idx == 0:
                # We save the SAR input, the generated EO, and the real EO for comparison
                comparison_grid = torch.cat([sar * 0.5 + 0.5, fake_eo_norm, eo_norm], dim=0)
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


def train(cycleGAN:model.Model):
    disc_SAR = cycleGAN.disc_SAR
    disc_EO = cycleGAN.disc_EO
    gen_SAR = cycleGAN.gen_SAR
    gen_EO = cycleGAN.gen_EO
    opt_disc = cycleGAN.opt_disc
    opt_gen = cycleGAN.opt_gen
    l1 = cycleGAN.L1
    mse = cycleGAN.mse
    d_scaler = cycleGAN.d_scaler
    g_scaler = cycleGAN.g_scaler
    loader = cycleGAN.train_dataloader
    loop = tqdm(loader,leave=False)

    ssim_loss_fn = model.SIMLoss()

    EO_reals = 0
    EO_fakes = 0

    total_g_loss = 0
    total_d_loss = 0

    for idx, (sar, eo) in enumerate(loop):
        sar = sar.to(config.DEVICE)
        eo = eo.to(config.DEVICE)

        # Train Discriminators
        with torch.amp.autocast('cuda'):
            # Train EO Discriminator
            fake_eo = gen_EO(sar)
            D_EO_real = disc_EO(eo)
            D_EO_fake = disc_EO(fake_eo.detach())
            EO_reals += D_EO_real.mean().item()
            EO_fakes += D_EO_fake.mean().item()
            D_EO_real_loss = mse(D_EO_real, torch.ones_like(D_EO_real))
            D_EO_fake_loss = mse(D_EO_fake, torch.zeros_like(D_EO_fake))
            D_EO_loss = D_EO_real_loss + D_EO_fake_loss

            # Train SAR Discriminator
            fake_sar = gen_SAR(eo)
            D_SAR_real = disc_SAR(sar)
            D_SAR_fake = disc_SAR(fake_sar.detach())
            D_SAR_real_loss = mse(D_SAR_real, torch.ones_like(D_SAR_real))
            D_SAR_fake_loss = mse(D_SAR_fake, torch.zeros_like(D_SAR_fake))
            D_SAR_loss = D_SAR_real_loss + D_SAR_fake_loss

            # Combine discriminator losses
            D_loss = (D_EO_loss + D_SAR_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.amp.autocast('cuda'):
            # Adversarial loss
            D_EO_fake = disc_EO(fake_eo)
            D_SAR_fake = disc_SAR(fake_sar)
            loss_G_EO = mse(D_EO_fake, torch.ones_like(D_EO_fake))
            loss_G_SAR = mse(D_SAR_fake, torch.ones_like(D_SAR_fake))

            # Cycle-consistency loss
            cycle_sar = gen_SAR(fake_eo)
            cycle_eo = gen_EO(fake_sar)
            cycle_sar_loss = l1(sar, cycle_sar)
            cycle_eo_loss = l1(eo, cycle_eo)

            # Identity loss
            identity_sar = gen_SAR(sar)
            identity_eo = gen_EO(eo)
            identity_sar_loss = l1(sar, identity_sar)
            identity_eo_loss = l1(eo, identity_eo)

            # --- NEW: Calculate SSIM Loss ---
            # Denormalize images to the [0, 1] range for SSIM calculation
            fake_eo_norm = fake_eo * 0.5 + 0.5
            eo_norm = eo * 0.5 + 0.5
            ssim_loss = ssim_loss_fn(fake_eo_norm, eo_norm)

            # Combine all generator losses
            G_loss = (
                loss_G_SAR
                + loss_G_EO
                + cycle_sar_loss * config.LAMBDA_CYCLE
                + cycle_eo_loss * config.LAMBDA_CYCLE
                + identity_eo_loss * config.LAMBDA_IDENTITY
                + identity_sar_loss * config.LAMBDA_IDENTITY
                + ssim_loss * config.LAMBDA_SSIM  # Add the weighted SSIM loss
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            # Denormalize from [-1, 1] to [0, 1] for saving
            save_image(fake_eo * 0.5 + 0.5, f"./saved_images/eo_{idx}.png")
            save_image(fake_sar * 0.5 + 0.5, f"./saved_images/sar_{idx}.png")
        total_g_loss += G_loss.item()
        total_d_loss += D_loss.item()
        # Update the progress bar with new labels
        loop.set_postfix(EO_real=EO_reals / (idx + 1), EO_fake=EO_fakes / (idx + 1))
    
    avg_g_loss = total_g_loss / len(loader)
    avg_d_loss = total_d_loss / len(loader)
    avg_eo_real_score = EO_reals / len(loader)
    avg_eo_fake_score = EO_fakes / len(loader)
    
    print(f"generator_loss: {avg_g_loss},\ndiscriminator_loss: {avg_d_loss},\ndiscriminator_eo_real_score: {avg_eo_real_score},\ndiscriminator_eo_fake_score: {avg_eo_fake_score}")
    return {
        "generator_loss": avg_g_loss,
        "discriminator_loss": avg_d_loss,
        "discriminator_eo_real_score": avg_eo_real_score,
        "discriminator_eo_fake_score": avg_eo_fake_score
    }
    

    
    

def main():
    cycleGAN = model.get_model(model.ModelType.SAR_TO_EORGB)

# =========For testing==============
    # epocs =[]
    # generator_loss=[]
    # discriminator_loss=[]
    # discriminator_eo_real_score = []
    # discriminator_eo_fake_score = []
    # validation_psnr =[]
    # validation_ssim=[]
# =========For testing==============

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

        current_lr = cycleGAN.opt_gen.param_groups[0]['lr']

        epoch_log_data = {
            "epoch": epoch + 1,
            "learning_rate": current_lr, # Add the learning rate here
            **train_metrics,
            **validation_metrics
        }

        cycleGAN.scheduler_gen.step()
        cycleGAN.scheduler_disc.step()
# =================================For testing=================================
        # epocs.append(epoch_log_data['epoch'])
        # generator_loss.append(epoch_log_data['generator_loss'])
        # discriminator_loss.append(epoch_log_data['discriminator_loss'])
        # discriminator_eo_real_score.append(epoch_log_data['discriminator_eo_real_score'])
        # discriminator_eo_fake_score.append(epoch_log_data['discriminator_eo_fake_score'])
        # validation_psnr.append(epoch_log_data['validation_psnr'])
        # validation_ssim.append(epoch_log_data['validation_ssim'])
# =================================For testing=================================

        logger.log_epoch(epoch_log_data)
    
        save_checkpoint(cycleGAN.gen_SAR, cycleGAN.opt_gen, filename=config.EORGB_CHECKPOINT_GEN_SAR)
        save_checkpoint(cycleGAN.gen_EO, cycleGAN.opt_gen, filename=config.EORGB_CHECKPOINT_GEN_EO)
        save_checkpoint(cycleGAN.disc_SAR, cycleGAN.opt_disc, filename=config.EORGB_CHECKPOINT_CRITIC_SAR)
        save_checkpoint(cycleGAN.disc_EO, cycleGAN.opt_disc, filename=config.EORGB_CHECKPOINT_CRITIC_EO)


if __name__=='__main__':
    main()