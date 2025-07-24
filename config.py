import torch

BASE_DIR="./data/" #Location of main data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #CPU or GPU
BATCH_SIZE = 24 #Number of images at once
LAMBDA_IDENTITY = 0.5   #Identity Loss
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 20 #Total Training Time
LOAD_MODEL = False
SAVE_MODEL = True
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 1e-4  # Slower discriminator learning
LAMBDA_SSIM = 0.1       # Increase SSIM weight
BETA1 = 0.5
BETA2 = 0.999
GRADIENT_CLIP = 1.0     # Reduce clipping
LABEL_SMOOTHING = 0.1   # Add label smoothing


EORGB_CHECKPOINT_GEN_SAR = "eorgb_gensar.pth.tar"
EORGB_CHECKPOINT_GEN_EO = "eorgb_geneo.pth.tar"
EORGB_CHECKPOINT_CRITIC_SAR = "eorgb_criticsar.pth.tar"
EORGB_CHECKPOINT_CRITIC_EO = "eorgb_criticeo.pth.tar"

EORGBNIR_CHECKPOINT_GEN_SAR = "eorgbnir_gensar.pth.tar"
EORGBNIR_CHECKPOINT_GEN_EO = "eorgbnir_geneo.pth.tar"
EORGBNIR_CHECKPOINT_CRITIC_SAR = "eorgbnir_criticsar.pth.tar"
EORGBNIR_CHECKPOINT_CRITIC_EO = "eorgbnir_criticeo.pth.tar"

EONIRSWIR_CHECKPOINT_GEN_SAR = "eonirswir_gensar.pth.tar"
EONIRSWIR_CHECKPOINT_GEN_EO = "eonirswir_geneo.pth.tar"
EONIRSWIR_CHECKPOINT_CRITIC_SAR = "eonirswir_criticsar.pth.tar"
EONIRSWIR_CHECKPOINT_CRITIC_EO = "eonirswir_criticeo.pth.tar"