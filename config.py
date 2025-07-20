import torch

BASE_DIR="./data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True

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