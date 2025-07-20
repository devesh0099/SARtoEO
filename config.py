import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_SAR = "gensar.pth.tar"
CHECKPOINT_GEN_EO = "geneo.pth.tar"
CHECKPOINT_CRITIC_SAR = "criticsar.pth.tar"
CHECKPOINT_CRITIC_EO = "criticeo.pth.tar"