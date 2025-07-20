import torch
import torch.nn as nn
import torch.optim as optim

from enum import Enum
from preprocess import get_dataloaders
import config

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.nn.utils import spectral_norm

import numpy as np

class ModelType(Enum):
    SAR_TO_EORGB =1
    SAR_TO_EONIRSWIR = 2
    SAR_TO_EORGBNIR = 3

class Model():
    """
    A Model class for handling model in the system
    """
    def __init__(self,
        disc_SAR,
        disc_EO,
        gen_EO ,
        gen_SAR,
        opt_disc, 
        opt_gen,    
        scheduler_gen,  
        scheduler_disc,  
        train_dataloader,
        val_dataloader,
        g_scaler = torch.amp.GradScaler('cuda'),
        d_scaler = torch.amp.GradScaler('cuda'),
        L1 = nn.L1Loss(),
        mse = nn.MSELoss(),
        ):
        self.disc_SAR = disc_SAR
        self.disc_EO = disc_EO
        self.gen_EO  = gen_EO
        self.gen_SAR = gen_SAR
        self.opt_disc  = opt_disc
        self.opt_gen = opt_gen
        self.scheduler_gen = scheduler_gen
        self.scheduler_disc = scheduler_disc
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.g_scaler = g_scaler
        self.d_scaler = d_scaler
        self.L1 = L1
        self.mse = mse


class SSIMLoss(nn.Module):
    # Converted the metrics in a loss so the model get incentive to improve
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, img1, img2):
        self.ssim_metric.to(img1.device)
        return 1 - self.ssim_metric(img1, img2)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.block(x) # Skip connection



class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, 4, padding=1))
        )

    def forward(self, img):
        return self.model(img)

def get_model(model:ModelType = ModelType.SAR_TO_EORGB):
    match model:
        case ModelType.SAR_TO_EORGB:
            sar_img_channels = 3
            eo_img_channels=3
        case ModelType.SAR_TO_EONIRSWIR:
            sar_img_channels = 3
            eo_img_channels=3
        case ModelType.SAR_TO_EORGBNIR:
            sar_img_channels=3
            eo_img_channels=4

    disc_SAR = Discriminator(in_channels=sar_img_channels).to(config.DEVICE)
    disc_EO = Discriminator(in_channels=eo_img_channels).to(config.DEVICE)
    gen_EO = Generator(in_channels=sar_img_channels, out_channels=eo_img_channels, num_residuals=6).to(config.DEVICE)
    gen_SAR = Generator(in_channels=eo_img_channels, out_channels=sar_img_channels, num_residuals=6).to(config.DEVICE)
    
    opt_disc = optim.Adam(
        list(disc_SAR.parameters()) + list(disc_EO.parameters()),
        lr=config.LEARNING_RATE_D,
        betas=(config.BETA1, config.BETA2),
    )

    opt_gen = optim.Adam(
        list(gen_EO.parameters()) + list(gen_SAR.parameters()),
        lr=config.LEARNING_RATE_G,
        betas=(config.BETA1, config.BETA2),
    )

    def cosine_schedule(epoch):
        return 0.5 * (1 + np.cos(np.pi * epoch / config.NUM_EPOCHS))
    
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=cosine_schedule)
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=cosine_schedule)


    train_loader, val_loader = get_dataloaders(config.BASE_DIR,config.BATCH_SIZE,val_split=0.25) 
    
    match model:
        case ModelType.SAR_TO_EORGB:
            return Model(
                disc_SAR=disc_SAR,
                disc_EO=disc_EO,
                gen_EO=gen_EO,
                gen_SAR=gen_SAR,
                opt_disc=opt_disc,
                opt_gen=opt_gen,
                scheduler_gen=scheduler_gen,    
                scheduler_disc=scheduler_disc, 
                train_dataloader=train_loader['a'],
                val_dataloader=val_loader['a']
            )
        case ModelType.SAR_TO_EONIRSWIR:
            return Model(
                disc_SAR=disc_SAR,
                disc_EO=disc_EO,
                gen_EO=gen_EO,
                gen_SAR=gen_SAR,
                opt_disc=opt_disc,
                opt_gen=opt_gen,
                scheduler_gen=scheduler_gen,    
                scheduler_disc=scheduler_disc, 
                train_dataloader=train_loader['b'],
                val_dataloader=val_loader['b']
            )
        case ModelType.SAR_TO_EORGBNIR:
            return Model(
                disc_SAR=disc_SAR,
                disc_EO=disc_EO,
                gen_EO=gen_EO,
                gen_SAR=gen_SAR,
                opt_disc=opt_disc,
                opt_gen=opt_gen,
                scheduler_gen=scheduler_gen,    
                scheduler_disc=scheduler_disc, 
                train_dataloader=train_loader['c'],
                val_dataloader=val_loader['c']
            )

    # if config.SAVE_MODEL:
    #     match model:
    #         case ModelType.SAR_TO_EORGB:
    #             save_checkpoint(gen_SAR, opt_gen, filename=config.EORGB_CHECKPOINT_GEN_SAR)
    #             save_checkpoint(gen_EO, opt_gen, filename=config.EORGB_CHECKPOINT_GEN_EO)
    #             save_checkpoint(disc_SAR, opt_disc, filename=config.EORGB_CHECKPOINT_CRITIC_SAR)
    #             save_checkpoint(disc_EO, opt_disc, filename=config.EORGB_CHECKPOINT_CRITIC_EO)
    #         case ModelType.SAR_TO_EORGBNIR:
    #             save_checkpoint(gen_SAR, opt_gen, filename=config.EORGBNIR_CHECKPOINT_GEN_SAR)
    #             save_checkpoint(gen_EO, opt_gen, filename=config.EORGBNIR_CHECKPOINT_GEN_EO)
    #             save_checkpoint(disc_SAR, opt_disc, filename=config.EORGBNIR_CHECKPOINT_CRITIC_SAR)
    #             save_checkpoint(disc_EO, opt_disc, filename=config.EORGBNIR_CHECKPOINT_CRITIC_EO)
    #         case ModelType.SAR_TO_EONIRSWIR:
    #             save_checkpoint(gen_SAR, opt_gen, filename=config.EONIRSWIR_CHECKPOINT_GEN_SAR)
    #             save_checkpoint(gen_EO, opt_gen, filename=config.EONIRSWIR_CHECKPOINT_GEN_EO)
    #             save_checkpoint(disc_SAR, opt_disc, filename=config.EONIRSWIR_CHECKPOINT_CRITIC_SAR)
    #             save_checkpoint(disc_EO, opt_disc, filename=config.EONIRSWIR_CHECKPOINT_CRITIC_EO)
    

    # if config.LOAD_MODEL:
    #     match model:
    #         case ModelType.SAR_TO_EORGB:
    #             load_checkpoint(
    #                 config.EORGB_CHECKPOINT_GEN_SAR,
    #                 gen_SAR,
    #                 opt_gen,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EORGB_CHECKPOINT_GEN_EO,
    #                 gen_EO,
    #                 opt_gen,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EORGB_CHECKPOINT_CRITIC_SAR,
    #                 disc_SAR,
    #                 opt_disc,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EORGB_CHECKPOINT_CRITIC_EO,
    #                 disc_EO,
    #                 opt_disc,
    #                 config.LEARNING_RATE,
    #             )
    #         case ModelType.SAR_TO_EORGBNIR:
    #             load_checkpoint(
    #                 config.EORGBNIR_CHECKPOINT_GEN_SAR,
    #                 gen_SAR,
    #                 opt_gen,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EORGBNIR_CHECKPOINT_GEN_EO,
    #                 gen_EO,
    #                 opt_gen,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EORGBNIR_CHECKPOINT_CRITIC_SAR,
    #                 disc_SAR,
    #                 opt_disc,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EORGBNIR_CHECKPOINT_CRITIC_EO,
    #                 disc_EO,
    #                 opt_disc,
    #                 config.LEARNING_RATE,
    #             )
    #         case ModelType.SAR_TO_EONIRSWIR:
    #             load_checkpoint(
    #                 config.EONIRSWIR_CHECKPOINT_GEN_SAR,
    #                 gen_SAR,
    #                 opt_gen,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EONIRSWIR_CHECKPOINT_GEN_EO,
    #                 gen_EO,
    #                 opt_gen,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EONIRSWIR_CHECKPOINT_CRITIC_SAR,
    #                 disc_SAR,
    #                 opt_disc,
    #                 config.LEARNING_RATE,
    #             )
    #             load_checkpoint(
    #                 config.EONIRSWIR_CHECKPOINT_CRITIC_EO,
    #                 disc_EO,
    #                 opt_disc,
    #                 config.LEARNING_RATE,
    #             )
