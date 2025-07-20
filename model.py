import torch
import torch.nn as nn
import torch.optim as optim

from enum import Enum
from preprocess import get_dataloaders
import config

class ModelType(Enum):
    SAR_TO_EORGB =1
    SAR_TO_EONIRSWIR = 2
    SAR_TO_EORGBNIR = 3

class Model():
    def __init__(self,
        disc_SAR,
        disc_EO,
        gen_EO ,
        gen_SAR,
        opt_disc, 
        opt_gen,    
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
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.g_scaler = g_scaler
        self.d_scaler = d_scaler
        self.L1 = L1
        self.mse = mse

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
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


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
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
    


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
    gen_EO = Generator(in_channels=sar_img_channels, out_channels=eo_img_channels, num_residuals=9).to(config.DEVICE)
    gen_SAR = Generator(in_channels=eo_img_channels, out_channels=sar_img_channels, num_residuals=9).to(config.DEVICE)
    
    opt_disc = optim.Adam(
        list(disc_SAR.parameters()) + list(disc_EO.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_EO.parameters()) + list(gen_SAR.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    train_loader, val_loader = get_dataloaders(config.BASE_DIR) 
    
    match model:
        case ModelType.SAR_TO_EORGB:
            return Model(
                disc_SAR=disc_SAR,
                disc_EO=disc_EO,
                gen_EO=gen_EO,
                gen_SAR=gen_SAR,
                opt_disc=opt_disc,
                opt_gen=opt_gen,
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
                train_dataloader=train_loader['c'],
                val_dataloader=val_loader['c']
            )

    # for epoch in range(config.NUM_EPOCHS):
    #     train_fn(
    #         disc_H,
    #         disc_Z,
    #         gen_Z,
    #         gen_H,
    #         loader,
    #         opt_disc,
    #         opt_gen,
    #         L1,
    #         mse,
    #         d_scaler,
    #         g_scaler,
    #     )

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
