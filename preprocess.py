from sen12ms_dataLoader import SEN12MSDataset, S1Bands, S2Bands, Seasons
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict
from scipy.ndimage import uniform_filter, variance


def lee_filter(img, size): 
    """
    Applies Speckle Filter on the Image
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

class SARDataset(Dataset):
    """
    Dataset class for loading and preprocessing Sentinel-1 (SAR) images.
    """
    def __init__(self, sen12ms_instance: SEN12MSDataset, season: Seasons, scene_patch_ids: List[Tuple[int, int]]):
        self.sen12ms = sen12ms_instance
        self.season = season
        self.scene_patch_ids = scene_patch_ids

    def __len__(self) -> int:
        return len(self.scene_patch_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        scene_id, patch_id = self.scene_patch_ids[idx]
        
        bands_to_load = [S1Bands.VV, S1Bands.VH]
        sar_image_raw, _ = self.sen12ms.get_patch(self.season, scene_id, patch_id, bands=bands_to_load)
        
        vv_channel_filtered = lee_filter(sar_image_raw[0, :, :], size=7)
        vh_channel_filtered = lee_filter(sar_image_raw[1, :, :], size=7)
        
        # Calculating ratio of third channel
        ratio_channel = vh_channel_filtered / (vv_channel_filtered + 1e-6)

        # Normalize each of the three channels independently
        # Normalize VV and VH channels to [-1, 1]
        vv_norm = (vv_channel_filtered / 2047.5) - 1.0
        vh_norm = (vh_channel_filtered / 2047.5) - 1.0
        
        # Independently normalize the ratio channel to [-1, 1]
        min_val, max_val = np.min(ratio_channel), np.max(ratio_channel)
        if max_val > min_val:
            ratio_norm = ((ratio_channel - min_val) / (max_val - min_val)) * 2.0 - 1.0
        else:
            # Handle the case where the channel is flat (all values are the same)
            ratio_norm = np.zeros_like(ratio_channel)
        
        sar_image_processed = np.stack([vv_norm, vh_norm, ratio_norm], axis=0)
        sar_tensor = torch.from_numpy(sar_image_processed.astype(np.float32))        
        return sar_tensor
        
class EORGBDataset(Dataset):
    """
    Dataset class for loading and preprocessing Sentinel-2 (EO) images with RGB bands.
    """
    def __init__(self, sen12ms_instance: SEN12MSDataset, season: Seasons, scene_patch_ids: List[Tuple[int, int]]):
        self.sen12ms = sen12ms_instance
        self.season = season
        self.scene_patch_ids = scene_patch_ids

    def __len__(self) -> int:
        return len(self.scene_patch_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        scene_id, patch_id = self.scene_patch_ids[idx]        
        bands_to_load = [S2Bands.B04, S2Bands.B03, S2Bands.B02]
        eo_image, _ = self.sen12ms.get_patch(self.season, scene_id, patch_id, bands=bands_to_load)
        
        # Preprocessing
        eo_tensor = torch.from_numpy(eo_image.astype(np.float32))
        eo_tensor = (eo_tensor / 2047.5) - 1.0
        
        return eo_tensor

class EONirSwirRedEdgeDataset(Dataset):
    """
    Dataset class for loading and preprocessing Sentinel-2 (EO) images with NIR, SWIR, and Red Edge bands.
    """
    def __init__(self, sen12ms_instance: SEN12MSDataset, season: Seasons, scene_patch_ids: List[Tuple[int, int]]):
        self.sen12ms = sen12ms_instance
        self.season = season
        self.scene_patch_ids = scene_patch_ids

    def __len__(self) -> int:
        return len(self.scene_patch_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        scene_id, patch_id = self.scene_patch_ids[idx]
        bands = [S2Bands.B08, S2Bands.B11, S2Bands.B05]
        eo_image, _ = self.sen12ms.get_patch(self.season, scene_id, patch_id, bands=bands)
        
        # Preprocessing
        eo_tensor = torch.from_numpy(eo_image.astype(np.float32))
        eo_tensor = (eo_tensor / 2047.5) - 1.0
        
        return eo_tensor

class EORGBNirDataset(Dataset):
    """
    Dataset class for loading and preprocessing Sentinel-2 (EO) images with RGB + NIR bands.
    """
    def __init__(self, sen12ms_instance: SEN12MSDataset, season: Seasons, scene_patch_ids: List[Tuple[int, int]]):
        self.sen12ms = sen12ms_instance
        self.season = season
        self.scene_patch_ids = scene_patch_ids

    def __len__(self) -> int:
        return len(self.scene_patch_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        scene_id, patch_id = self.scene_patch_ids[idx]
        bands_to_load = [S2Bands.B04, S2Bands.B03, S2Bands.B02, S2Bands.B08]
        eo_image, _ = self.sen12ms.get_patch(self.season, scene_id, patch_id, bands=bands_to_load)
        
        # Preprocessing
        eo_tensor = torch.from_numpy(eo_image.astype(np.float32))
        eo_tensor = (eo_tensor / 2047.5) - 1.0
        
        return eo_tensor

class PairedSarEoDataset(Dataset):
    """
    Helper dataset to pair SAR and EO images for CycleGAN training.
    Applies optional, identical random augmentations to the pair.
    """
    def __init__(self, sar_dataset: Dataset, eo_dataset: Dataset, augment: bool = False):
        self.sar_dataset = sar_dataset
        self.eo_dataset = eo_dataset
        self.augment = augment
        assert len(self.sar_dataset) == len(self.eo_dataset), "SAR and EO datasets must have the same length."

    def __len__(self) -> int:
        return len(self.sar_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sar_img = self.sar_dataset[idx]
        eo_img = self.eo_dataset[idx]

        if self.augment:
            # Apply random horizontal flip
            if random.random() > 0.5:
                sar_img = torch.flip(sar_img, dims=[2])
                eo_img = torch.flip(eo_img, dims=[2])
            
            # Apply random vertical flip
            if random.random() > 0.5:
                sar_img = torch.flip(sar_img, dims=[1])
                eo_img = torch.flip(eo_img, dims=[1])

        return sar_img, eo_img


def get_dataloaders(base_dir: str, batch_size: int, val_split: float = 0.15, season: Seasons = Seasons.WINTER) -> Tuple[Dict, Dict]:
    """
    Prepares and returns training and validation dataloaders for all three configurations.
    
    Args:
        base_dir: The root directory of the SEN12MS dataset.
        batch_size: The batch size for the dataloaders.
        val_split: The fraction of scenes to use for validation.
        season: The season to load data from.

    Returns:
        A tuple containing two dictionaries: (train_dataloaders, val_dataloaders).
        Each dictionary has keys 'a', 'b', 'c' for the three configurations.
    """
    sen12ms = SEN12MSDataset(base_dir)
    all_scene_ids = list(sen12ms.get_scene_ids(season))
    random.shuffle(all_scene_ids)

    # Split scenes for training and validation
    split_idx = int(len(all_scene_ids) * val_split)
    val_scenes = all_scene_ids[:split_idx]
    train_scenes = all_scene_ids[split_idx:]

    # Get all patch IDs for the selected scenes
    all_patches = sen12ms.get_season_ids(season)
    train_scene_patch_ids = [(s, p) for s in train_scenes for p in all_patches.get(s, [])]
    val_scene_patch_ids = [(s, p) for s in val_scenes for p in all_patches.get(s, [])]

    # --- Create Training Datasets ---
    sar_train_ds = SARDataset(sen12ms, season, train_scene_patch_ids)
    eo_rgb_train_ds = EORGBDataset(sen12ms, season, train_scene_patch_ids)
    eo_nir_train_ds = EONirSwirRedEdgeDataset(sen12ms, season, train_scene_patch_ids)
    eo_rgbnir_train_ds = EORGBNirDataset(sen12ms, season, train_scene_patch_ids)
    
    # --- Create Validation Datasets ---
    sar_val_ds = SARDataset(sen12ms, season, val_scene_patch_ids)
    eo_rgb_val_ds = EORGBDataset(sen12ms, season, val_scene_patch_ids)
    eo_nir_val_ds = EONirSwirRedEdgeDataset(sen12ms, season, val_scene_patch_ids)
    eo_rgbnir_val_ds = EORGBNirDataset(sen12ms, season, val_scene_patch_ids)

    # --- Create Paired Datasets and DataLoaders ---
    train_dataloaders = {
        'a': DataLoader(PairedSarEoDataset(sar_train_ds, eo_rgb_train_ds, augment=True), batch_size=batch_size, shuffle=True, num_workers=4),
        'b': DataLoader(PairedSarEoDataset(sar_train_ds, eo_nir_train_ds, augment=True), batch_size=batch_size, shuffle=True, num_workers=4),
        'c': DataLoader(PairedSarEoDataset(sar_train_ds, eo_rgbnir_train_ds, augment=True), batch_size=batch_size, shuffle=True, num_workers=4)
    }

    val_dataloaders = {
        'a': DataLoader(PairedSarEoDataset(sar_val_ds, eo_rgb_val_ds), batch_size=batch_size, shuffle=False, num_workers=4),
        'b': DataLoader(PairedSarEoDataset(sar_val_ds, eo_nir_val_ds), batch_size=batch_size, shuffle=False, num_workers=4),
        'c': DataLoader(PairedSarEoDataset(sar_val_ds, eo_rgbnir_val_ds), batch_size=batch_size, shuffle=False, num_workers=4)
    }

    return train_dataloaders, val_dataloaders

def get_test_loader(base_dir: str, config: str, test_scenes: List[int], batch_size: int, season: Seasons = Seasons.WINTER) -> DataLoader:
    """
    Prepares a dataloader for the test set for a specific configuration.
    
    Args:
        base_dir: The root directory of the SEN12MS dataset.
        config: The model configuration to test ('a', 'b', or 'c').
        test_scenes: A list of scene IDs to be used exclusively for testing.
        batch_size: The batch size for the dataloader.
        season: The season to load data from.
        
    Returns:
        A PyTorch DataLoader for the test set.
    """
    sen12ms = SEN12MSDataset(base_dir)
    all_patches = sen12ms.get_season_ids(season)
    test_scene_patch_ids = [(s, p) for s in test_scenes for p in all_patches.get(s, [])]
    
    sar_test_ds = SARDataset(sen12ms, season, test_scene_patch_ids)
    
    if config == 'a':
        eo_test_ds = EORGBDataset(sen12ms, season, test_scene_patch_ids)
    elif config == 'b':
        eo_test_ds = EONirSwirRedEdgeDataset(sen12ms, season, test_scene_patch_ids)
    elif config == 'c':
        eo_test_ds = EORGBNirDataset(sen12ms, season, test_scene_patch_ids)
    else:
        raise ValueError("Configuration must be one of 'a', 'b', or 'c'.")
        
    paired_test_ds = PairedSarEoDataset(sar_test_ds, eo_test_ds, augment=False)
    test_loader = DataLoader(paired_test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader

if __name__ == '__main__':
    # This is an example of how you would use the functions in your training script.
    # Make sure you have downloaded the SEN12MS dataset and placed it in a 'data' directory.
    DATA_DIR = "/kaggle/input/satellitessubset/" # Or the path to your SEN12MS dataset
    BATCH_SIZE = 4
    
    print("Preparing training and validation dataloaders...")
    try:
        train_loaders, val_loaders = get_dataloaders(DATA_DIR, BATCH_SIZE, val_split=0.3)
        
        # You can now access the dataloaders for each configuration
        train_loader_a = train_loaders['a']
        val_loader_c = val_loaders['a']
        
        print(f"Number of batches in training loader 'a': {len(train_loader_a)}")
        print(f"Number of batches in validation loader 'c': {len(val_loader_c)}")
        
        # Example of getting one batch from loader 'a'
        sar_batch, eo_batch = next(iter(train_loader_a))
        print(f"SAR batch shape: {sar_batch.shape}")  # Expected: [4, 2, 256, 256]
        print(f"EO (RGB) batch shape: {eo_batch.shape}") # Expected: [4, 3, 256, 256]

        # Example of preparing a test loader
        # NOTE: You should use a separate set of scene IDs for testing that were not in train/val
        sen12ms_main = SEN12MSDataset(DATA_DIR)
        all_scene_ids = list(sen12ms_main.get_scene_ids(Seasons.WINTER))
        if len(all_scene_ids) > 10: # Ensure there are enough scenes to create a test set
            test_scene_ids = all_scene_ids[-2:] # Use last 2 scenes for testing as an example
            print(f"\nPreparing test loader for config 'b' using scenes: {test_scene_ids}")
            test_loader_b = get_test_loader(DATA_DIR, 'b', test_scenes=test_scene_ids, batch_size=BATCH_SIZE)
            
            sar_test_batch, eo_test_batch = next(iter(test_loader_b))
            print(f"Test SAR batch shape: {sar_test_batch.shape}")
            print(f"Test EO (NIR/SWIR/RE) batch shape: {eo_test_batch.shape}") # Expected: [4, 3, 256, 256]
        else:
            print("\nNot enough scenes in the dataset to create a separate test set for this example.")

    except Exception as e:
        print(f"An error occurred. Please ensure your data directory is correct and contains the dataset.")
        print(f"Error: {e}")

