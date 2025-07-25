from sen12ms_dataLoader import SEN12MSDataset, S1Bands, S2Bands, Seasons
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict
from scipy.ndimage import uniform_filter, variance , gaussian_filter
import os

def norm_db(arr_db: np.ndarray, lo: float = -30.0, hi: float = 0.0) -> np.ndarray:
    arr_db = np.clip(arr_db, lo, hi)
    return (arr_db - lo) / (hi - lo) * 2.0 - 1.0

class SARDataset(Dataset):
    def __init__(self, sen12ms_instance: SEN12MSDataset, season: Seasons, scene_patch_ids: List[Tuple[int, int]],cache_dir: str | None = None):
        self.sen12ms = sen12ms_instance
        self.season = season
        self.scene_patch_ids = scene_patch_ids
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.scene_patch_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        scene_id, patch_id = self.scene_patch_ids[idx]

        # Caching for faster access
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{scene_id}_{patch_id}.npy")
            if os.path.isfile(cache_path):
                return torch.from_numpy(np.load(cache_path)).float()
        bands_to_load = [S1Bands.VV, S1Bands.VH]
        sar_image_raw, _ = self.sen12ms.get_patch(self.season, scene_id, patch_id, bands=bands_to_load)
                
        vv_db = sar_image_raw[0, :, :] 
        vh_db = sar_image_raw[1, :, :] 

        # Apply speckle filtering 
        vv_db = gaussian_filter(vv_db, sigma=0.8)  
        vh_db = gaussian_filter(vh_db, sigma=0.8)

        vv_norm = norm_db(vv_db, lo=-30.0, hi=0.0)    
        vh_norm = norm_db(vh_db, lo=-35.0, hi=-5.0)   
        
        # Compute ratio channel (difference in dB = log ratio)
        ratio_db = vh_db - vv_db
        ratio_db = np.clip(ratio_db, -15.0, 5.0)
        ratio_norm = (ratio_db + 15.0) / 20.0 * 2.0 - 1.0
        
        sar_processed = np.stack([vv_norm, vh_norm, ratio_norm], axis=0)
        sar_tensor = torch.from_numpy(sar_processed.astype(np.float32))
        
        if self.cache_dir:
            np.save(cache_path, sar_processed.astype(np.float32))
        return sar_tensor
    
class EORGBDataset(Dataset):
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
    sar_train_ds = SARDataset(sen12ms, season, train_scene_patch_ids,cache_dir="./sar_cache")
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