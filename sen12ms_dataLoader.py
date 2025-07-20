"""
    Generic data loading routines for the SEN12MS dataset of corresponding Sentinel 1,
    Sentinel 2 and Modis LandCover data.

    The SEN12MS class is meant to provide a set of helper routines for loading individual
    image patches as well as triplets of patches from the dataset. These routines can easily
    be wrapped or extended for use with many deep learning frameworks or as standalone helper 
    methods. For an example use case please see the "main" routine at the end of this file.

    NOTE: Some folder/file existence and validity checks are implemented but it is 
          by no means complete.

    Author: Lloyd Hughes (lloyd.hughes@tum.de)
"""

import os
import rasterio
import re

import numpy as np

from enum import Enum
from glob import glob


class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    RGB_NIR = [B04,B03,B02,B08]
    NIR_SWIR_REDEDGE = [B08,B11,B05]
    NONE = []


class LCBands(Enum):
    IGBP = igbp = 1
    LCCS1 = landcover = 2
    LCCS2 = landuse = 3
    LCCS3 = hydrology = 4
    ALL = [IGBP, LCCS1, LCCS2, LCCS3]
    NONE = []


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    lc = "lc"

# Note: The order in which you request the bands is the same order they will be returned in.

class SEN12MSDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            raise Exception(
                "The specified base_dir for SEN12MS dataset does not exist")

    """
        Returns a list of scene ids for a specific season.
    """

    def get_scene_ids(self, season: "Seasons") -> List[int]:
        s1_season_name = f"{season.value}_s1"

        # MODIFICATION HERE: Added "ROIs2017_winter" to the path
        extra_folder = "ROIs2017_winter"
        scenes_dir = os.path.join(self.base_dir, s1_season_name, extra_folder)

        scene_ids = []
        for scene_name in os.listdir(scenes_dir):
            if scene_name.startswith("s1_"):
                scene_id = int(scene_name.split("_")[1])
                scene_ids.append(scene_id)
        return scene_ids

    """
        Returns a list of patch ids for a specific scene within a specific season
    """

    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season, f"s1_{scene_id}")

        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season))

        patch_ids = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]

        return patch_ids

    """
        Return a dict of scene ids and their corresponding patch ids.
        key => scene_ids, value => list of patch_ids
    """

    def get_season_ids(self, season: "Seasons") -> Dict[int, List[int]]:
        season_ids = {}
        for scene_id in self.get_scene_ids(season):
            s1_scene_name = f"s1_{scene_id}"
            s1_season_name = f"{season.value}_s1"
    
            # This part accounts for the extra nested folder for Kaggle
            extra_folder = "ROIs2017_winter"
            scene_dir = os.path.join(self.base_dir, s1_season_name, extra_folder, s1_scene_name)
    
            patch_ids = []
            # Check if the directory exists before trying to list its contents
            if not os.path.isdir(scene_dir):
                continue
            pattern = re.compile(r'_p(\d+)\.tif$')
            for patch_name in os.listdir(scene_dir):
                try:
                    match = pattern.search(patch_name)
                    patch_ids.append(int(match.group(1)))
                except ValueError:
                    print("Conversion failed!")
                    # If conversion to int fails, simply ignore this file and continue
                    continue
            season_ids[scene_id] = patch_ids
        return season_ids

    """
        Returns raster data and image bounds for the defined bands of a specific patch
        This method only loads a sinlge patch from a single sensor as defined by the bands specified
    """

    def get_patch(self, season: "Seasons", scene_id: int, patch_id: int, bands: List[Enum]) -> Tuple[object, str]:
        s1_path, s2_path = self.get_s1_s2_lc_triplet(season, scene_id, patch_id)
    
        if bands[0] in S1Bands:
            path = s1_path
        elif bands[0] in S2Bands:
            path = s2_path
        else:
            raise ValueError(f"Unknown band: {bands[0]}")
    
        band_ids = [band.value for band in bands]
    
        with rasterio.open(path) as f:
            patch = f.read(band_ids)
    
        return patch, path
    """
        Returns a triplet of patches. S1, S2 and LC as well as the geo-bounds of the patch
    """

    def get_s1_s2_lc_triplet(self, season: "Seasons", scene_id: int, patch_id: int) -> Tuple[str, str, str]:
        s1_scene_name = f"s1_{scene_id}"
        s2_scene_name = f"s2_{scene_id}"

        s1_season_name = f"{season.value}_s1"
        s2_season_name = f"{season.value}_s2"

        # MODIFICATION HERE: Added "ROIs2017_winter" to the path construction
        extra_folder = "ROIs2017_winter"
        s1_patch_name = f"{season.value}_{s1_scene_name}_p{patch_id}.tif"
        s2_patch_name = f"{season.value}_{s2_scene_name}_p{patch_id}.tif" # Use s2_scene_name here
        
        s1_path = os.path.join(self.base_dir, s1_season_name, extra_folder, s1_scene_name, s1_patch_name)
        s2_path = os.path.join(self.base_dir, s2_season_name, extra_folder, s2_scene_name, s2_patch_name)

        return s1_path, s2_path

    """
        Returns a triplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2 or LC
    """

    def get_triplets(self, season, scene_ids=None, patch_ids=None, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL):
        season = Seasons(season)
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        lc_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, lc, bound = self.get_s1s2lc_triplet(
                    season, sid, pid, s1_bands, s2_bands, lc_bands)
                s1_data.append(s1)
                s2_data.append(s2)
                lc_data.append(lc)
                bounds.append(bound)

        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(lc_data, axis=0), bounds