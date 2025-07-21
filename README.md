# SAR-to-EO Image Translation Using CycleGAN

## Team Members

- Devesh Bhushan


## Project Overview

This project implements a **CycleGAN-based SAR-to-EO image translation system** that converts Sentinel-1 SAR images into corresponding Sentinel-2 EO images. The implementation explores multiple spectral band configurations and incorporates several advanced techniques beyond the baseline CycleGAN architecture to improve translation quality and training stability.

The project addresses the challenge of translating synthetic aperture radar (SAR) imagery into electro-optical (EO) imagery, enabling better visualization and interpretation of SAR data for remote sensing applications.

## Key Novelties and Improvements Beyond Baseline

### 1. **Multi-Configuration Spectral Band Analysis**

Unlike standard single-output approaches, our implementation systematically evaluates three distinct spectral band configurations:

- **Configuration A**: RGB bands (B4, B3, B2) - Standard visible spectrum
- **Configuration B**: NIR-SWIR-RedEdge bands (B8, B11, B5) - Advanced spectral information
- **Configuration C**: RGB+NIR bands (B4, B3, B2, B8) - Hybrid approach with 4-channel output


### 2. **Enhanced Loss Function Architecture**

- **SSIM Loss Integration**: Added Structural Similarity Index Measure (SSIM) loss with λ=0.1 to improve perceptual quality
- **Weighted Loss Combination**: Optimized combination of adversarial, cycle-consistency, identity, and perceptual losses
- **Label Smoothing**: Implemented with factor 0.1 to improve training stability


### 3. **Advanced Network Architecture Improvements**

- **Spectral Normalization**: Applied to discriminator layers for training stabilization
- **Gradient Clipping**: Implemented with max_norm=1.0 to prevent gradient explosion
- **Instance Normalization**: Used throughout generator and discriminator for better performance


### 4. **Sophisticated Data Processing Pipeline**

- **Speckle Filtering**: Gaussian filtering (σ=0.8) applied to SAR VV and VH channels
- **Multi-Channel SAR Processing**: VV, VH, and VV/VH ratio channels for comprehensive SAR representation
- **Intelligent Caching System**: Implemented for faster data loading during training
- **Data Augmentation**: Random horizontal/vertical flips with consistent application to SAR-EO pairs


### 5. **Advanced Training Strategies**

- **Mixed Precision Training**: CUDA AMP integration for faster training and memory efficiency
- **Cosine Learning Rate Scheduling**: Smooth learning rate decay for better convergence
- **Differential Learning Rates**: Generator (2e-4) vs Discriminator (1e-4) for balanced training
- **Early Stopping with Patience**: PSNR-based monitoring with minimum improvement threshold


### 6. **Comprehensive Evaluation Framework**

- **Real-time Validation**: PSNR and SSIM metrics computed during training
- **Multi-GPU Support**: DataParallel implementation for scalable training
- **Detailed Logging**: CSV-based training metrics logging for analysis
- **Visual Validation**: Automatic generation of comparison grids during training


## Project Structure

```
Project1_SAR_to_EO/
│
├── preprocess.py              # Data loading and preprocessing
├── train_cycleGAN.py         # Main training script with advanced features
├── model.py                  # CycleGAN architecture with improvements  
├── config.py                 # Configuration parameters
├── utils.py                  # Utility functions for checkpointing
├── sen12ms_dataLoader.py     # SEN12MS dataset interface
├── evaluate_results.py       # Comprehensive evaluation and visualization
├── cyclegan_performance_comparison.csv  # Performance results
└── generated_samples/             # Generated sample images
```


## Data Preprocessing Steps

### SAR Data Processing

1. **Band Selection**: VV and VH polarizations from Sentinel-1
2. **Speckle Reduction**: Gaussian filtering with σ=0.8 applied to both channels
3. **Normalization**:
    - VV: dB range [-30, 0] → [-1, 1]
    - VH: dB range [-35, -5] → [-1, 1]
    - Ratio: dB range [-15, 5] → [-1, 1]
4. **Multi-Channel Representation**: Stack of VV, VH, and VV/VH ratio

### EO Data Processing

1. **Band Selection**: Configuration-specific spectral bands
2. **Normalization**: Raw values → [-1, 1]
3. **Channel Alignment**: Proper handling of different channel counts across configurations

## Models Used

### Generator Architecture

- **ResNet-based Generator**: 6 residual blocks for effective feature learning
- **Instance Normalization**: Applied throughout for improved training dynamics
- **Reflection Padding**: Reduces boundary artifacts in generated images
- **Tanh Activation**: Output range [-1, 1] matching input normalization


### Discriminator Architecture

- **PatchGAN Discriminator**: 70×70 receptive field for local realism assessment
- **Spectral Normalization**: Stabilizes discriminator training
- **Leaky ReLU Activation**: α=0.2 for improved gradient flow
- **Progressive Feature Extraction**: Multi-scale feature analysis


## Key Findings and Observations

### Performance Analysis

Based on our comprehensive evaluation[^1]:


| Configuration | Final PSNR | Final SSIM | Generator Loss | Discriminator Loss |
| :-- | :-- | :-- | :-- | :-- |
| RGB (A) | **20.23 dB** | **0.6609** | 2.149 | 0.098 |
| NIR-SWIR-RedEdge (B) | 17.15 dB | 0.3975 | 2.720 | 0.080 |
| RGB+NIR (C) | 20.14 dB | **0.6670** | 2.198 | 0.091 |

### Key Observations

1. **RGB Configuration (A)** achieves the highest PSNR (20.23 dB), indicating superior pixel-level accuracy
2. **RGB+NIR Configuration (C)** achieves the highest SSIM (0.6670), suggesting better structural similarity
3. **NIR-SWIR-RedEdge Configuration (B)** shows more challenging training dynamics but provides unique spectral information
4. **Training Stability**: All configurations demonstrate stable discriminator scores (Real: ~0.67, Fake: ~0.32)

## Tools and Frameworks Used

- **PyTorch**: Deep learning framework
- **torchmetrics**: PSNR and SSIM evaluation
- **torchvision**: Image utilities and transformations
- **NumPy**: Numerical computations
- **Pandas**: Data analysis and logging
- **Matplotlib/Seaborn**: Visualization and plotting
- **SciPy**: Advanced image filtering
- **Rasterio**: Geospatial image I/O
- **tqdm**: Progress monitoring


## Instructions to Run Code

### Prerequisites

```bash
pip install torch torchvision torchmetrics numpy pandas matplotlib seaborn scipy rasterio tqdm
```


### Dataset Setup

1. Download SEN12MS winter data:
    - `ROIs2017_winter_s1.tar.gz`
    - `ROIs2017_winter_s2.tar.gz`
2. Extract to `./data/` directory
3. Ensure proper folder structure as expected by `sen12ms_dataLoader.py`

### Training

```bash
# Train RGB configuration (default)
python train_cycleGAN.py

# Modify model.py to change configuration:
# - ModelType.SAR_TO_EORGB (Configuration A)
# - ModelType.SAR_TO_EONIRSWIR (Configuration B)  
# - ModelType.SAR_TO_EORGBNIR (Configuration C)
```


### Evaluation

```bash
python evaluate_results.py
```


## Sample Results

### Visual Translation Examples

Our CycleGAN implementation produces high-quality SAR-to-EO translations across all three spectral configurations. Below are representative examples showcasing the translation quality for different band combinations:

First Layer is of SAR Images of 3 bands (VV, VH, Ratio of VV and VH)
Second Layer is Generated EO Image
Third Layer is Ground Truth

#### Configuration A: RGB Translation (B4, B3, B2)
**Standard visible spectrum translation providing natural color representation**

![RGB Translation Example](./generated_samples/SAR%20->%20EO(RGB%20band)/SAR_TO_EO_RGB.png)

*SAR-to-RGB translation demonstrating natural color synthesis with high structural preservation*

#### Configuration B: NIR-SWIR-RedEdge Translation (B8, B11, B5)
**Advanced spectral information capture for specialized remote sensing applications**

![NIR-SWIR Translation Example](./generated_samples/SAR%20->%20EO%20(NIR,%20SWIR,%20Red%20Edges)/SAR_TO_EO_NIRSWIR.png)

*SAR-to-NIR-SWIR translation highlighting vegetation and moisture content analysis capabilities*

#### Configuration C: RGB+NIR Translation (B4, B3, B2, B8)
**Hybrid approach combining visible spectrum with near-infrared information**

##### NIR Image only
![RGB-NIR Translation Example 1](generated_samples/SAR%20->%20EO%20(RGB%20and%20NIR)/SAR_TO_EONIRRGB_NIR.png)

##### RGB Image only
![RGB-NIR Translation Example 2](generated_samples/SAR%20->%20EO%20(RGB%20and%20NIR)/SAR_TO_EONIRRGB_RGB.png)

*Top: NIR channel visualization | Bottom: RGB channel visualization from the 4-channel output*

### Translation Quality Analysis

#### Visual Quality Assessment
- **Structural Preservation**: All configurations maintain geometric accuracy and spatial relationships from SAR input
- **Spectral Consistency**: Generated EO images exhibit realistic spectral characteristics for their respective band combinations
- **Artifact Reduction**: Minimal translation artifacts with smooth transitions and natural textures

#### Configuration-Specific Strengths

| Configuration | Key Strengths | Best Use Cases |
|---------------|--------------|----------------|
| **RGB (A)** | Natural color representation, highest PSNR (20.23 dB) | Visual interpretation, general mapping |
| **NIR-SWIR-RedEdge (B)** | Specialized spectral information, vegetation analysis | Agricultural monitoring, environmental studies |
| **RGB+NIR (C)** | Comprehensive spectral coverage, highest SSIM (0.6670) | Multi-spectral analysis, research applications |

### Sample Image Details

**Image Specifications:**
- **Resolution**: 256×256 pixels
- **Data Format**: PNG (8-bit per channel)
- **Normalization**: Values scaled to [0, 255] for display
- **Color Space**: 
  - RGB configurations: Standard RGB color space
  - NIR/SWIR configurations: False-color representation optimized for analysis

**Translation Pipeline:**
1. **Input**: Sentinel-1 SAR imagery (VV, VH polarizations + ratio)
2. **Processing**: CycleGAN generator with 6 ResNet blocks
3. **Output**: Corresponding Sentinel-2 EO imagery in target spectral bands
4. **Post-processing**: Denormalization and format conversion for visualization

### Performance Validation

The sample results demonstrate our model's capability to:
- **Preserve spatial information** while transforming from radar to optical domain
- **Generate realistic textures** appropriate for each spectral configuration
- **Maintain consistency** across different terrain types and land cover classes
- **Handle various weather conditions** present in the original SAR imagery

For quantitative performance metrics and detailed analysis, refer to the [Performance Analysis](#key-findings-and-observations) section above.


### Performance Metrics

Our advanced CycleGAN implementation achieved:

- **Best PSNR**: 20.23 dB (RGB configuration)
- **Best SSIM**: 0.6670 (RGB+NIR configuration)
- **Training Stability**: Consistent discriminator performance across all configurations
- **Convergence**: Stable learning curves with effective loss balancing


### Visual Quality Improvements

The implementation produces high-quality SAR-to-EO translations with:

- Enhanced structural preservation through SSIM loss
- Reduced artifacts via spectral normalization
- Improved color consistency across different spectral configurations
- Better handling of various terrain types and weather conditions


## Configuration Parameters

Key hyperparameters optimized for best performance:

- **Batch Size**: 8 (memory-optimized for multi-GPU training)
- **Learning Rates**: Generator (2e-4), Discriminator (1e-4)
- **Loss Weights**: Cycle (10), Identity (0.5), SSIM (0.1)
- **Training Epochs**: 20 with cosine scheduling
- **Gradient Clipping**: 1.0 for stability


## Reproducibility

- **Deterministic Training**: Fixed random seeds (42) for reproducible results
- **Checkpoint System**: Automatic model saving for best validation performance
- **Environment Configuration**: CUDA optimization with mixed precision training
- **Logging System**: Comprehensive CSV logging for analysis and comparison
