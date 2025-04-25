"""
Satellite Image Processing Library for Agricultural Analysis

This module provides functions for processing satellite imagery to analyze 
agricultural data, compute vegetation indices, and estimate carbon sequestration.

Main functionality includes:
- Radiometric correction and geometric alignment
- Noise reduction using various filtering methods
- NDVI (Normalized Difference Vegetation Index) computation
- Biomass and carbon storage estimation
- Carbon credit calculation
"""

import json
import os
import time
import numpy as np
import concurrent.futures
import tifffile as tiff
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, warp
from skimage.io import imsave
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
from PIL import Image


# -----------------------------------------------------------------------------
# S3 OPERATIONS
# -----------------------------------------------------------------------------

def read_image_from_s3(bucket_name, key):
    """
    Download an image from S3 and load it based on its file type.
    
    Parameters:
        bucket_name (str): The S3 bucket name.
        key (str): The S3 object key.
    
    Returns:
        numpy.ndarray or PIL.Image: TIFF images are returned as NumPy arrays (float32),
                                    JPEG images are returned as PIL.Image instances.
    
    Raises:
        ValueError: If the image format is not supported.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
        region_name=os.environ.get('AWS_REGION')
    )
    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    image_data = response['Body'].read()
    
    # Create a file-like object from the data
    file_obj = BytesIO(image_data)
    
    # Check file extension to decide how to load the image
    key_lower = key.lower()
    if key_lower.endswith(('.tif', '.tiff')):
        # Load TIFF image using tifffile and convert to float32 NumPy array
        image = tiff.imread(file_obj).astype(np.float32)
    elif key_lower.endswith(('.jpg', '.jpeg')):
        # Load JPEG image using PIL
        image = Image.open(file_obj)
    else:
        raise ValueError("Unsupported image format. Please use TIFF or JPEG.")
    
    return image


# -----------------------------------------------------------------------------
# RADIOMETRIC CORRECTION
# -----------------------------------------------------------------------------

def dark_object_subtraction(band, low_percentile=1):
    """
    Compute a robust dark object value by averaging all pixels below the given low percentile.
    
    Parameters:
        band (np.ndarray): Single band reflectance array.
        low_percentile (float): Percentile (0-100) used to threshold dark pixels (default is 1).
        
    Returns:
        float: Robust dark object value for the band.
    """
    # Determine the threshold at the low_percentile (e.g., 1st percentile)
    threshold = np.percentile(band, low_percentile)
    
    # Select pixels that are below or equal to this threshold
    dark_pixels = band[band <= threshold]
    
    # Compute and return the mean of these dark pixels
    return np.mean(dark_pixels) if dark_pixels.size > 0 else 0.0


def rigorous_radiometric_correction(image, gain, offset, sunelev, edist, Esun, 
                                   blackadjust=0.01, low_percentile=1):
    """
    Perform rigorous radiometric correction for a multispectral image using a robust dark object subtraction.
    
    Parameters:
        image (np.ndarray): Input multispectral image.
        gain (array-like): Sensor gain for each band.
        offset (array-like): Sensor offset for each band.
        sunelev (float): Sun elevation in degrees.
        edist (float): Earth–Sun distance (in appropriate units).
        Esun (array-like): Exo-atmospheric solar irradiance for each band.
        blackadjust (float): Adjustment factor for dark object subtraction.
        low_percentile (float): Percentile (0-100) to estimate the dark object using a trimmed mean.
        
    Returns:
        np.ndarray: Corrected reflectance image in shape (bands, H, W) with values in [0,1].
    """
    # The array is (H, W, bands) and we need (bands, H, W), transpose it.
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    
    if image is None:
        raise ValueError("Could not read image!")
    
    n_bands, h, w = image.shape
    radiance = np.empty_like(image)
    
    # Convert DN to radiance for each band using provided gain and offset.
    for i in range(n_bands):
        radiance[i] = gain[i] * image[i] + offset[i]
    
    # Compute sun zenith angle and its cosine.
    sun_zenith = 90 - sunelev
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    
    # Ensure Esun is an array.
    Esun = np.array(Esun)
    
    # Apply the standard TOA reflectance formula:
    reflectance = (np.pi * radiance * (edist**2)) / (Esun[:, None, None] * cos_sun_zenith)
    
    # Use the robust dark object method: for each band, compute the mean of the lowest low_percentile values.
    dark_obj = np.empty(n_bands, dtype=np.float32)
    for i in range(n_bands):
        dark_obj[i] = dark_object_subtraction(reflectance[i], low_percentile=low_percentile)
    
    # Apply an adjustment factor to the estimated dark object values.
    dark_obj_adj = dark_obj * (1 - blackadjust)
    
    # Subtract the dark object value from each band and clip the results to [0, 1].
    corrected = np.empty_like(reflectance)
    for i in range(n_bands):
        corrected[i] = np.clip(reflectance[i] - dark_obj_adj[i], 0, 1)
    
    return corrected


# -----------------------------------------------------------------------------
# GEOMETRIC CORRECTION
# -----------------------------------------------------------------------------

def geometric_correction_pipeline(image, radiometric_params, detector_type='ORB', visualize=False):
    """
    Executes the radiometric correction followed by geometric correction.
    
    Parameters:
        image (np.ndarray): Input multispectral image.
        radiometric_params (dict): Radiometric parameters.
        detector_type (str): Feature detector type.
        visualize (bool): If True, display images.
        
    Returns:
        np.ndarray: Radiometrically corrected image (H, W, channels).
    """
    # Apply radiometric correction (output shape: (bands, H, W))
    tgt_corr = rigorous_radiometric_correction(image, **radiometric_params)
    
    # Transpose to (H, W, bands) for geometric processing
    tgt_img = np.transpose(tgt_corr, (1, 2, 0))
    
    if visualize:
        def prepare_display(img):
            # Use only the first 3 bands for RGB visualization if available.
            return img[..., :3] if img.shape[2] >= 3 else img
            
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 2)
        plt.imshow(prepare_display(tgt_img))
        plt.title("Target (Radiometrically Corrected)")
        plt.show()
    
    return tgt_img


# -----------------------------------------------------------------------------
# NOISE REDUCTION
# -----------------------------------------------------------------------------

def noise_reduction_filter(image, method='median', kernel_size=3, sigma=1.0):
    """
    Apply noise reduction to a multispectral image.
    
    Parameters:
        image (np.ndarray): Input image (H, W, bands) with values in [0, 1].
        method (str): 'median' or 'gaussian'.
        kernel_size (int): Kernel size (must be odd).
        sigma (float): Standard deviation for Gaussian (ignored for median).
        
    Returns:
        np.ndarray: Filtered image.
    """
    filtered = np.empty_like(image)
    for band in range(image.shape[2]):
        if method == 'median':
            filtered[..., band] = median_filter(image[..., band], size=kernel_size)
        elif method == 'gaussian':
            truncate_val = ((kernel_size - 1) / 2) / sigma
            filtered[..., band] = gaussian_filter(image[..., band], sigma=sigma, truncate=truncate_val)
        else:
            raise ValueError("Method must be either 'median' or 'gaussian'.")
    return filtered


def noise_reduction_filter_parallel(image, method='median', kernel_size=3, sigma=1.0):
    """
    Parallel noise reduction filter using ThreadPoolExecutor for each band.
    
    Parameters:
        image (np.ndarray): Input image (H, W, bands) with values in [0, 1].
        method (str): 'median' or 'gaussian'.
        kernel_size (int): Kernel size.
        sigma (float): Sigma for Gaussian.
        
    Returns:
        np.ndarray: Filtered image.
    """
    filtered = np.empty_like(image)
    
    def process_band(band):
        if method == 'median':
            return median_filter(image[..., band], size=kernel_size)
        elif method == 'gaussian':
            truncate_val = ((kernel_size - 1) / 2) / sigma
            return gaussian_filter(image[..., band], sigma=sigma, truncate=truncate_val)
        else:
            raise ValueError("Method must be either 'median' or 'gaussian'.")
            
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_band, range(image.shape[2])))
        
    for i, res in enumerate(results):
        filtered[..., i] = res
        
    return filtered


# -----------------------------------------------------------------------------
# VEGETATION INDEX CALCULATIONS
# -----------------------------------------------------------------------------

def compute_ndvi(image, red_band_index, nir_band_index):
    """
    Compute the Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Parameters:
        image (np.ndarray): Input image (H, W, bands), assumed in [0,1].
        red_band_index (int): Index for Red band.
        nir_band_index (int): Index for NIR band.
        
    Returns:
        np.ndarray: NDVI image.
    """
    nir = image[..., nir_band_index].astype(np.float32)
    red = image[..., red_band_index].astype(np.float32)
    return (nir - red) / (nir + red + 1e-10)  # Small epsilon to avoid division by zero


def ndvi_to_kc(ndvi, a=1.25, b=0.1):
    """
    Converts NDVI values to crop coefficient (Kc) using a linear relation.
    
    Parameters:
        ndvi (np.ndarray): NDVI image.
        a (float): Slope coefficient (default: 1.25).
        b (float): Intercept coefficient (default: 0.1).
        
    Returns:
        np.ndarray: Crop coefficient map.
    """
    kc = a * ndvi + b
    kc = np.clip(kc, 0.1, 1.2)  # Typical Kc range
    return kc


def estimate_cwr_from_ndvi_and_weather(kc_map, eto_today):
    """
    Estimate Crop Water Requirement (CWR) using average Kc from NDVI and ET₀.
    
    Parameters:
        kc_map (np.ndarray): Crop coefficient map.
        eto_today (float): Reference evapotranspiration for today.
        
    Returns:
        float: Estimated crop water requirement.
    """
    avg_kc = np.mean(kc_map)
    return round(avg_kc * eto_today, 2)


# -----------------------------------------------------------------------------
# BIOMASS AND CARBON ESTIMATION
# -----------------------------------------------------------------------------

def estimate_biomass(ndvi, a=2.5, b=1.2):
    """
    Estimate Above-Ground Biomass (AGB) using an empirical NDVI-to-biomass model.
    
    Parameters:
        ndvi (np.ndarray): NDVI image.
        a (float): Biomass coefficient (default: 2.5).
        b (float): NDVI exponent factor (default: 1.2).
        
    Returns:
        np.ndarray: Estimated biomass per pixel in kg/m².
    """
    return a * np.exp(b * ndvi)  # Biomass in kg/m²


def calculate_carbon_storage(biomass, carbon_fraction, default_carbon_fraction=0.47):
    """
    Convert biomass to stored carbon using the carbon fraction from the analysis response,
    or a default value if none is provided.

    Parameters:
        biomass (np.ndarray): Biomass per pixel (kg/m²).
        carbon_fraction (float): Fraction of biomass that is carbon.
        default_carbon_fraction (float): Default fraction used if carbon_fraction is None.
        
    Returns:
        np.ndarray: Carbon stored per pixel (kg/m²).
    """
    # Use the carbon fraction from the analysis response if available and valid
    if carbon_fraction is not None:
        carbon_fraction = carbon_fraction
    else:
        carbon_fraction = default_carbon_fraction
    
    return biomass * carbon_fraction


def estimate_root_biomass(above_ground_biomass, root_shoot_ratio, default_ratio=0.5):
    """
    Estimates below-ground biomass using the root-shoot ratio.

    Parameters:
        above_ground_biomass (np.ndarray): Above-ground biomass per pixel (kg/m²).
        root_shoot_ratio (float): Ratio of below-ground to above-ground biomass.
        default_ratio (float): Default root-shoot ratio if specific ratio is not provided.

    Returns:
        np.ndarray: Estimated below-ground biomass per pixel (kg/m²).
    """
    if root_shoot_ratio is not None:
        return above_ground_biomass * root_shoot_ratio
    else:
        print("Warning: Using default root-shoot ratio for root biomass estimation.")
        return above_ground_biomass * default_ratio


def calculate_root_carbon_storage(root_biomass, carbon_fraction, default_carbon_fraction=0.47):
    """
    Convert root biomass to stored carbon.

    Parameters:
        root_biomass (np.ndarray): Root biomass per pixel (kg/m²).
        carbon_fraction (float): Fraction of biomass that is carbon.
        default_carbon_fraction (float): Default fraction if carbon_fraction is None.

    Returns:
        np.ndarray: Carbon stored in roots per pixel (kg/m²).
    """
    if carbon_fraction is None:
        carbon_fraction = default_carbon_fraction
    return root_biomass * carbon_fraction


def convert_carbon_to_co2(carbon_storage):
    """
    Convert stored carbon to CO₂ equivalent.
    
    Parameters:
        carbon_storage (np.ndarray): Carbon stored per pixel (kg/m²).
        
    Returns:
        np.ndarray: CO₂ sequestered per pixel (kg/m²).
    """
    return carbon_storage * (44 / 12)  # CO₂ equivalent (kg/m²)


def calculate_carbon_credits(co2_sequestration, pixel_area_m2):
    """
    Calculate total carbon credits from CO₂ sequestration.
    
    Parameters:
        co2_sequestration (np.ndarray): CO₂ stored per pixel (kg/m²).
        pixel_area_m2 (float): Area of each pixel in square meters.
        
    Returns:
        float: Total carbon credits (1 credit = 1 metric ton CO₂).
    """
    total_co2_kg = np.sum(co2_sequestration * pixel_area_m2)  # Convert per-pixel to total CO₂ in kg
    total_co2_tonnes = total_co2_kg / 1000  # Convert kg to metric tonnes
    return total_co2_tonnes  # Carbon credits (1 credit = 1 tCO₂e)


# -----------------------------------------------------------------------------
# INTEGRATED PROCESSING PIPELINES
# -----------------------------------------------------------------------------

def full_image_processing_pipeline(image, radiometric_params, detector_type,
                                  noise_method, noise_kernel_size, sigma, 
                                  nir_band_index, red_band_index,
                                  visualize=False, use_parallel_noise_reduction=False):
    """
    Pipeline for radiometric correction, noise reduction, and NDVI computation.
    
    Parameters:
        image (np.ndarray): Input multispectral image.
        radiometric_params (dict): Parameters for radiometric correction.
        detector_type (str): Type of feature detector to use.
        noise_method (str): Noise reduction method ('median' or 'gaussian').
        noise_kernel_size (int): Kernel size for noise reduction.
        sigma (float): Sigma for Gaussian filter.
        nir_band_index (int): Index of NIR band.
        red_band_index (int): Index of Red band.
        visualize (bool): Whether to visualize results.
        use_parallel_noise_reduction (bool): Whether to use parallel processing for noise reduction.
        
    Returns:
        tuple: (NDVI image after noise reduction, NDVI RMSE, crop coefficient map)
    """
    # Step 1: Apply radiometric and geometric corrections
    aligned_img = geometric_correction_pipeline(image, radiometric_params, detector_type, visualize)

    # Step 2: Apply noise reduction
    noise_reduction_fn = noise_reduction_filter_parallel if use_parallel_noise_reduction else noise_reduction_filter
    noise_reduced_img = noise_reduction_fn(
        aligned_img,
        method=noise_method,
        kernel_size=noise_kernel_size,
        sigma=sigma
    )

    # Step 3: Compute NDVI for both aligned and noise-reduced images
    ndvi_aligned = compute_ndvi(aligned_img, red_band_index, nir_band_index)
    ndvi_noise_reduced = compute_ndvi(noise_reduced_img, red_band_index, nir_band_index)
    
    # Step 4: Calculate RMSE between the two NDVI results
    ndvi_rmse = np.sqrt(np.mean((ndvi_aligned - ndvi_noise_reduced) ** 2))
    
    # Step 5: Calculate crop coefficient map
    kc_map = ndvi_to_kc(ndvi_noise_reduced)

    # Visualization if requested
    if visualize:
        plt.figure(figsize=(12, 5))
        for i, (title, data) in enumerate(zip(["Aligned Image NDVI", "Noise-Reduced NDVI"], 
                                             [ndvi_aligned, ndvi_noise_reduced])):
            plt.subplot(1, 2, i + 1)
            plt.imshow(data, cmap='RdYlGn')
            plt.title(title)
            plt.colorbar()
        plt.show()

    return ndvi_noise_reduced, ndvi_rmse, kc_map


def process_ndvi_for_carbon_credits(ndvi_noise_reduced, carbon_fraction, pixel_area_m2, root_shoot_ratio):
    """
    Full pipeline to process NDVI into carbon credits.
    
    Parameters:
        ndvi_noise_reduced (np.ndarray): NDVI image.
        carbon_fraction (float): Fraction of biomass that is carbon.
        pixel_area_m2 (float): Area of each pixel in square meters.
        root_shoot_ratio (float): Ratio of below-ground to above-ground biomass.
        
    Returns:
        float: Total carbon credits for the analyzed area.
    """
    # Step 1: Estimate above-ground biomass
    biomass = estimate_biomass(ndvi_noise_reduced)

    # Step 2: Convert above-ground biomass to carbon storage
    carbon_storage_above_ground = calculate_carbon_storage(biomass, carbon_fraction)

    # Step 3: Estimate below-ground biomass
    root_biomass = estimate_root_biomass(biomass, root_shoot_ratio, default_ratio=0.5)

    # Step 4: Convert root biomass to carbon storage
    carbon_storage_below_ground = calculate_root_carbon_storage(root_biomass, carbon_fraction)

    # Step 5: Calculate total carbon storage
    total_carbon_storage = carbon_storage_above_ground + carbon_storage_below_ground

    # Step 6: Convert total carbon storage to CO₂ equivalent
    co2_sequestration = convert_carbon_to_co2(total_carbon_storage)

    # Step 7: Calculate total carbon credits
    carbon_credits = calculate_carbon_credits(co2_sequestration, pixel_area_m2)
    
    return carbon_credits