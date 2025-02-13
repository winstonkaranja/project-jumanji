import os
import time
import numpy as np
import concurrent.futures
import rasterio
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, warp
from skimage.io import imsave
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

# -------------------------------
# Radiometric & Geometric Correction (Tasks 1 & 2)
# -------------------------------
def rigorous_radiometric_correction_rasterio(image_path, gain, offset, sunelev, edist, Esun, blackadjust=0.01, percentile=0.1):
    """
    Perform rigorous radiometric correction for a multispectral image.
    
    Parameters:
      image_path (str): Path to the multispectral image.
      gain (array-like): Sensor gain for each band.
      offset (array-like): Sensor offset for each band.
      sunelev (float): Sun elevation in degrees.
      edist (float): Earth–Sun distance.
      Esun (array-like): Exo-atmospheric solar irradiance for each band.
      blackadjust (float): Adjustment for dark object subtraction.
      percentile (float): Percentile for dark object estimation.
      
    Returns:
      np.ndarray: Corrected reflectance image in shape (bands, H, W) with values in [0,1].
    """
    # Task 6: Memory Optimization – ensure we use float32 and avoid extra copies.
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # Shape: (bands, H, W)
    
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    n_bands, h, w = image.shape
    radiance = np.empty_like(image)
    for i in range(n_bands):
        radiance[i, :, :] = gain[i] * image[i, :, :] + offset[i]
    
    sun_zenith = 90 - sunelev
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    Esun = np.array(Esun)
    reflectance = (np.pi * radiance) / (Esun[:, None, None] * cos_sun_zenith)
    reflectance /= edist**2
    
    dark_obj = np.percentile(reflectance, percentile, axis=(1, 2))
    dark_obj_adj = dark_obj * (1 - blackadjust)
    corrected = np.clip(reflectance - dark_obj_adj[:, None, None], 0, 1)
    return corrected

def compute_homography_sk(ref_img, tgt_img, detector_type='ORB'):
    """
    Computes the projective transform (homography) between two images using ORB features.
    
    Parameters:
      ref_img (np.ndarray): Reference image (H, W, channels).
      tgt_img (np.ndarray): Target image (H, W, channels).
      detector_type (str): Currently supports 'ORB'.
      
    Returns:
      ProjectiveTransform: Estimated projective transform.
    """
    # Convert to grayscale by averaging over bands
    ref_gray = np.mean(ref_img, axis=2) if ref_img.ndim == 3 else ref_img
    tgt_gray = np.mean(tgt_img, axis=2) if tgt_img.ndim == 3 else tgt_img

    orb = ORB(n_keypoints=500)
    orb.detect_and_extract(ref_gray)
    keypoints1, descriptors1 = orb.keypoints, orb.descriptors

    orb.detect_and_extract(tgt_gray)
    keypoints2, descriptors2 = orb.keypoints, orb.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches to compute homography.")
    
    src = keypoints1[matches[:, 0]]
    dst = keypoints2[matches[:, 1]]
    
    model_robust, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4,
                                   residual_threshold=2, max_trials=1000)
    return model_robust

def warp_image_sk(image, transform, output_shape):
    """
    Warps the image using the given projective transform.
    
    Parameters:
      image (np.ndarray): Image (H, W, channels).
      transform (ProjectiveTransform): Estimated transform.
      output_shape (tuple): Desired output shape.
      
    Returns:
      np.ndarray: Warped (aligned) image.
    """
    return warp(image, inverse_map=transform.inverse, output_shape=output_shape)

def validate_alignment(ref_img, aligned_img):
    """
    Computes the RMSE between the reference and aligned images.
    
    Parameters:
      ref_img (np.ndarray): Reference image (H, W, channels).
      aligned_img (np.ndarray): Aligned image (H, W, channels).
      
    Returns:
      float: RMSE value.
    """
    diff = (ref_img - aligned_img) ** 2
    return np.sqrt(np.mean(diff))

def geometric_correction_pipeline(reference_image_path, target_image_path, radiometric_params, detector_type='ORB', visualize=False):
    """
    Executes the radiometric correction followed by geometric correction.
    
    Parameters:
      reference_image_path (str): Path to the reference image.
      target_image_path (str): Path to the target image.
      radiometric_params (dict): Radiometric parameters.
      detector_type (str): Feature detector type.
      visualize (bool): If True, display images.
      
    Returns:
      tuple: (Estimated transform, aligned image (H, W, channels), alignment RMSE)
    """
    # Apply radiometric correction to both images (output shape: (bands, H, W))
    ref_corr = rigorous_radiometric_correction_rasterio(reference_image_path, **radiometric_params)
    tgt_corr = rigorous_radiometric_correction_rasterio(target_image_path, **radiometric_params)
    
    # Transpose to (H, W, bands) for geometric processing
    ref_img = np.transpose(ref_corr, (1, 2, 0))
    tgt_img = np.transpose(tgt_corr, (1, 2, 0))
    
    # Compute the transform
    transform = compute_homography_sk(ref_img, tgt_img, detector_type)
    aligned_img = warp_image_sk(tgt_img, transform, ref_img.shape)
    rmse = validate_alignment(ref_img, aligned_img)
    
    if visualize:
        def prepare_display(img):
            # Use only the first 3 bands for RGB visualization if available.
            return img[..., :3] if img.shape[2] >= 3 else img
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(prepare_display(ref_img))
        plt.title("Reference (Radiometrically Corrected)")
        plt.subplot(1, 3, 2)
        plt.imshow(prepare_display(tgt_img))
        plt.title("Target (Radiometrically Corrected)")
        plt.subplot(1, 3, 3)
        plt.imshow(prepare_display(aligned_img))
        plt.title("Aligned Image")
        plt.show()
    
    return transform, aligned_img, rmse

# -------------------------------
# Noise Reduction (Task 3)
# -------------------------------
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

# Task 6: New noise reduction implementation using parallel processing for each band.
def noise_reduction_filter_parallel(image, method='median', kernel_size=3, sigma=1.0):
    """
    Task 6: Parallel noise reduction filter using ThreadPoolExecutor for each band.
    
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


# -------------------------------
# NDVI Analysis (Task 5)
# -------------------------------
def compute_ndvi(image, nir_band_index, red_band_index):
    """
    Compute the Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Parameters:
      image (np.ndarray): Input image (H, W, bands), assumed in [0,1].
      nir_band_index (int): Index for NIR.
      red_band_index (int): Index for Red.
      
    Returns:
      np.ndarray: NDVI image.
    """
    nir = image[..., nir_band_index].astype(np.float32)
    red = image[..., red_band_index].astype(np.float32)
    return (nir - red) / (nir + red + 1e-10)

def validate_ndvi_improvement(original_image, filtered_image, nir_band_index, red_band_index):
    """
    Validate noise reduction effect by computing the RMSE between NDVI images.
    
    Parameters:
      original_image (np.ndarray): Original image (H, W, bands).
      filtered_image (np.ndarray): Noise-reduced image (H, W, bands).
      nir_band_index (int): Index for NIR.
      red_band_index (int): Index for Red.
      
    Returns:
      float: NDVI RMSE.
    """
    ndvi_orig = compute_ndvi(original_image, nir_band_index, red_band_index)
    ndvi_filt = compute_ndvi(filtered_image, nir_band_index, red_band_index)
    return np.sqrt(np.mean((ndvi_orig - ndvi_filt) ** 2))

# ----------------------------------------------
# Unified Pipeline: Combine Tasks 1, 2, 3 and 5
# ----------------------------------------------
def full_image_processing_pipeline(ref_image_path, target_image_path, radiometric_params, detector_type,
                                   noise_method, noise_kernel_size, sigma, nir_band_index, red_band_index,
                                   visualize=False, use_parallel_noise_reduction=False):
    """
    Unified pipeline that executes:
      1. Radiometric and geometric correction.
      2. Noise reduction.
      3. NDVI computation and validation.
    
    Parameters:
      ref_image_path (str): Path to the reference image.
      target_image_path (str): Path to the target image.
      radiometric_params (dict): Radiometric parameters.
      detector_type (str): Feature detector type (e.g., 'ORB').
      noise_method (str): 'median' or 'gaussian'.
      noise_kernel_size (int): Kernel size for noise filtering.
      sigma (float): Sigma for Gaussian filtering.
      nir_band_index (int): Index for NIR band.
      red_band_index (int): Index for Red band.
      visualize (bool): If True, display intermediate results.
      use_parallel_noise_reduction (bool): If True, use the parallel noise reduction filter.
    
    Returns:
      tuple: (transform, aligned image, noise-reduced image, geometric RMSE, NDVI RMSE)
    """
    # Step 1: Radiometric and geometric correction.
    transform, aligned_img, geo_rmse = geometric_correction_pipeline(ref_image_path, target_image_path,
                                                                       radiometric_params, detector_type, visualize)
    
    # Step 2: Apply noise reduction to the aligned image.
    if use_parallel_noise_reduction:
        # Task 6: Using the parallel noise reduction filter.
        noise_reduced_img = noise_reduction_filter_parallel(aligned_img, method=noise_method,
                                                            kernel_size=noise_kernel_size, sigma=sigma)
    else:
        noise_reduced_img = noise_reduction_filter(aligned_img, method=noise_method,
                                                   kernel_size=noise_kernel_size, sigma=sigma)
    
    # Step 3: Compute NDVI before and after noise reduction.
    ndvi_aligned = compute_ndvi(aligned_img, nir_band_index, red_band_index)
    ndvi_noise_reduced = compute_ndvi(noise_reduced_img, nir_band_index, red_band_index)
    ndvi_rmse = np.sqrt(np.mean((ndvi_aligned - ndvi_noise_reduced) ** 2))
    
    if visualize:
        # Display NDVI images for comparison.
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(ndvi_aligned, cmap='RdYlGn')
        plt.title("Aligned Image NDVI")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(ndvi_noise_reduced, cmap='RdYlGn')
        plt.title("Noise-Reduced NDVI")
        plt.colorbar()
        plt.show()
    
    return transform, aligned_img, noise_reduced_img, geo_rmse, ndvi_rmse