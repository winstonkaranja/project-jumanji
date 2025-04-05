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
import tifffile as tiff
import numpy as np
from io import BytesIO




# -------------------------------
# Radiometric & Geometric Correction (Tasks 1 & 2)
# -------------------------------

def read_tiff_from_s3(bucket_name, key):
    # Initialize S3 client
    s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
    region_name=os.environ.get('AWS_REGION')
  )

    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    
    # Read the data using tifffile from the response body
    image_data = response['Body'].read()
    
    # Create a file-like object
    file_obj = BytesIO(image_data)
    
    # Read the TIFF using tifffile
    image = tiff.imread(file_obj).astype(np.float32)
    
    return image



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
      image_path (str): Path to the multispectral image.
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
        raise ValueError(f"Could not read image!")
    
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



def geometric_correction_pipeline(image, radiometric_params, detector_type='ORB', visualize=False):
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

    tgt_corr = rigorous_radiometric_correction(image, **radiometric_params)
    
    # Transpose to (H, W, bands) for geometric processing

    tgt_img = np.transpose(tgt_corr, (1, 2, 0))
    

    
    if visualize:
        def prepare_display(img):
            # Use only the first 3 bands for RGB visualization if available.
            return img[..., :3] if img.shape[2] >= 3 else img
        plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(prepare_display(ref_img))
        # plt.title("Reference (Radiometrically Corrected)")
        plt.subplot(1, 3, 2)
        plt.imshow(prepare_display(tgt_img))
        plt.title("Target (Radiometrically Corrected)")
        # plt.subplot(1, 3, 3)
        # plt.imshow(prepare_display(aligned_img))
        # plt.title("Aligned Image")
        plt.show()
    
    return tgt_img

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
def compute_ndvi(image, red_band_index, nir_band_index):
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



# ----------------------------------------------
# Unified Pipeline: Combine Tasks 1, 2, 3 and 5
# ----------------------------------------------
def full_image_processing_pipeline(image, radiometric_params, detector_type,
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
      tuple: (noise-reduced image, ndvi_rmse)
    """
    # Step 1: Radiometric and geometric correction.
    aligned_img = geometric_correction_pipeline(image, radiometric_params, detector_type, visualize)
    
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
    
    return ndvi_noise_reduced, ndvi_rmse



def estimate_biomass(ndvi, a=2.5, b=1.2):
    """
    Estimate Above-Ground Biomass (AGB) using an empirical NDVI-to-biomass model.
    
    Parameters:
      ndvi (np.ndarray): NDVI image.
      a (float): Biomass coefficient (default: 5).
      b (float): NDVI exponent factor (default: 1.2).
      
    Returns:
      np.ndarray: Estimated biomass per pixel.
    """
    return a * np.exp(b * ndvi)  # Biomass in kg/m²

def calculate_carbon_storage(biomass, carbon_fraction=0.47):
    """
    Convert biomass to stored carbon using IPCC carbon fraction.
    
    Parameters:
      biomass (np.ndarray): Biomass per pixel (kg/m²).
      carbon_fraction (float): Fraction of biomass that is carbon (default: 0.47).
      from compute import process_ndvi_for_carbon_credits
    Returns:
      np.ndarray: Carbon stored per pixel (kg/m²).
    """
    return biomass * carbon_fraction

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

# ==== AUTOMATION PIPELINE ====
def process_ndvi_for_carbon_credits(ndvi_noise_reduced, pixel_area_m2):
    """
    Full pipeline to process NDVI into carbon credits.
    
    Parameters:
      ndvi_noise_reduced (np.ndarray): NDVI image.
      pixel_area_m2 (float): Area of each pixel in square meters.
      
    Returns:
      float: Total carbon credits for the farm.
    """
    biomass = estimate_biomass(ndvi_noise_reduced)  # Step 1: Estimate biomass
    carbon_storage = calculate_carbon_storage(biomass)  # Step 2: Convert to carbon storage
    co2_sequestration = convert_carbon_to_co2(carbon_storage)  # Step 3: Convert to CO₂ equivalent
    carbon_credits = calculate_carbon_credits(co2_sequestration, pixel_area_m2)  # Step 4: Calculate credits
    
    return carbon_credits