# distributed_pipeline.py
import dcp
dcp.init()  # Initialize DCP

from full_pipeline_module import full_image_processing_pipeline

def work_function(datum):
    """
    Remote work function that processes one image pair.
    
    Expected datum is a tuple:
      (ref_image_path, target_image_path, radiometric_params, detector_type,
       noise_method, noise_kernel_size, sigma, nir_band_index, red_band_index,
       use_parallel_noise_reduction)
    """
    # Ensure progress is tracked on the remote worker
    dcp.progress()
    
    # Unpack the input datum
    (ref_image_path, target_image_path, radiometric_params, detector_type,
     noise_method, noise_kernel_size, sigma, nir_band_index, red_band_index,
     use_parallel_noise_reduction) = datum
    
    # Execute the full processing pipeline (disable visualization for remote execution)
    results = full_image_processing_pipeline(
        ref_image_path, target_image_path, radiometric_params, detector_type,
        noise_method, noise_kernel_size, sigma, nir_band_index, red_band_index,
        visualize=False, use_parallel_noise_reduction=use_parallel_noise_reduction
    )
    # results is a tuple: (transform, aligned_img, noise_reduced_img, geo_rmse, ndvi_rmse)
    return results
