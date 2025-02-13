import os
import glob
import dcp
dcp.init()  # Initialize DCP

from distributed_pipeline import work_function

def get_image_pairs(ref_folder, target_folder, pattern="*.tif"):
    """
    List and pair image files from two folders.
    Assumes files are ordered such that the nth file in ref_folder 
    corresponds to the nth file in target_folder.
    """
    ref_files = sorted(glob.glob(os.path.join(ref_folder, pattern)))
    target_files = sorted(glob.glob(os.path.join(target_folder, pattern)))
    if len(ref_files) != len(target_files):
        raise ValueError("The number of files in the reference and target folders do not match.")
    return list(zip(ref_files, target_files))

# Define the folder paths
ref_folder = "testfiles/ref_folder"
target_folder = "testfiles/target_folder"

# Get paired image paths from the folders
pairs_files = get_image_pairs(ref_folder, target_folder, pattern="*.tif")

# Define radiometric and constant processing parameters.
radiometric_params = {
    'gain': [0.1, 0.11, 0.09, 0.1, 0.1],
    'offset': [1, 1, 1, 1, 1],
    'sunelev': 60.0,
    'edist': 1.0,
    'Esun': [1800, 1700, 1600, 1500, 1400],
    'blackadjust': 0.01,
    'percentile': 0.1
}

detector_type = "ORB"
noise_method = "median"
noise_kernel_size = 3
sigma = 1.0
nir_band_index = 4
red_band_index = 2
use_parallel_noise_reduction = False

# Build the list of data tuples for each image pair.
data = []
for ref_img, target_img in pairs_files:
    data.append((
        ref_img,                   # Reference image path
        target_img,                # Target image path
        radiometric_params,        # Radiometric parameters
        detector_type,             # Detector type
        noise_method,              # Noise reduction method
        noise_kernel_size,         # Kernel size
        sigma,                     # Sigma for Gaussian filtering
        nir_band_index,            # NIR band index
        red_band_index,            # Red band index
        use_parallel_noise_reduction  # Parallel noise reduction flag
    ))

# Create a batch job for all the image pairs using DCP.
job = dcp.compute_for(data, work_function)

job.on('readystatechange', print)
job.on('accepted', lambda: print(f"Batch job accepted with id: {job.id}"))

@job.on('result')
def on_result(event):
    print(f'New result for slice {event.sliceNumber}:')
    print(event.result)

# Deploy the compute workload to the DCP network.
job.exec()

# Wait for all jobs to complete and then print the results.
results = job.wait()
print("Batch Results:", results)

