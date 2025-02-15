# dcp_client_integration.py
import dcp
dcp.init()  # Initialize DCP

from distributed_pipeline import work_function

# Define the data for a single image pair as a tuple.
data = [
    (
        "testfiles/20180627_seq_50m_NC.tif",         # Reference image path
        "testfiles/20180627_seq_50m_NC copy.tif",    # Target image path
        {   # Radiometric parameters
            'gain': [0.012, 0.012, 0.012, 0.012, 0.012],
            'offset': [0, 0, 0, 0, 0],
            'sunelev': 60.0,
            'edist': 1.0,
            'Esun': [1913, 1822, 1557, 1317, 1074],
            'blackadjust': 0.01,
            'percentile': 1
        },
        "ORB",           # Detector type (not provided, kept as "ORB" placeholder)
        "median",        # Noise reduction method
        3,               # Kernel size
        1.0,             # Sigma for Gaussian (if applicable)
        4,               # NIR band index
        2,               # Red band index
        False            # Use parallel noise reduction flag
    )
]


# Define the compute workload (job) using dcp.compute_for.
job = dcp.compute_for(data, work_function)

# Add event listeners for progress and debugging.
job.on('readystatechange', print)
job.on('accepted', lambda: print(f"Job accepted with id: {job.id}"))

@job.on('result')
def on_result(event):
    print(f'New result for slice {event.sliceNumber}:')
    print(event.result)

# Deploy the compute workload to the DCP network.
job.exec()

# Wait for the compute workload to complete and process the results.
results = job.wait()
print("Results:", results)
