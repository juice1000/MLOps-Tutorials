import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Simulate a large dataset (e.g., a CSV file with 10 million rows)
num_rows = 10_000_000
df = pd.DataFrame(
    {"col1": np.random.randint(0, 100, num_rows), "col2": np.random.randn(num_rows)}
)


# Function to process a batch of data
def process_batch(batch):
    """Dummy function that processes a batch of data."""
    print("Processing a batch of data...")
    batch["col3"] = batch["col1"] * batch["col2"]  # Example transformation
    return batch


# Define batch size
batch_size = 100_000  # Adjust based on memory and performance

# Split data into chunks
num_batches = len(df) // batch_size + (len(df) % batch_size != 0)
batches = [df.iloc[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

# Parallel processing
processed_batches = Parallel(n_jobs=num_cores)(
    delayed(process_batch)(batch) for batch in batches
)

# Concatenate results back into a DataFrame
df_processed = pd.concat(processed_batches, ignore_index=True)

# Show results
print(df_processed.head())
