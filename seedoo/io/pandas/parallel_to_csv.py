import pandas as pd
import multiprocessing
from multiprocessing import shared_memory
from io import StringIO

def _write_to_shared_memory(df_chunk, shm_name, size):
    """Write a chunk of DataFrame to shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    csv_str = df_chunk.to_csv(index=False, header=False)
    encoded_csv = csv_str.encode() + b'\0'  # Adding a null terminator
    shm.buf[:len(encoded_csv)] = encoded_csv
    shm.close()

def to_csv_parallel(df):
    num_processes = multiprocessing.cpu_count() - 1 or 1

    # Estimate the required size per chunk
    sample_csv = df.head(3).to_csv(index=False)
    estimated_size_per_chunk = int(len(sample_csv.encode()) / 3) * len(df) // num_processes
    estimated_size_per_chunk = int(estimated_size_per_chunk * 1.2)  # Adding a 20% buffer

    chunk_size = len(df) // num_processes
    num_chunks = len(df) // chunk_size

    # Split DataFrame into chunks and create shared memory for each chunk
    shared_memories = []
    processes = []

    for i in range(0, len(df) + chunk_size, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        if len(chunk) > 0:
            shm = shared_memory.SharedMemory(create=True, size=estimated_size_per_chunk)
            shared_memories.append(shm)
            processes.append(multiprocessing.Process(target=_write_to_shared_memory, args=(chunk, shm.name, estimated_size_per_chunk)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    # Combine data from all shared memories
    csv_string_io = StringIO()
    for shm in shared_memories:
        chunk_csv = shm.buf.tobytes().split(b'\0', 1)[0].decode()  # Read up to null terminator
        csv_string_io.write(chunk_csv)
        shm.close()
        shm.unlink()

    # Create StringIO from combined CSV data
    csv_string_io.seek(0)

    return csv_string_io


import pandas as pd
import numpy as np

def run_test_to_csv_parallel():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.random.randint(1, 100, 100)
    })

    # Convert DataFrame to CSV using the parallel function
    csv_string_io_parallel = to_csv_parallel(df)

    # Convert DataFrame to CSV using the standard method
    csv_standard = df.to_csv(index=False, header=False)

    # Read the content from the StringIO object
    csv_parallel = csv_string_io_parallel.getvalue()

    # Compare the results (ignoring potential differences in floating-point representation)
    assert all(line1.split(',') == line2.split(',') for line1, line2 in zip(csv_standard.splitlines(), csv_parallel.splitlines()))

    print("Test passed: Parallel CSV conversion matches standard conversion.")

# Run the test
if __name__ == "__main__":
    run_test_to_csv_parallel()
