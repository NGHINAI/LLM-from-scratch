import os

def truncate_file(file_path, desired_size_mb):
    # Convert MB to bytes (1 MB = 1024 * 1024 bytes)
    desired_size_bytes = (desired_size_mb * 1024 * 1024)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    # Get the current size of the file
    current_size = os.path.getsize(file_path)
    print(f"Current file size: {current_size / (1024 * 1024):.2f} MB")

    # If the file is larger than the desired size, truncate it
    if current_size > desired_size_bytes:
        with open(file_path, 'r+b') as f:
            f.truncate(desired_size_bytes)
            print(f"File truncated to {desired_size_mb} MB.")
    else:
        print("No truncation needed. File is already smaller than or equal to the desired size.")

# Example usage
file_path = '../datasets/processed_instruction_dataset_sample.json'
desired_size_mb = 20  # Desired size in MB
truncate_file(file_path, desired_size_mb)
