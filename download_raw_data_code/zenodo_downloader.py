from zenodo_get import download
import os

# List of Zenodo record IDs
record_ids = ["15056046", "17425252", "4751204","6481135", "7446728", "14178056", "10404966"]

base_output_dir = "/mnt/files_copied_to_mn5"

for rid in record_ids:
    # Create a subdirectory for this record
    output_dir = os.path.join(base_output_dir, rid)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {rid} into {output_dir}...", flush=True)
    try:
        download(rid, output_dir=output_dir)
        print(f"Finished {rid}\n", flush=True)
    except Exception as e:
        print(f"Failed to download {rid}: {e}\n", flush=True)
