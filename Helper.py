import json
from pathlib import Path
import shutil
import os


TRANSACTION_FILE = Path(__file__).parent / "resources" / "transaction_data.json"


def copy_file_from_minimon(filename):
    """
    Copies a file from the MiniMon resources folder to the detect/resources folder.
    
    Args:
        filename (str): The name of the file to copy.

    Returns:
        bool: True if the file was copied successfully, False otherwise.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    destination_dir = os.path.join(script_dir, "resources")
    os.makedirs(destination_dir, exist_ok=True)
    
    parent_dir = os.path.dirname(script_dir)
    source_dir = os.path.join(parent_dir, "MiniMon", "resources")
    
    source_file_path = os.path.join(source_dir, filename)
    destination_file_path = os.path.join(destination_dir, filename)
    
    if not os.path.exists(source_file_path):
        print(f"Error: Could not find '{filename}' inside '{source_dir}'.")
        return False
        
    try:
        shutil.copy2(source_file_path, destination_file_path)
        print(f"Success! Copied '{filename}' into the 'detect/resources' folder.")

        return True
    
    except Exception as e:
        print(f"An error occurred while copying: {e}")

        return False

def transform_data():
    with open(TRANSACTION_FILE) as f:
        data = json.load(f)
    
    with open(TRANSACTION_FILE, "w") as f:
        json.dump(data["data"], f, indent=4)
        
    print("Data transformation complete. The 'data' key has been removed from the JSON file.")
    return

if __name__ == "__main__":
    # copy_file_from_minimon("transaction_data.json")
    transform_data()
