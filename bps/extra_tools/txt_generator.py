import os

def list_files_in_directory(input_dir, output_dir, output_file_name="file_list.txt"):
    """
    Lists all files in a given directory and writes their names (with extensions) to a text file in the output directory.
    
    Parameters:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory where the file list will be saved.
        output_file_name (str): Name of the output text file. Defaults to "file_list.txt".
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input directory '{input_dir}' does not exist.")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in the input directory
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # Output file path
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Write file names to the output file
    with open(output_file_path, "w") as file:
        for f in files:
            file.write(f"{f}\n")
    
    print(f"File list saved to: {output_file_path}")


# Example usage
if __name__ == "__main__":
    input_directory = input("Enter the path to the input directory: ")
    output_directory = input("Enter the path to the output directory: ")
    list_files_in_directory(input_directory, output_directory)
