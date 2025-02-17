import os
import h5py
import numpy as np


def read_off(file_path):
    """Reads a .off file and returns vertices and faces."""
    with open(file_path, "r") as file:
        lines = file.readlines()
        if lines[0].strip() != "OFF":
            raise ValueError("Not a valid OFF file.")

        # Get the counts of vertices and faces
        counts = lines[1].strip().split()
        num_vertices = int(counts[0])
        num_faces = int(counts[1])

        # Read vertices
        vertices = []
        for i in range(2, 2 + num_vertices):
            vertex = list(map(float, lines[i].strip().split()))
            vertices.append(vertex)

        # Read faces (optional if needed)
        faces = []
        for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
            face = list(map(int, lines[i].strip().split()[1:]))
            faces.append(face)

        return np.array(vertices), np.array(faces)


def convert_off_to_h5(input_dir, output_dir):
    """Converts .off files in input_dir to .h5 files in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".off"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace(".off", ".h5"))

            try:
                vertices, faces = read_off(input_path)

                # Save as .h5 file
                with h5py.File(output_path, "w") as h5_file:
                    h5_file.create_dataset("data", data=vertices)
                    h5_file.create_dataset("label", data=np.array([0]))  # Dummy label, adjust as needed
                print(f"Converted: {file_name} -> {output_path}")

            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")


if __name__ == "__main__":
    # Input directory containing .off files
    input_directory = "/home/khanm/workfolder/bps/bps_demos/data/modelnet40_ply_hdf5_2048/"  # Replace with the actual path

    # Output directory to save .h5 files
    output_directory = "/home/khanm/workfolder/bps/data/modelnet40_ply_hdf5_2048/"  # Replace with the actual path

    convert_off_to_h5(input_directory, output_directory)
