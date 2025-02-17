import os
import random
import numpy as np

def load_off(file_path):
    """Load an .off file and return its points and faces."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError(f"File {file_path} is not a valid OFF file.")

    # Read the number of vertices and faces
    n_vertices, n_faces, _ = map(int, lines[1].strip().split())

    # Load vertices
    vertices = np.array([
        list(map(float, lines[i + 2].strip().split())) for i in range(n_vertices)
    ])

    # Load faces (not used for this task, but included for completeness)
    faces = [
        list(map(int, lines[i + 2 + n_vertices].strip().split()[1:]))
        for i in range(n_faces)
    ]

    return vertices, faces

def save_off(file_path, vertices, faces):
    """Save points and faces to an .off file."""
    with open(file_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for vertex in vertices:
            f.write(f"{' '.join(map(str, vertex))}\n")
        for face in faces:
            f.write(f"3 {' '.join(map(str, face))}\n")

def process_off_files(input_dir):
    """Process all .off files in the given directory."""
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.off'):
            file_path = os.path.join(input_dir, file_name)
            try:
                vertices, faces = load_off(file_path)

                if len(vertices) > 2048:
                    # Randomly sample 2048 points
                    sampled_indices = random.sample(range(len(vertices)), 2048)
                    vertices = vertices[sampled_indices]

                # Save updated .off file
                save_off(file_path, vertices, faces)
                print(f"Processed and updated file: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_name} due to error.")
                except Exception as delete_error:
                    print(f"Failed to delete file {file_name}: {delete_error}")

if __name__ == "__main__":
    input_directory = input("Enter the path to the directory containing .off files: ")

    if not os.path.isdir(input_directory):
        print("Invalid directory path.")
    else:
        process_off_files(input_directory)
        print("Processing complete.")
