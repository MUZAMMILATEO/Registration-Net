import sys
sys.path.append('/home/khanm/workfolder/bps/')  # Add BPS path to Python's sys.path

import json
import numpy as np
import torch
from bps import bps
import gc
from bps import bps  # assuming your bps module is in PYTHONPATH or sys.path
import open3d as o3d
import copy
import random
from scipy.spatial.transform import Rotation as R

def reshape_pairwise_diff_to_grid(diff_vector, patch_len=16):
    diff_vector = np.squeeze(diff_vector, axis=0)
    n_total = len(diff_vector)
    if n_total % patch_len != 0:
        raise ValueError("Length of diff_vector must be a multiple of patch_len.")
    
    patch_side = int(np.sqrt(patch_len))
    num_patches = n_total // patch_len
    patches = []
    for i in range(num_patches):
        patch = diff_vector[i*patch_len:(i+1)*patch_len].reshape(patch_side, patch_side)
        patches.append(patch)
    
    block = []
    i = 0
    while i < len(patches):
        if i == 0:
            block = np.concatenate((patches[i], patches[i+1]), axis=0)
            i += 2
        else:
            block = np.concatenate((block, patches[i]), axis=0)
            i += 1
    block = np.expand_dims(block, axis=0)
    return block

def extract_local_region(source_pcd, region_proportion=0.2):
    points = np.asarray(source_pcd.points)
    num_points = len(points)
    if num_points < 10:
        raise ValueError("Point cloud has too few points for region selection.")
    
    seed_idx = np.random.randint(num_points)
    seed_point = points[seed_idx]
    distances = np.linalg.norm(points - seed_point, axis=1)
    num_selected = int(region_proportion * num_points)
    nearest_indices = np.argsort(distances)[:num_selected]
    
    extracted_pcd = o3d.geometry.PointCloud()
    extracted_pcd.points = o3d.utility.Vector3dVector(points[nearest_indices])
    return extracted_pcd

def apply_local_deformations(mesh, deformation_strength=0.1, fold_intensity=0.2):
    vertices = np.asarray(mesh.vertices)
    pull_axis = np.array([0, 1, 0])
    distances = np.linalg.norm(vertices, axis=1)
    vertices += deformation_strength * (distances[:, None] * pull_axis)
    bend_effect = np.sin(vertices[:, 0] * np.pi * 2) * fold_intensity
    vertices[:, 1] += bend_effect
    folding_region = vertices[:, 0] > 0
    vertices[folding_region] *= 0.9
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

def generate_random_transformation():
    angles = np.radians(np.random.uniform(-45, 45, size=3))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(angles)
    quat = R.from_matrix(rotation_matrix).as_quat()
    translation_vector = np.random.uniform(-1, 1, size=3)
    scaling_factor = np.random.uniform(0.5, 2.0)
    return {
        "rotation": quat.tolist(),
        "translation": translation_vector.tolist(),
        "scaling": scaling_factor
    }, rotation_matrix, translation_vector, scaling_factor

# Define a modified Dataset that does NOT run in an infinite loop.
class BPSTransformationDatasetOffline:
    """
    This dataset generates the diff images and transformation parameters,
    then returns them as NumPy arrays.
    """
    def __init__(self, mesh_path, n_bps_points, dataset_size):
        self.mesh_path = mesh_path
        self.n_bps_points = n_bps_points
        self.dataset_size = dataset_size
        self.original_mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.basis_set = bps.compute_basis_set(n_bps_points=self.n_bps_points)
        self.mesh = copy.deepcopy(self.original_mesh)
        self.vertices = np.asarray(self.mesh.vertices)
        self.num_vertices = self.vertices.shape[0]
        self.indices = np.random.choice(self.num_vertices, size=1000, replace=False)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # We'll try only once per sample here.
        try:
            mesh = copy.deepcopy(self.original_mesh)
            vertices = np.asarray(mesh.vertices)
            point_cloud_OG = o3d.geometry.PointCloud()
            point_cloud_OG.points = o3d.utility.Vector3dVector(vertices[self.indices])
            
            deformation_strength = np.random.uniform(0.005, 0.025)
            fold_intensity = np.random.uniform(0.01, 0.06)
            # mesh = apply_local_deformations(mesh, deformation_strength, fold_intensity)
            vertices_deformed = np.asarray(mesh.vertices)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(vertices_deformed[self.indices])
            
            region_proportion = np.random.uniform(0.2, 0.6)
            extracted_region = extract_local_region(point_cloud, region_proportion)
            
            transform_params, rotation_matrix, translation_vector, scaling_factor = generate_random_transformation()
            target = np.array(transform_params["rotation"] + transform_params["translation"] + [transform_params["scaling"]],
                              dtype=np.float32)
            
            extracted_region.rotate(rotation_matrix, center=(0, 0, 0))
            extracted_region.translate(translation_vector)
            extracted_region.scale(scaling_factor, center=(0, 0, 0))
            
            extracted_region_bps = np.asarray(extracted_region.points)
            extracted_region_bps = np.expand_dims(extracted_region_bps, axis=0)
            extracted_region_bps = bps.normalize(extracted_region_bps)
            extracted_region_bps = bps.encode(
                extracted_region_bps,
                self.basis_set,
                bps_arrangement='random',
                n_bps_points=self.n_bps_points,
                bps_cell_type='dists'
            )
            
            point_cloud_bps = np.asarray(point_cloud_OG.points)
            point_cloud_bps = np.expand_dims(point_cloud_bps, axis=0)
            point_cloud_bps = bps.normalize(point_cloud_bps)
            point_cloud_bps = bps.encode(
                point_cloud_bps,
                self.basis_set,
                bps_arrangement='random',
                n_bps_points=self.n_bps_points,
                bps_cell_type='dists'
            )
            
            pairwise_diff = np.abs(extracted_region_bps - point_cloud_bps)
            diff_image = reshape_pairwise_diff_to_grid(pairwise_diff, patch_len=16)
            diff_image = np.squeeze(diff_image, axis=0)  # now (H, W)
            diff_image = np.expand_dims(diff_image, axis=0)  # (1, H, W)
            diff_image = diff_image - diff_image.min()
            diff_image = diff_image / (diff_image.max() - diff_image.min())
            
            # Cleanup (optional, since these objects will be garbage collected)
            del extracted_region, point_cloud, point_cloud_OG, mesh, vertices, vertices_deformed
            gc.collect()
            
            return diff_image, target  # Both are NumPy arrays.
        except Exception as e:
            raise RuntimeError(f"Error generating sample {idx}: {e}")

def generate_dataset_json(mesh_path, n_bps_points, dataset_size, output_file):
    dataset = BPSTransformationDatasetOffline(mesh_path, n_bps_points, dataset_size)
    data_list = []
    for idx in range(len(dataset)):
        try:
            diff_image, target = dataset[idx]
            # Convert arrays to lists so they can be stored in JSON.
            diff_image_list = diff_image.tolist()  # Shape: [1, H, W]
            target_list = target.tolist()            # Shape: [8]
            data_list.append({"diff": diff_image_list, "target": target_list})
            print(f"Processed sample {idx}")
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    with open(output_file, "w") as f:
        json.dump(data_list, f)
    print(f"Dataset saved to {output_file}")

if __name__ == '__main__':
    mesh_path = '/home/khanm/workfolder/registration_mk/bunny.ply'
    n_bps_points = 4096
    dataset_size = 100   # Adjust as needed.
    output_file = "/home/khanm/workfolder/registration_mk/registration/data/bps_dataset_test.json"
    generate_dataset_json(mesh_path, n_bps_points, dataset_size, output_file)
