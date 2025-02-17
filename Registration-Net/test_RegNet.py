import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model.RegNet import RegNet  # Import your model
from model.configs_RegNet import get_2DRegNet_config  # Import your config
import gc

#########################################
# --- JSON Dataset Definition (Same as Training)
#########################################
class JSONBPSTransformationDataset(Dataset):
    """
    Dataset that loads samples from a JSON file.
    Each sample is expected to be a dictionary with keys:
      - "diff": a list (representing a 3D array with shape [1, H, W])
      - "target": a list of 8 parameters.
    """
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data_list = json.load(f)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        # Convert the diff image and target vector from lists to numpy arrays.
        diff_image = np.array(sample["diff"], dtype=np.float32)  # expected shape: [1, H, W]
        target = np.array(sample["target"], dtype=np.float32)      # shape: [8]
        
        # Convert to PyTorch tensors.
        diff_image_tensor = torch.from_numpy(diff_image)
        target_tensor = torch.from_numpy(target)
        return diff_image_tensor, target_tensor

#########################################
# --- Test Script
#########################################
def test_model():
    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Specify path to your test JSON file.
    json_file = "/home/khanm/workfolder/registration_mk/registration/data/bps_dataset_test.json"
    
    # Create dataset and DataLoader.
    batch_size = 64
    dataset = JSONBPSTransformationDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load model configuration and instantiate model.
    config = get_2DRegNet_config()
    model = RegNet(config).to(device)
    
    # Run a dummy forward pass to ensure fc1 is instantiated.
    # Use an input that matches your expected input dimensions (e.g., [1, 1, 1024, 4])
    dummy_input = torch.randn(1, 1, 1024, 4, device=device)
    _ = model(dummy_input)
    
    # Load saved model weights.
    model_path = "/home/khanm/workfolder/registration_mk/registration/saved_model/pytorch_transformation_model_from_json.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    
    # Define loss function (same as training) for overall loss.
    criterion = nn.MSELoss()
    
    total_overall_loss = 0.0
    total_quat_loss = 0.0
    total_trans_loss = 0.0
    total_scale_loss = 0.0
    total_samples = 0
    
    # Disable gradient computation for inference.
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass.
            outputs = model(inputs)
            
            # Compute overall loss.
            overall_loss = criterion(outputs, targets)
            
            # Compute losses for each transformation component.
            # Quaternion: first 4 parameters.
            quat_loss = F.mse_loss(outputs[:, :4], targets[:, :4], reduction='mean')
            # Translation: next 3 parameters.
            trans_loss = F.mse_loss(outputs[:, 4:7], targets[:, 4:7], reduction='mean')
            # Scaling: last parameter.
            scale_loss = F.mse_loss(outputs[:, 7:], targets[:, 7:], reduction='mean')
            
            batch_size_curr = inputs.size(0)
            total_overall_loss += overall_loss.item() * batch_size_curr
            total_quat_loss += quat_loss.item() * batch_size_curr
            total_trans_loss += trans_loss.item() * batch_size_curr
            total_scale_loss += scale_loss.item() * batch_size_curr
            total_samples += batch_size_curr
            
            print(f"Batch {i+1}/{len(dataloader)} - Overall Loss: {overall_loss.item():.4f} | "
                  f"Quat Loss: {quat_loss.item():.4f} | Trans Loss: {trans_loss.item():.4f} | "
                  f"Scale Loss: {scale_loss.item():.4f}")
    
    avg_overall_loss = total_overall_loss / total_samples
    avg_quat_loss = total_quat_loss / total_samples
    avg_trans_loss = total_trans_loss / total_samples
    avg_scale_loss = total_scale_loss / total_samples

    print("\nAverage Losses on Test Set:")
    print(f"Overall Loss: {avg_overall_loss:.4f}")
    print(f"Quaternion Loss: {avg_quat_loss:.4f}")
    print(f"Translation Loss: {avg_trans_loss:.4f}")
    print(f"Scaling Loss: {avg_scale_loss:.4f}")
    
    # Optionally force garbage collection.
    gc.collect()

if __name__ == "__main__":
    test_model()
