import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc
from model.RegNet import RegNet  # Import the new model
from model.configs_RegNet import get_2DRegNet_config 
config = get_2DRegNet_config()

#########################################
# --- JSON Dataset Definition
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
# --- Training Loop
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Training parameters.
    json_file = "/home/khanm/workfolder/registration_mk/registration/data/bps_dataset.json"   # Path to your JSON file with pre-generated data.
    batch_size = 64
    num_epochs = 1000
    learning_rate = 0.001
    
    # Create dataset and DataLoader.
    dataset = JSONBPSTransformationDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # For demonstration, inspect one sample.
    sample_img, sample_target = dataset[0]
    print(f"Sample image shape (C, H, W): {sample_img.shape}")  # e.g., (1, 1024, 4)
    print(f"Sample target shape: {sample_target.shape}")          # e.g., (8,)
    
    # Load model
    model = RegNet(config).to(device)
    model.train()
    
    # Define loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop.
    for epoch in range(num_epochs):
        running_loss = 0.0
        tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (inputs, targets) in tqdm_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                tqdm_bar.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0.0
        
        # Optionally, force garbage collection between epochs.
        gc.collect()
    
    # Save the trained model.
    torch.save(model.state_dict(), "/home/khanm/workfolder/registration_mk/registration/saved_model/pytorch_transformation_model_from_json.pth")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_model()
