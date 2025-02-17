import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

#########################################
# --- JSON Dataset Definition (Same as Training)
#########################################

class JSONBPSTransformationDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data_list = json.load(f)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        diff_image = np.array(sample["diff"], dtype=np.float32)
        target = np.array(sample["target"], dtype=np.float32)
        return torch.from_numpy(diff_image), torch.from_numpy(target)

#########################################
# --- Model Definition (Same as Training)
#########################################

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Pool only along height, keeping width unchanged
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Again, pool only along height
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Pool only along height
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))  # Pool only along height
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 64 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

#########################################
# --- Testing Function
#########################################

def test_model(model_path, test_json_file, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load dataset
    test_dataset = JSONBPSTransformationDataset(test_json_file)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load model
    model = SimpleCNN(input_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_quaternion_loss = 0.0
    total_translation_loss = 0.0
    total_scaling_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, desc="Testing Model"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            quaternion_loss = criterion(outputs[:, :4], targets[:, :4])
            translation_loss = criterion(outputs[:, 4:7], targets[:, 4:7])
            scaling_loss = criterion(outputs[:, 7], targets[:, 7])
            
            total_loss += loss.item() * inputs.size(0)
            total_quaternion_loss += quaternion_loss.item() * inputs.size(0)
            total_translation_loss += translation_loss.item() * inputs.size(0)
            total_scaling_loss += scaling_loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
    
    avg_loss = total_loss / num_samples
    avg_quaternion_loss = total_quaternion_loss / num_samples
    avg_translation_loss = total_translation_loss / num_samples
    avg_scaling_loss = total_scaling_loss / num_samples
    
    print(f"Test MSE Loss: {avg_loss:.6f}")
    print(f"Avg Quaternion Loss: {avg_quaternion_loss:.6f}")
    print(f"Avg Translation Loss: {avg_translation_loss:.6f}")
    print(f"Avg Scaling Loss: {avg_scaling_loss:.6f}")
    
    return avg_loss, avg_quaternion_loss, avg_translation_loss, avg_scaling_loss

if __name__ == "__main__":
    model_path = "/home/khanm/workfolder/registration_mk/registration/saved_model/pytorch_transformation_model_from_json.pth"
    test_json_file = "/home/khanm/workfolder/registration_mk/registration/data/bps_dataset_test.json"  # Provide the correct path to your test JSON dataset
    test_model(model_path, test_json_file)
