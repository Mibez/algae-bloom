from datetime import datetime, timedelta
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import numpy as np
import glob

# --- CONFIGURATION ---
RAW_DATA_FOLDER = "raw_data_ml"
ALGAE_FOLDER = "prediction_data/algae/"
SST_FOLDER = "prediction_data/sst/"
PREDICTION_FOLDER = "prediction_data/prediction/"
MASK_CACHE_FILENAME = "land_mask_predict.png"
MODEL_FILENAME = "algae_predictor_unet.pth"
# Hyperparameters
# REDUCED BATCH SIZE for lower peak GPU memory usage
BATCH_SIZE = 1
# Gradient Accumulation: Wait for this many forward/backward passes before updating weights.
# Effective Batch Size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 0.001
EPOCHS = 5
IMAGE_SIZE = (768, 445) 

# Colors & Thresholds for Gap Filling
LAND_COLOR = (50, 50, 50)
SEA_FALLBACK_COLOR = (77, 124, 166)
CLOUD_BRIGHTNESS_THRESHOLD = 230

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


class AlgaeTimeSeriesDataset(Dataset):
    def __init__(self, algae_dir, sst_dir, mask_path):
        self.algae_files = sorted(glob.glob(os.path.join(algae_dir, "*.png")))
        self.sst_dir = sst_dir
        
        # Load Land Mask
        if os.path.exists(mask_path):
            self.land_mask = Image.open(mask_path).convert('1').resize(IMAGE_SIZE)
            self.land_mask_np = np.array(self.land_mask)
        else:
            print("Warning: Land mask not found. Assuming all sea.")
            self.land_mask = None
            self.land_mask_np = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)

        # Build valid pairs list
        self.valid_indices = []
        for i in range(len(self.algae_files) - 1):
            curr_file = self.algae_files[i]
            
            date_name = os.path.basename(curr_file)
            sst_path = os.path.join(self.sst_dir, date_name)
            
            if os.path.exists(sst_path):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        path_curr = self.algae_files[i]
        path_next = self.algae_files[i+1]
        
        date_name = os.path.basename(path_curr)
        path_sst = os.path.join(self.sst_dir, date_name)

        img_algae_curr = Image.open(path_curr).convert('RGB').resize(IMAGE_SIZE)
        img_algae_next = Image.open(path_next).convert('RGB').resize(IMAGE_SIZE)
        img_sst = Image.open(path_sst).convert('RGB').resize(IMAGE_SIZE)


        # --- TENSOR CONVERSION ---

        np_algae = np.array(img_algae_curr).astype(np.float32) / 255.0
        np_sst = np.array(img_sst).astype(np.float32) / 255.0
        np_target = np.array(img_algae_next).astype(np.float32) / 255.0

        # Masking Land (Force to 0)
        mask_3d = self.land_mask_np[..., None]
        np_algae[mask_3d.repeat(3, axis=2)] = 0
        np_sst[mask_3d.repeat(3, axis=2)] = 0
        np_target[mask_3d.repeat(3, axis=2)] = 0

        input_combined = np.concatenate((np_algae, np_sst), axis=2)
        
        input_tensor = torch.from_numpy(input_combined.transpose((2, 0, 1)))
        target_tensor = torch.from_numpy(np_target.transpose((2, 0, 1)))

        return input_tensor, target_tensor

# --- 3. MODEL ARCHITECTURE (Simple U-Net) ---
# (Architecture remains unchanged as it doesn't affect memory)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(6, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        
        d3 = self.up3(b)
        if d3.size() != e3.size(): d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.size() != e2.size(): d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.size() != e1.size(): d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.sigmoid(self.final(d1))

# --- 4. TRAINING LOOP ---

def train_model():
    full_dataset = AlgaeTimeSeriesDataset(ALGAE_FOLDER, SST_FOLDER, MASK_CACHE_FILENAME)
    
    if len(full_dataset) == 0:
        print("No valid image pairs found. Run the fetch scripts first.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # DataLoader uses the smaller BATCH_SIZE (e.g., 2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleUNet().to(device)
    criterion = nn.MSELoss()
    criterion_val = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize gradient zeroing
    optimizer.zero_grad() 

    print(f"Starting training on {len(train_dataset)} pairs for {EPOCHS} epochs with Effective Batch Size = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # We iterate over the small batches (size 2)
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Scale the loss down by the accumulation factor
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backpropagation (computes gradients, doesn't update weights yet)
            loss.backward()
            
            # Multiply back to log the full unscaled loss
            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # Perform the optimization step only when the accumulation count is reached
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()       # Update weights
                optimizer.zero_grad()  # Reset gradients for the next large batch
        
        # Handle the case where the last mini-batch didn't complete the accumulation cycle
        if (len(train_loader) % GRADIENT_ACCUMULATION_STEPS) != 0:
             optimizer.step()
             optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), MODEL_FILENAME)

    print("Model saved as algae_predictor_unet.pth")

# --- 5. INFERENCE ---

def predict_sample(model_path, algae_img_path, sst_img_path):
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if os.path.exists(MASK_CACHE_FILENAME):
        mask_img = Image.open(MASK_CACHE_FILENAME).convert('1').resize(IMAGE_SIZE)
        mask_np = np.array(mask_img)
    else:
        mask_img = None
        mask_np = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)

    img_algae = Image.open(algae_img_path).convert('RGB').resize(IMAGE_SIZE)
    img_sst = Image.open(sst_img_path).convert('RGB').resize(IMAGE_SIZE)
    

    np_algae = np.array(img_algae).astype(np.float32) / 255.0
    np_sst = np.array(img_sst).astype(np.float32) / 255.0
    
    mask_3d = mask_np[..., None]
    np_algae[mask_3d.repeat(3, axis=2)] = 0
    np_sst[mask_3d.repeat(3, axis=2)] = 0

    input_combined = np.concatenate((np_algae, np_sst), axis=2)
    input_tensor = torch.from_numpy(input_combined.transpose((2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output_np = output.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    output_np = (output_np * 255).astype(np.uint8)
    
    res_img = Image.fromarray(output_np)

    # res_img.save("ml_prediction_result.png")PREDICTION_FOLDER
    date = algae_img_path.split("/")[-1].split(".")[0]
    date = datetime.strptime(date, "%Y-%m-%d")
    date = date + timedelta(days=1)

    os.makedirs(PREDICTION_FOLDER, exist_ok=True)

    res_img.save(f"{PREDICTION_FOLDER}/{date}.png")
    print("Prediction saved as ml_prediction_result.png")

if __name__ == "__main__":
    if not os.path.exists(ALGAE_FOLDER):
        print("Raw data not found.")
    else:
        if not os.path.exists(MODEL_FILENAME):
            train_model()

        files = sorted(glob.glob(os.path.join(ALGAE_FOLDER, "*.png")))
        if len(files) > 0:
            sample_algae = files[-50]
            sample_sst = os.path.join(SST_FOLDER, os.path.basename(sample_algae))
            if os.path.exists(sample_sst):
                print(f"Running test prediction on {sample_sst}")
                predict_sample(MODEL_FILENAME, sample_algae, sample_sst)