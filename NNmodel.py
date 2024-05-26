import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Dataset class to load the images and coordinates
class ReflectionProbesDataset(Dataset):
    def __init__(self, root_dir, coord_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.coordinates = []

        with open(coord_file, 'r') as file:
            for line in file:
                parts = line.strip().split(': ')
                probe_name = parts[0]
                coords = list(map(float, parts[1].split(', ')))
                self.coordinates.append(torch.tensor(coords))
                for i in range(6):
                    self.image_paths.append(os.path.join(root_dir, f"{probe_name}_face{i}.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        coord_idx = idx // 6  # There are 6 images per probe
        if self.transform:
            image = self.transform(image)
        return image, self.coordinates[coord_idx]

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create the dataset and dataloader
dataset = ReflectionProbesDataset(root_dir='./ReflectionProbes', coord_file='./ReflectionProbes/coordinates.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the CoordNet model
class CoordNet(nn.Module):
    def __init__(self):
        super(CoordNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128*128*3)  # Output a flattened image of shape 128x128x3

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 3, 128, 128)  # Reshape to (batch_size, 3, 128, 128)
        return x

# Define the ImageNet model
class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

# Instantiate the models
coord_net = CoordNet()
image_net = ImageNet()

# Training settings
learning_rate = 0.0005
num_epochs = 500

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer_coord = optim.Adam(coord_net.parameters(), lr=learning_rate)
optimizer_image = optim.Adam(image_net.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, coords in dataloader:
        # Train ImageNet
        optimizer_image.zero_grad()
        image_outputs = image_net(images)
        image_loss = criterion(image_outputs, images)
        image_loss.backward()
        optimizer_image.step()

        # Train CoordNet
        optimizer_coord.zero_grad()
        coord_outputs = coord_net(coords)
        # Ensure coord_outputs are in the right shape for ImageNet input
        synthetic_images = image_net(coord_outputs)
        coord_loss = criterion(synthetic_images, images)
        coord_loss.backward()
        optimizer_coord.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Image Loss: {image_loss.item():.12f}, Coord Loss: {coord_loss.item():.12f}')

# Save trained models
torch.save(coord_net.state_dict(), './coord_net.pth')
torch.save(image_net.state_dict(), './image_net.pth')

# Dummy inputs for ONNX export
dummy_input_coord = torch.randn(1, 3)  # Shape [1, 3] for CoordNet
dummy_input_image = torch.randn(1, 3, 128, 128)  # Shape [1, 3, 128, 128] for ImageNet

# Export models to ONNX
torch.onnx.export(coord_net, dummy_input_coord, './coord_net.onnx')
torch.onnx.export(image_net, dummy_input_image, './image_net.onnx')