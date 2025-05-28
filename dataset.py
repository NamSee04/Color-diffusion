import glob
import random
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops

from utils import load_default_configs, split_lab_channels


def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    You can use this to filter your dataset of black and white images 
    """
    if isinstance(im, str):
        im = Image.open(im).convert("RGB")
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsupported image mode")
    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


class CustomColorizationDataset(Dataset):
    def __init__(self, L_arrays, ab_arrays, split='train', config=None):
        """
        Args:
            L_arrays: List of numpy arrays of shape (224, 224) containing L channel
            ab_arrays: List of numpy arrays of shape (224, 224, 2) containing ab channels
            split: 'train' or 'val'
            config: Configuration dictionary
            device: 'cpu' or 'cuda' for GPU support
        """
        self.L_arrays = L_arrays
        self.ab_arrays = ab_arrays
        self.config = config
        self.split = split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Verify input shapes
        for L, ab in zip(L_arrays, ab_arrays):
            assert L.shape == (224, 224), f"L channel shape should be (224, 224), got {L.shape}"
            assert ab.shape == (224, 224, 2), f"ab channels shape should be (224, 224, 2), got {ab.shape}"
        
        # Normalization factors
        self.L_norm = 50.0  # L channel normalization
        self.ab_norm = 110.0  # ab channels normalization

    def normalize_lab(self, L, ab):
        """Normalize L and ab channels to [-1, 1] range and move to device"""
        L = torch.from_numpy(L).float() / self.L_norm - 1.0  # L: [-1, 1]
        ab = torch.from_numpy(ab).float() / self.ab_norm     # ab: [-1, 1]
        return L.to(self.device), ab.to(self.device)

    def __getitem__(self, idx):
        L = self.L_arrays[idx]
        ab = self.ab_arrays[idx]
        
        # Normalize channels
        L, ab = self.normalize_lab(L, ab)
        
        # Add channel dimension to L
        L = L.unsqueeze(0)  # Shape: (1, 224, 224)
        
        # Permute ab to channel first format
        ab = ab.permute(2, 0, 1)  # Shape: (2, 224, 224)
        
        # Combine L and ab channels
        lab = torch.cat([L, ab], dim=0)  # Shape: (3, 224, 224)
        
        return lab

    def __len__(self):
        return len(self.L_arrays)


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', config=None):
        size = config["img_size"]
        self.resize = transforms.Resize((size, size), Image.BICUBIC)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3,
                                       contrast=0.1,
                                       saturation=(1., 2.),
                                       hue=0.05),
                self.resize
            ])
        elif split == 'val':
            self.transforms = self.resize
        self.paths = paths

    def tensor_to_lab(self, base_img_tens):
        base_img = np.array(base_img_tens)
        img_lab = rgb2lab(base_img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return torch.cat((L, ab), dim=0).to(self.device)

    def get_lab_from_path(self, path):
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return self.tensor_to_lab(img)

    def get_rgb(self, idx=0):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        return (img)

    def get_grayscale(self, idx=0):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.resize(img)
        img = np.array(img)
        return (img)

    def get_lab_grayscale(self, idx=0):
        img = self.get_lab_from_path(self.paths[idx])
        l, _ = split_lab_channels(img.unsqueeze(0))
        return torch.cat((l, *[torch.zeros_like(l)] * 2), dim=1)

    def __getitem__(self, idx):
        return self.get_lab_from_path(self.paths[idx])

    def __len__(self):
        return len(self.paths)


class PickleColorizationDataset(ColorizationDataset):
    def __getitem__(self, idx):
        try:
            return torch.load(self.paths[idx])
        except Exception as e:
            print(f"Error loading file {self.paths[idx]}: {str(e)}")
            raise

def make_datasets(path, config, limit=None):
    img_paths = glob.glob(path + "/*")
    if limit:
        img_paths = random.sample(img_paths, limit)
    n_imgs = len(img_paths)
    train_split = img_paths[:int(n_imgs * .9)]
    val_split = img_paths[int(n_imgs * .9):]

    train_dataset = ColorizationDataset(
        train_split, split="train", config=config)
    val_dataset = ColorizationDataset(val_split, split="val", config=config)
    print(f"Train size: {len(train_split)}")
    print(f"Val size: {len(val_split)}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    return train_dataset, val_dataset


def make_dataloaders(path, config, num_workers=2, shuffle=True, limit=None):
    train_dataset, val_dataset = make_datasets(path, config, limit=limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(train_dataset,
                          batch_size=config["batch_size"],
                          num_workers=num_workers,
                          pin_memory=True if device == 'cuda' else config.get("pin_memory", True),
                          persistent_workers=True,
                          shuffle=shuffle)
    val_dl = DataLoader(val_dataset,
                        batch_size=config["batch_size"],
                        num_workers=num_workers,
                        pin_memory=True if device == 'cuda' else config.get("pin_memory", True),
                        persistent_workers=True,
                        shuffle=shuffle)
    return train_dl, val_dl


def make_custom_dataloaders(L_arrays, ab_arrays, config, train_split=0.9, num_workers=2, batch_size=None):
    """
    Create dataloaders for custom L and ab arrays
    
    Args:
        L_arrays: List of numpy arrays of shape (224, 224)
        ab_arrays: List of numpy arrays of shape (224, 224, 2)
        config: Configuration dictionary
        train_split: Fraction of data to use for training
        num_workers: Number of workers for dataloader
        batch_size: Batch size (uses config if None)
        device: 'cpu' or 'cuda' for GPU support
    """
    # Split data into train and validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = len(L_arrays)
    indices = list(range(n_samples))
    random.shuffle(indices)
    split_idx = int(n_samples * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create datasets
    train_dataset = CustomColorizationDataset(
        [L_arrays[i] for i in train_indices],
        [ab_arrays[i] for i in train_indices],
        split='train',
        config=config,
    )
    
    val_dataset = CustomColorizationDataset(
        [L_arrays[i] for i in val_indices],
        [ab_arrays[i] for i in val_indices],
        split='val',
        config=config,
    )
    
    # Create dataloaders
    batch_size = batch_size or config.get("batch_size", 36)
    
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else config.get("pin_memory", True),
        persistent_workers=True,
        shuffle=True
    )
    
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else config.get("pin_memory", True),
        persistent_workers=True,
        shuffle=False
    )
    
    print(f"Train size: {len(train_indices)}")
    print(f"Val size: {len(val_indices)}")
    print(f"Using device: {device}")
    
    return train_dl, val_dl


if __name__ == "__main__":
    enc_config, unet_config, colordiff_config = load_default_configs()
    train_dl, val_dl = make_dataloaders("./fairface",
                                        colordiff_config,
                                        num_workers=4)
    x = next(iter(train_dl))
    y = next(iter(val_dl))
    print(f"y.shape = {y.shape}")
    print(f"x.shape = {x.shape}")
