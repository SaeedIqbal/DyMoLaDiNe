import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import random
import string
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalMedicalSample:
    """Container for a single multi-modal medical sample."""
    def __init__(self, data: Dict[str, Any], label: int):
        self.data = data
        self.label = label

class GenericMultiModalDataset(Dataset):
    """
    A generic dataset class for multi-modal medical data.
    This base class handles common loading logic.
    Specific datasets should inherit and implement _load_data_list.
    """
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, 
                 modality_types: List[str] = ['image'], is_train: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.modality_types = modality_types
        self.is_train = is_train
        
        logger.info(f"Loading data list from {data_dir}...")
        self.data_list = self._load_data_list()
        logger.info(f"Loaded {len(self.data_list)} samples.")

    def _load_data_list(self) -> List[Dict[str, str]]:
        """
        Load list of data samples. This is a placeholder implementation.
        Should return a list of dictionaries with paths/identifiers for each modality.
        Override this method in specific dataset classes.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            if not os.path.exists(img_path):
                logger.warning(f"Image file not found: {img_path}. Using blank image.")
                return torch.zeros(3, 224, 224)
                
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                # Default preprocessing
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image = preprocess(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224)

    def _load_text(self, text_path: str) -> torch.Tensor:
        """Load and preprocess text data."""
        try:
            if not os.path.exists(text_path):
                logger.warning(f"Text file not found: {text_path}. Using dummy text.")
                text = " ".join(random.choices(string.ascii_letters, k=100))
            else:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Simple tokenization - in practice, use a proper tokenizer like BERT's
            tokens = text.split()[:512]  # Limit tokens
            if not tokens:
                tokens = ["<PAD>"]
            
            # Create a simple vocabulary (in practice, use a pre-trained tokenizer)
            vocab = defaultdict(lambda: 0)  # 0 for unknown/pad
            # Add common tokens for demo
            vocab["<PAD>"] = 0
            vocab["<UNK>"] = 1
            
            # Assign indices to unique tokens (simplified)
            unique_tokens = list(set(tokens))
            for i, token in enumerate(unique_tokens):
                vocab[token] = i + 2  # Start from 2 to avoid PAD/UNK
            
            indices = [vocab.get(token, 1) for token in tokens]  # 1 for UNK
            
            # Pad or truncate to fixed length
            max_len = 512
            if len(indices) < max_len:
                indices.extend([0] * (max_len - len(indices)))
            else:
                indices = indices[:max_len]
                
            return torch.tensor(indices, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading text {text_path}: {e}")
            return torch.zeros(512, dtype=torch.long)

    def _load_tabular(self, tabular_path: str) -> torch.Tensor:
        """Load and preprocess tabular data."""
        try:
            if not os.path.exists(tabular_path):
                logger.warning(f"Tabular file not found: {tabular_path}. Using dummy data.")
                tabular_data = np.random.rand(10).astype(np.float32)  # 10 features
            else:
                # Try to load as CSV
                df = pd.read_csv(tabular_path)
                # Assume first row is the data for this sample
                if len(df) > 0:
                    tabular_data = df.iloc[0].values.astype(np.float32)
                else:
                    tabular_data = np.random.rand(10).astype(np.float32)
            
            # Ensure fixed size
            target_size = 50  # Example size
            if len(tabular_data) < target_size:
                # Pad with zeros
                tabular_data = np.pad(tabular_data, (0, target_size - len(tabular_data)), mode='constant')
            elif len(tabular_data) > target_size:
                # Truncate
                tabular_data = tabular_data[:target_size]
                
            return torch.tensor(tabular_data)
        except Exception as e:
            logger.error(f"Error loading tabular data {tabular_path}: {e}")
            return torch.randn(50)  # Return random tensor of expected size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_item = self.data_list[idx]
        sample = {}

        # Load modalities based on what's specified
        if 'image' in self.modality_types and 'image_path' in data_item:
            sample['image'] = self._load_image(data_item['image_path'])

        if 'text' in self.modality_types and 'text_path' in data_item:
            sample['text'] = self._load_text(data_item['text_path'])

        if 'tabular' in self.modality_types and 'tabular_path' in data_item:
            sample['tabular'] = self._load_tabular(data_item['tabular_path'])

        # Load label - this is highly dataset specific
        # For now, we'll assume it's in the data_item or generate a dummy one
        if 'label' in data_item:
            sample['label'] = torch.tensor(data_item['label'], dtype=torch.long)
        else:
            # Dummy binary label
            sample['label'] = torch.tensor(random.randint(0, 1), dtype=torch.long)

        return sample


# Specific dataset classes inheriting from the base class
class MedMDAndRadMDDataset(GenericMultiModalDataset):
    """Dataset class for MedMD&RadMD."""
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image', 'text'], is_train=is_train)

    def _load_data_list(self) -> List[Dict[str, str]]:
        """Load MedMD&RadMD specific data list."""
        data_list = []
        # Assuming a structure like:
        # /data_dir/images/.../*.jpg
        # /data_dir/reports/.../*.txt
        image_dir = os.path.join(self.data_dir, 'images')
        text_dir = os.path.join(self.data_dir, 'reports')
        
        if not os.path.exists(image_dir):
            logger.warning(f"Image directory not found: {image_dir}")
            return []
            
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    data_id = os.path.splitext(file)[0]
                    img_path = os.path.join(root, file)
                    
                    # Find corresponding text report
                    # This is a simplification - real implementation would need more robust matching
                    relative_path = os.path.relpath(root, image_dir)
                    text_subdir = os.path.join(text_dir, relative_path)
                    text_path = os.path.join(text_subdir, data_id + '.txt')
                    
                    # Only add if image exists (text is optional but good to check)
                    if os.path.exists(img_path):
                        item = {'id': data_id, 'image_path': img_path}
                        if os.path.exists(text_path):
                            item['text_path'] = text_path
                        # For this dataset, we might also have a label file or infer from path
                        # This is a placeholder - real implementation needs proper label handling
                        item['label'] = 0  # Placeholder
                        data_list.append(item)
                        
        return data_list[:1000]  # Limit for demo


class MultiCaReDataset(GenericMultiModalDataset):
    """Dataset class for MultiCaRe."""
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image', 'text', 'tabular'], is_train=is_train)

    def _load_data_list(self) -> List[Dict[str, str]]:
        """Load MultiCaRe specific data list."""
        data_list = []
        # Assuming a structure where each case is in its own directory
        # /data_dir/case_001/image.jpg, report.txt, metadata.csv, label.json
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return []
            
        for case_dir in os.listdir(self.data_dir):
            case_path = os.path.join(self.data_dir, case_dir)
            if os.path.isdir(case_path):
                item = {'id': case_dir}
                
                # Check for image
                img_path = os.path.join(case_path, 'image.jpg')  # Or other extensions
                if os.path.exists(img_path):
                    item['image_path'] = img_path
                else:
                    # Try other common extensions
                    for ext in ['.png', '.jpeg', '.tiff']:
                        img_path_test = os.path.join(case_path, 'image' + ext)
                        if os.path.exists(img_path_test):
                            item['image_path'] = img_path_test
                            break
                
                # Check for text report
                text_path = os.path.join(case_path, 'report.txt')
                if os.path.exists(text_path):
                    item['text_path'] = text_path
                    
                # Check for tabular metadata
                tabular_path = os.path.join(case_path, 'metadata.csv')
                if os.path.exists(tabular_path):
                    item['tabular_path'] = tabular_path
                    
                # Check for label
                label_path = os.path.join(case_path, 'label.json')
                if os.path.exists(label_path):
                    try:
                        with open(label_path, 'r') as f:
                            label_data = json.load(f)
                            item['label'] = label_data.get('label', 0)
                    except:
                        item['label'] = 0
                else:
                    item['label'] = 0  # Placeholder
                    
                # Only add if we have at least an image
                if 'image_path' in item:
                    data_list.append(item)
                    
        return data_list[:1000]  # Limit for demo


# Placeholder classes for other datasets with image-only modalities
class PadChestDataset(GenericMultiModalDataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image'], is_train=is_train)
        
    def _load_data_list(self) -> List[Dict[str, str]]:
        data_list = []
        image_dir = self.data_dir
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(root, file)
                    data_id = os.path.splitext(file)[0]
                    # Placeholder label - in real implementation, get from file name or metadata
                    data_list.append({
                        'id': data_id,
                        'image_path': img_path,
                        'label': 0  # Placeholder
                    })
        return data_list[:500]  # Limit for demo


class TCIAREMINDDataset(GenericMultiModalDataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image'], is_train=is_train)
        
    def _load_data_list(self) -> List[Dict[str, str]]:
        data_list = []
        image_dir = self.data_dir
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(root, file)
                    data_id = os.path.splitext(file)[0]
                    data_list.append({
                        'id': data_id,
                        'image_path': img_path,
                        'label': 0  # Placeholder
                    })
        return data_list[:500]  # Limit for demo


class BRaTSDataset(GenericMultiModalDataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image'], is_train=is_train)
        
    def _load_data_list(self) -> List[Dict[str, str]]:
        data_list = []
        image_dir = self.data_dir
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(root, file)
                    data_id = os.path.splitext(file)[0]
                    data_list.append({
                        'id': data_id,
                        'image_path': img_path,
                        'label': 0  # Placeholder
                    })
        return data_list[:500]  # Limit for demo


class Camelyon16Dataset(GenericMultiModalDataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image'], is_train=is_train)
        
    def _load_data_list(self) -> List[Dict[str, str]]:
        data_list = []
        image_dir = self.data_dir
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(root, file)
                    data_id = os.path.splitext(file)[0]
                    data_list.append({
                        'id': data_id,
                        'image_path': img_path,
                        'label': 0  # Placeholder
                    })
        return data_list[:500]  # Limit for demo


class PANDADataset(GenericMultiModalDataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, is_train: bool = True):
        super().__init__(data_dir, transform, modality_types=['image'], is_train=is_train)
        
    def _load_data_list(self) -> List[Dict[str, str]]:
        data_list = []
        image_dir = self.data_dir
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(root, file)
                    data_id = os.path.splitext(file)[0]
                    data_list.append({
                        'id': data_id,
                        'image_path': img_path,
                        'label': 0  # Placeholder
                    })
        return data_list[:500]  # Limit for demo


# Data Perturbation Functions (as described in the paper)
def apply_gaussian_noise(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian noise to an image."""
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0, 1)

def apply_resolution_degradation(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Apply resolution degradation by downsampling and upsampling."""
    if factor <= 1:
        return image
    
    C, H, W = image.shape
    new_H, new_W = max(1, int(H / factor)), max(1, int(W / factor))
    
    # Downsample
    downsampled = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    # Upsample back
    upsampled = torch.nn.functional.interpolate(
        downsampled.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    return upsampled

def apply_contrast_adjustment(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image contrast."""
    # Ensure image is in [0,1] range
    image = torch.clamp(image, 0, 1)
    mean = torch.mean(image, dim=(1, 2), keepdim=True)
    adjusted = (image - mean) * factor + mean
    return torch.clamp(adjusted, 0, 1)

def apply_tabular_missingness(tabular_data: torch.Tensor, p_miss: float) -> torch.Tensor:
    """Simulate missingness in tabular data."""
    mask = torch.rand(tabular_data.shape) > p_miss
    corrupted_data = tabular_data * mask
    return corrupted_data

def apply_tabular_noise(tabular_data: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to tabular data."""
    noise = torch.randn_like(tabular_data) * sigma
    return tabular_data + noise

def apply_text_masking(text_tensor: torch.Tensor, mask_rate: float = 0.15) -> torch.Tensor:
    """Apply random word masking to text."""
    mask_token_id = 0  # Assuming 0 is the mask/pad token
    mask = torch.rand(text_tensor.shape) < mask_rate
    masked_text = text_tensor * (~mask) + mask_token_id * mask
    return masked_text.long()


class PerturbedMultiModalDataset(Dataset):
    """
    Wrapper class to apply dynamic perturbations to a base dataset.
    """
    def __init__(self, base_dataset: Dataset, perturbation_config: Optional[Dict[str, float]] = None):
        self.base_dataset = base_dataset
        self.perturbation_config = perturbation_config or {
            'sigma_img_max': 1.0,
            'downsample_factor_max': 8.0,
            'contrast_min': 0.5,
            'contrast_max': 1.5,
            'p_miss_max': 0.5,
            'sigma_tab_max': 0.1,
            'p_mask_max': 0.3
        }

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        perturbed_sample = {}
        
        # Sample perturbation severities dynamically for this instance
        sigma_img = np.random.uniform(0, self.perturbation_config.get('sigma_img_max', 1.0))
        w = np.random.uniform(1, self.perturbation_config.get('downsample_factor_max', 8.0))
        r = np.random.uniform(
            self.perturbation_config.get('contrast_min', 0.5), 
            self.perturbation_config.get('contrast_max', 1.5)
        )
        p_miss = np.random.uniform(0, self.perturbation_config.get('p_miss_max', 0.5))
        sigma_tab = np.random.uniform(0, self.perturbation_config.get('sigma_tab_max', 0.1))
        p_mask = np.random.uniform(0, self.perturbation_config.get('p_mask_max', 0.3))

        # Apply perturbations to each modality if present
        if 'image' in sample:
            img = sample['image']
            # Apply image perturbations
            img = apply_gaussian_noise(img, sigma_img)
            img = apply_resolution_degradation(img, w)
            img = apply_contrast_adjustment(img, r)
            perturbed_sample['image'] = img
            # Store perturbation levels for potential use in attribution
            perturbed_sample['image_perturbations'] = {
                'sigma': sigma_img, 
                'downsample_factor': w, 
                'contrast_factor': r
            }

        if 'text' in sample:
            text = sample['text']
            # Apply text perturbations
            text = apply_text_masking(text, p_mask)
            perturbed_sample['text'] = text
            perturbed_sample['text_perturbations'] = {'mask_rate': p_mask}

        if 'tabular' in sample:
            tab = sample['tabular']
            # Apply tabular perturbations
            tab = apply_tabular_missingness(tab, p_miss)
            tab = apply_tabular_noise(tab, sigma_tab)
            perturbed_sample['tabular'] = tab
            perturbed_sample['tabular_perturbations'] = {
                'missing_rate': p_miss, 
                'noise_sigma': sigma_tab
            }

        perturbed_sample['label'] = sample['label']
        return perturbed_sample


def load_and_preprocess_datasets(base_path: str = '/home/phd/dataset') -> Tuple[Dict, Dict]:
    """
    Loads and preprocesses all specified datasets.
    
    Args:
        base_path: Base path where datasets are stored
        
    Returns:
        Tuple of (datasets_dict, dataloaders_dict)
    """
    datasets = {}
    dataloaders = {}

    # Define dataset configurations
    dataset_configs = {
        'MedMD&RadMD': {
            'class': MedMDAndRadMDDataset,
            'path': os.path.join(base_path, 'MedMD_RadMD'),
            'modalities': ['image', 'text']
        },
        'MultiCaRe': {
            'class': MultiCaReDataset,
            'path': os.path.join(base_path, 'MultiCaRe'),
            'modalities': ['image', 'text', 'tabular']
        },
        'PadChest': {
            'class': PadChestDataset,
            'path': os.path.join(base_path, 'PadChest'),
            'modalities': ['image']
        },
        'TCIA_RE-MIND': {
            'class': TCIAREMINDDataset,
            'path': os.path.join(base_path, 'TCIA_RE-MIND'),
            'modalities': ['image']
        },
        'BRaTS': {
            'class': BRaTSDataset,
            'path': os.path.join(base_path, 'BRaTS'),
            'modalities': ['image']
        },
        'Camelyon16': {
            'class': Camelyon16Dataset,
            'path': os.path.join(base_path, 'Camelyon16'),
            'modalities': ['image']
        },
        'PANDA': {
            'class': PANDADataset,
            'path': os.path.join(base_path, 'PANDA'),
            'modalities': ['image']
        }
    }

    # Common image transform
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Perturbation configuration
    perturbation_config = {
        'sigma_img_max': 1.0,
        'downsample_factor_max': 8.0,
        'contrast_min': 0.5,
        'contrast_max': 1.5,
        'p_miss_max': 0.5,
        'sigma_tab_max': 0.1,
        'p_mask_max': 0.3
    }

    for name, config in dataset_configs.items():
        data_path = config['path']
        if not os.path.exists(data_path):
            logger.warning(f"Path {data_path} for dataset {name} does not exist. Skipping.")
            continue

        try:
            logger.info(f"Loading {name}...")
            # Load clean dataset
            clean_dataset = config['class'](
                data_path, 
                transform=common_transform, 
                is_train=True
            )
            
            # Check if dataset loaded successfully
            if len(clean_dataset) == 0:
                logger.warning(f"No samples found in {name}. Skipping.")
                continue
            
            # Split into train/val/test (simplified split)
            total_size = len(clean_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            if test_size <= 0:
                logger.warning(f"Not enough samples in {name} for splitting. Skipping.")
                continue
                
            train_dataset, temp_dataset = random_split(clean_dataset, [train_size, val_size + test_size])
            val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

            # Create perturbed test set
            perturbed_test_dataset = PerturbedMultiModalDataset(test_dataset, perturbation_config)

            datasets[name] = {
                'train': train_dataset,
                'val': val_dataset,
                'test_clean': test_dataset,
                'test_perturbed': perturbed_test_dataset
            }

            # Create DataLoaders
            batch_size = 32  # Standard batch size for medical imaging
            num_workers = 4   # Adjust based on your system
            
            dataloaders[name] = {
                'train': DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    pin_memory=True
                ),
                'val': DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=num_workers,
                    pin_memory=True
                ),
                'test_clean': DataLoader(
                    test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=num_workers,
                    pin_memory=True
                ),
                'test_perturbed': DataLoader(
                    perturbed_test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=num_workers,
                    pin_memory=True
                )
            }
            logger.info(f"  Loaded {name} with {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
            continue

    return datasets, dataloaders


# Example usage
if __name__ == "__main__":
    # This would be called from your main training script
    try:
        datasets, dataloaders = load_and_preprocess_datasets('/home/phd/dataset')
        logger.info(f"\nAvailable datasets: {list(datasets.keys())}")
        
        # Print some info about loaded datasets
        for name, dls in dataloaders.items():
            logger.info(f"\n{name} Dataloaders:")
            for split_name, dl in dls.items():
                try:
                    logger.info(f"  {split_name}: {len(dl)} batches")
                    # Show shape of first batch as a check
                    first_batch = next(iter(dl))
                    logger.info(f"    Sample batch keys: {list(first_batch.keys())}")
                    for k, v in first_batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"      {k} shape: {v.shape}")
                    break  # Just check the first split
                except Exception as e:
                    logger.error(f"    Error inspecting {split_name} dataloader: {e}")
                    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")