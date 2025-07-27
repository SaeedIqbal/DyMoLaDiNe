import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
import os
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import your custom modules ---
# These need to be implemented and available in your project structure.
# Replace 'your_project' with your actual package name if applicable.
try:
    # from mm_vit import MultiModalVisionTransformer
    # from dynamic_weighting import DynamicModalityWeightingMechanism
    # from diffusion_ensemble import RobustMultiModalDiffusionEnsemble
    # from uncertainty_quantification import ModalityAttributedUncertaintyQuantifier
    # from data.perturbation_protocols import apply_perturbation # Hypothetical module for perturbs
    print("Warning: Actual model imports are commented out. Using mock classes.")
    raise ImportError # Trigger mock imports for this example
except ImportError:
    # --- Mock Classes for Demonstration ---
    # In a real scenario, replace these with imports from your actual component files.
    from types import SimpleNamespace
    import torch.nn as nn
    class MultiModalVisionTransformer(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): return torch.randn(x[list(x.keys())[0]].size(0), 512), [torch.randn(x[list(x.keys())[0]].size(0), 512) for _ in range(3)]
    class DynamicModalityWeightingMechanism(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args, **kwargs): return torch.rand(kwargs.get('z_common', torch.randn(1, 512)).size(0), 3), torch.softmax(torch.randn(kwargs.get('z_common', torch.randn(1, 512)).size(0), 5), dim=-1), {}
    class RobustMultiModalDiffusionEnsemble(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def sample(self, *args, **kwargs): return F.softmax(torch.randn(args[0][0].size(0), 2), dim=-1), None
    class ModalityAttributedUncertaintyQuantifier:
        def __init__(self, *args, **kwargs): pass
        def quantify(self, *args, **kwargs):
            B = kwargs['original_input'][list(kwargs['original_input'].keys())[0]].size(0)
            return {
                'predictive_mean': torch.rand(B, 2),
                'marginal_vars': torch.rand(B, 2),
                'cpiw_like': torch.rand(B, 2),
                'modality_uncertainties': {'image': torch.rand(B), 'text': torch.rand(B)},
                'most_uncertain_modalities': [list(kwargs['original_input'].keys())[i%len(kwargs['original_input'])] for i in range(B)],
                'max_attributed_uncertainties': torch.rand(B),
                'ablated_predictive_means': {'image': torch.rand(B, 2), 'text': torch.rand(B, 2)}
            }
    def apply_perturbation(data, perturb_type, severity): # Mock perturbation
        # In reality, this would modify the data tensors
        perturbed_data = {k: v.clone() for k, v in data.items()}
        # Add testing perturbation info
        perturbed_data['perturbation_info'] = f"{perturb_type}_level_{severity}"
        return perturbed_data


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device (CUDA, MPS, CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path: str, device: torch.device) -> nn.ModuleDict:
    """Load the trained DyMoLaDiNE model."""
    logger.info(f"Loading model from {model_path}")
    # In a real scenario, you would reconstruct the model architecture
    # and load the state dict.
    # Example (conceptual):
    # model = nn.ModuleDict({
    #     'mm_vit': MultiModalVisionTransformer(...),
    #     'dmw': DynamicModalityWeightingMechanism(...),
    #     'diffusion_ensemble': RobustMultiModalDiffusionEnsemble(...),
    #     # 'uncertainty_quantifier' is usually not a nn.Module, created separately
    # })
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to(device)
    # model.eval()
    # return model
    
    # For demo with mocks, return a testing ModuleDict
    model = nn.ModuleDict({
        'mm_vit': MultiModalVisionTransformer(),
        'dmw': DynamicModalityWeightingMechanism(),
        'diffusion_ensemble': RobustMultiModalDiffusionEnsemble()
    })
    model.to(device)
    model.eval()
    logger.info("Model loaded (mocked) and set to eval mode.")
    return model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DyMoLaDiNE Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the test dataset directory')
    parser.add_argument('--config', type=str, required=True, help='Path to evaluation configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save results')
    parser.add_argument('--perturbation_types', nargs='+', default=['clean'], help='Types of perturbations to evaluate on')
    return parser.parse_args()

def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# --- Evaluation Metrics ---

def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    if labels.dim() > 1 and labels.size(1) > 1: # One-hot
        labels = torch.argmax(labels, dim=1)
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

def calculate_ece(predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    Adapted from Guo et al. (2017).
    """
    if labels.dim() > 1 and labels.size(1) > 1: # One-hot
        labels = torch.argmax(labels, dim=1)
        
    confidences, predictions = torch.max(predictions, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = labels.size(0)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate bin location
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def calculate_cpiw(cpiw_like_per_class: torch.Tensor) -> float:
    """Calculate average CPIW-like metric across classes."""
    # cpiw_like_per_class is (B, C)
    return torch.mean(cpiw_like_per_class).item()

def calculate_cnpv(marginal_vars_per_class: torch.Tensor) -> float:
    """Calculate average CNPV-like metric across classes."""
    # marginal_vars_per_class is (B, C), analogous to CNPV calculation
    return torch.mean(marginal_vars_per_class).item()

def calculate_attribution_fidelity(
    predicted_modalities: List[str], 
    ground_truth_modalities: List[str]
) -> float:
    """Calculate Attribution Fidelity (AF)."""
    if len(predicted_modalities) != len(ground_truth_modalities):
        raise ValueError("Length of predicted and ground truth modalities must match.")
    if not predicted_modalities:
        return 0.0
    correct = sum(1 for p, gt in zip(predicted_modalities, ground_truth_modalities) if p == gt)
    return correct / len(predicted_modalities)

# --- Perturbation and Evaluation Logic ---

def evaluate_model_on_loader(
    model: nn.ModuleDict,
    dataloader: DataLoader,
    device: torch.device,
    perturbation_type: str = 'clean',
    perturbation_severity: float = 1.0,
    uncertainty_quantifier: Optional[ModalityAttributedUncertaintyQuantifier] = None
) -> Dict[str, Any]:
    """
    Evaluate the model on a given DataLoader, potentially with perturbations.
    Returns a dictionary of metrics.
    """
    logger.info(f"Starting evaluation on '{perturbation_type}' (severity: {perturbation_severity})...")
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_cpiw = []
    all_cnpv = []
    all_predicted_modalities = []
    all_ground_truth_modalities = [] # This would need to be determined per sample

    # Define a simple sample generator function for the uncertainty quantifier
    # This bridges the gap between the model components and the quantifier.
    def sample_generator(input_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, None]:
        """Generates final predictions and raw samples (mocked as None here)."""
        with torch.no_grad():
            # --- Forward Pass through DyMoLaDiNE Components ---
            # 1. MM-ViT
            _, invariant_features_list = model['mm_vit'](input_dict)
            z_common_from_mm_vit = invariant_features_list[-1] if invariant_features_list else torch.randn(input_dict[list(input_dict.keys())[0]].size(0), 512, device=device)
            z_common_list = [z_common_from_mm_vit] * 5 # Mock K=5

            # 2. Dynamic Modality Weighting
            s_m_prime, pi_k_m_omega, _ = model['dmw'](
                modality_features_dict=input_dict,
                z_common=z_common_from_mm_vit,
                original_input=input_dict
            )
            
            # 3. Diffusion Ensemble Sampling
            final_preds, raw_samples = model['diffusion_ensemble'].sample(
                z_common_list=z_common_list,
                reliability_scores=s_m_prime,
                raw_modalities=input_dict,
                dynamic_weights=pi_k_m_omega,
                num_samples_per_member=20 # M from config
            )
            # raw_samples are None in mock, but could be used by uncertainty quantifier
            return final_preds, raw_samples

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            try:
                # --- 1. Apply Perturbation ---
                if perturbation_type != 'clean':
                    # In a real scenario, you'd have a function like:
                    # data = apply_perturbation(data, perturbation_type, perturbation_severity)
                    # For demo, we just add a key
                    data = apply_perturbation(data, perturbation_type, perturbation_severity)
                    logger.debug(f"Applied {perturbation_type} perturbation to batch {batch_idx}")

                # Move data to device
                inputs = {k: v.to(device) for k, v in data.items() if k != 'label' and not k.startswith('perturbation')}
                labels = data['label'].to(device)
                
                # --- 2. Get Predictions ---
                if uncertainty_quantifier is not None:
                    # --- Use Uncertainty Quantifier for Prediction and Metrics ---
                    # The quantifier handles the sampling internally via sample_generator
                    uq_results = uncertainty_quantifier.quantify(
                        sample_generator=sample_generator,
                        original_input=inputs,
                    )
                    predictions = uq_results['predictive_mean'] # (B, C)
                    # Collect metrics from UQ results
                    if uq_results['cpiw_like'] is not None:
                        all_cpiw.extend(uq_results['cpiw_like'].cpu().numpy())
                    if uq_results['marginal_vars'] is not None: # Use as CNPV proxy
                         all_cnpv.extend(uq_results['marginal_vars'].cpu().numpy())
                    all_predicted_modalities.extend(uq_results['most_uncertain_modalities'])
                    # Determine ground truth modality for attribution fidelity
                    # This is highly dataset/perturbation specific.
                    # Example logic (simplified):
                    # Assume perturbation_info tells us which modality was most affected.
                    # In a real implementation, you'd have a more robust mapping.
                    gt_modality = "unknown" # Placeholder
                    if 'perturbation_info' in data:
                        info = data['perturbation_info']
                        if 'image' in info.lower():
                            gt_modality = 'image'
                        elif 'text' in info.lower():
                            gt_modality = 'text'
                        # Add more logic for other modalities/perturbations
                    all_ground_truth_modalities.extend([gt_modality] * predictions.shape[0])

                else:
                    # --- Direct Prediction from Model (without detailed UQ) ---
                    predictions, _ = sample_generator(inputs) # Use the internal generator

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue # Skip problematic batch

    if not all_predictions:
        logger.warning("No predictions were made. Returning zero metrics.")
        return {
            'accuracy': 0.0, 'ece': 1.0, 'cpiw': float('inf'), 'cnpv': float('inf'), 'attribution_fidelity': 0.0
        }

    # --- 3. Aggregate Predictions and Labels ---
    all_predictions_tensor = torch.cat(all_predictions, dim=0) # (Total_B, C)
    all_labels_tensor = torch.cat(all_labels, dim=0)          # (Total_B, ...) 

    # --- 4. Calculate Metrics ---
    accuracy = calculate_accuracy(all_predictions_tensor, all_labels_tensor)
    ece = calculate_ece(all_predictions_tensor, all_labels_tensor)
    
    avg_cpiw = np.mean(all_cpiw) if all_cpiw else float('inf')
    avg_cnpv = np.mean(all_cnpv) if all_cnpv else float('inf')
    
    attribution_fidelity = calculate_attribution_fidelity(all_predicted_modalities, all_ground_truth_modalities)

    logger.info(f"Evaluation on '{perturbation_type}' completed.")
    return {
        'accuracy': accuracy,
        'ece': ece,
        'cpiw': avg_cpiw,
        'cnpv': avg_cnpv,
        'attribution_fidelity': attribution_fidelity
    }

def run_evaluation(args):
    """Main evaluation function."""
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load configuration
    config = load_eval_config(args.config)
    logger.info(f"Loaded evaluation configuration from {args.config}")

    # Load model
    model = load_model(args.model_path, device)

    # Initialize Uncertainty Quantifier (if required for detailed metrics)
    # The quantifier needs to know the modality names from the data
    modality_names = config.get('modality_names', ['image', 'text']) # Default example
    uncertainty_quantifier = ModalityAttributedUncertaintyQuantifier(
        num_classes=config.get('num_classes', 2),
        modality_names=modality_names,
        sharpness_param=config.get('sharpness_param', 0.1737)
    ) if config.get('use_uncertainty_quantifier', True) else None

    # --- Setup Data Loading ---
    # This part is highly dependent on your data structure and preprocessing pipeline.
    # You would typically load a test DataLoader here.
    # For demonstration, we'll create a testing one or assume it's passed correctly.
    # A real implementation would involve:
    # from data.load_datasets import load_and_preprocess_datasets # Hypothetical
    # datasets, dataloaders = load_and_preprocess_datasets(args.data_dir)
    # test_loader_clean = dataloaders['your_dataset']['test_clean']
    # For demo, let's assume a function or create a testing loader.
    # Let's assume a function `get_test_dataloader` exists.
    def get_test_dataloader(data_dir, perturbation_type='clean', severity=1.0):
        # This is a placeholder. Replace with your actual data loading logic.
        # It should return a DataLoader for the specified perturbation.
        # For demo, we return a testing loader.
        from torch.utils.data import TensorDataset, DataLoader
        import random
        # testing data: 10 samples, 2 modalities (image 3x32x32, text 100), 2 classes
        testing_data = []
        for _ in range(10):
            sample = {
                'image': torch.randn(3, 32, 32),
                'text': torch.randn(100),
                'label': torch.tensor(random.randint(0, 1))
            }
            if perturbation_type != 'clean':
                sample['perturbation_info'] = f"{perturbation_type}_level_{severity}"
            testing_data.append(sample)
        
        def collate_fn(batch):
            collated = {}
            for key in batch[0].keys():
                collated[key] = torch.stack([sample[key] for sample in batch])
            return collated
            
        return DataLoader(testing_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # --- Run Evaluations ---
    results = {}
    perturbation_types = args.perturbation_types
    perturbation_severities = config.get('perturbation_severities', [1.0]) # e.g., [0.5, 1.0]

    for p_type in perturbation_types:
        results[p_type] = {}
        severities_to_test = [1.0] if p_type == 'clean' else perturbation_severities
        for severity in severities_to_test:
            severity_key = f"severity_{severity}"
            logger.info(f"Evaluating model on {p_type} with severity {severity}...")
            
            # Get the appropriate DataLoader for this perturbation
            # test_loader = get_test_dataloader(args.data_dir, p_type, severity)
            test_loader = get_test_dataloader(args.data_dir, p_type, severity) # Use testing
            
            # Perform evaluation
            metrics = evaluate_model_on_loader(
                model, test_loader, device, p_type, severity, uncertainty_quantifier
            )
            results[p_type][severity_key] = metrics
            logger.info(f"Metrics for {p_type} ({severity_key}): {metrics}")

    # --- Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {results_file}")

    # --- (Optional) Print Summary ---
    print("\n--- Evaluation Summary ---")
    for p_type, severities in results.items():
        print(f"\n Perturbation: {p_type}")
        for severity_key, metrics in severities.items():
            print(f"  {severity_key}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
    print("--------------------------\n")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
    logger.info("Evaluation script finished.")
