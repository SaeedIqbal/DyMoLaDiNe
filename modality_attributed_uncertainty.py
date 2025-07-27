# modality_attributed_uncertainty.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityAttributedUncertaintyQuantifier:
    """
    Framework for Modality-Attributed Uncertainty Quantification.
    
    Generates samples, computes total uncertainty, and decomposes it
    by modality influence using ablation sampling.
    """
    def __init__(
        self,
        num_classes: int, # C
        num_ensemble_members: int, # K
        modality_names: List[str], # e.g., ['image_features', 'text_features', 'tabular_features']
        baseline_generator: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None
    ):
        """
        Initializes the uncertainty quantifier.

        Args:
            num_classes (int): Number of classes (C).
            num_ensemble_members (int): Number of ensemble members (K).
            modality_names (List[str]): List of modality names.
            baseline_generator (Optional[Callable]): Function to generate baseline
                for a modality. Takes (modality_name, original_tensor) and returns baseline.
                If None, uses zero tensor.
        """
        self.num_classes = num_classes
        self.num_ensemble_members = num_ensemble_members
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        
        if baseline_generator is None:
            # Default: zero tensor baseline
            self.baseline_generator = lambda name, tensor: torch.zeros_like(tensor)
        else:
            self.baseline_generator = baseline_generator
            
        logger.info(f"Initialized ModalityAttributedUncertaintyQuantifier for {self.num_modalities} modalities: {modality_names}")

    def compute_predictive_statistics(
        self, 
        samples: torch.Tensor # Shape (B, K*M, C) or list of (B, M, C) tensors from ensemble
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the predictive mean, covariance, and marginal variances from samples.

        Args:
            samples (torch.Tensor): Monte Carlo samples of shape (B, N, C) where N = K*M,
                                    or a list of K tensors each (B, M, C) which will be concatenated.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Predictive mean y_bar: (B, C)
                - Predictive covariance Cov[Y|m']: (B, C, C)
                - Marginal predictive variances Var[Y_a|m']: (B, C)
        """
        if isinstance(samples, list):
            # Concatenate list of (B, M, C) tensors from K members into (B, K*M, C)
            samples = torch.cat(samples, dim=1) # (B, K*M, C)
            
        B, N, C = samples.shape # N is K*M
        
        if N < 2:
            raise ValueError("Need at least 2 samples to compute variance/covariance.")
            
        # 1. Compute predictive mean \bar{y} = (1/KM) * sum(y_{k,m})
        y_bar = torch.mean(samples, dim=1) # (B, C)
        
        # 2. Compute predictive covariance \hat{Cov}[Y|m'] = (1/(KM-1)) * sum((y_{k,m} - \bar{y})(y_{k,m} - \bar{y})^T)
        # Center the samples
        centered_samples = samples - y_bar.unsqueeze(1) # (B, N, C)
        # Compute outer products and sum
        # torch.bmm(centered_samples.transpose(1, 2), centered_samples) gives (B, C, C) sum of outer products
        # Divide by (N-1) for unbiased estimate
        cov_matrix = torch.bmm(centered_samples.transpose(1, 2), centered_samples) / (N - 1) # (B, C, C)
        
        # 3. Compute marginal variances \hat{Var}[Y_a|m'] = (1/(KM-1)) * sum((y_{k,m,a} - \bar{y}_a)^2)
        # This is the diagonal of the covariance matrix
        marginal_vars = torch.diagonal(cov_matrix, dim1=-2, dim2=-1) # (B, C)
        # Or compute directly:
        # marginal_vars = torch.var(samples, dim=1, unbiased=True) # (B, C) - This is simpler
        
        return y_bar, cov_matrix, marginal_vars

    def perform_ablation_sampling(
        self,
        original_input: Dict[str, torch.Tensor],
        ablated_modality_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        Creates an ablated input by replacing one modality with its baseline.

        Args:
            original_input (Dict[str, torch.Tensor]): The original multi-modal input.
            ablated_modality_name (str): Name of the modality to ablate.

        Returns:
            Dict[str, torch.Tensor]: The ablated input dictionary.
        """
        if ablated_modality_name not in original_input:
            raise ValueError(f"Modality {ablated_modality_name} not found in original input.")
            
        ablated_input = {}
        for mod_name, mod_tensor in original_input.items():
            if mod_name == ablated_modality_name:
                # Replace with baseline
                ablated_input[mod_name] = self.baseline_generator(mod_name, mod_tensor)
            else:
                # Keep original
                ablated_input[mod_name] = mod_tensor
        return ablated_input

    def compute_modality_attributed_uncertainty(
        self,
        predictive_mean: torch.Tensor, # \bar{y}, shape (B, C)
        ablated_predictive_means: Dict[str, torch.Tensor] # {\bar{y}^{(-i)}}, each shape (B, C)
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the attributed uncertainty U_{x^(i)} for each modality.

        U_{x^(i)} = ||\bar{y} - \bar{y}^{(-i)}||_2^2

        Args:
            predictive_mean (torch.Tensor): \bar{y}, shape (B, C).
            ablated_predictive_means (Dict[str, torch.Tensor]): Dict of \bar{y}^{(-i)}, shape (B, C) each.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping modality names to their attributed
                                     uncertainty U_{x^(i)} of shape (B,).
        """
        modality_uncertainties = {}
        for mod_name, ablated_mean in ablated_predictive_means.items():
            # Compute squared L2 norm of difference
            diff = predictive_mean - ablated_mean # (B, C)
            # Sum of squares across classes, for each item in batch
            uncertainty = torch.sum(diff**2, dim=1) # (B,)
            modality_uncertainties[mod_name] = uncertainty
            
        return modality_uncertainties

    def quantify(
        self,
        sample_generator: Callable[[Dict[str, torch.Tensor]], torch.Tensor], # Function to generate samples (B, N, C)
        original_input: Dict[str, torch.Tensor], # m'
        # The following are needed if CDMs require specific conditioning from DMW mechanism
        # They can be None if the sample_generator handles internal conditioning
        z_common_list: Optional[List[torch.Tensor]] = None, # List of z_common for each member
        reliability_scores: Optional[torch.Tensor] = None, # s(m')
        dynamic_weights: Optional[torch.Tensor] = None, # π_k(m')
        num_samples_per_member: int = 20 # M
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs the full uncertainty quantification pipeline.

        1. Generate samples from the full model.
        2. Compute predictive statistics (mean, cov, var).
        3. For each modality, perform ablation sampling and generate ablated samples.
        4. Compute ablated predictive means.
        5. Decompose uncertainty by computing U_{x^(i)}.

        Args:
            sample_generator (Callable): A function that takes an input dict and returns samples (B, N, C).
                                         This function should encapsulate the ensemble sampling logic,
                                         potentially using z_common_list, reliability_scores, dynamic_weights.
            original_input (Dict[str, torch.Tensor]): The original multi-modal input m'.
            z_common_list (Optional[List[torch.Tensor]]): List of z_common for each member.
            reliability_scores (Optional[torch.Tensor]): Scores s(m').
            dynamic_weights (Optional[torch.Tensor]): Weights π_k(m').
            num_samples_per_member (int, optional): Number of samples M per member. Defaults to 20.

        Returns:
            Tuple containing:
            - Predictive mean y_bar: (B, C)
            - Predictive covariance matrix: (B, C, C)
            - Marginal predictive variances: (B, C)
            - Dictionary of modality-attributed uncertainties U_{x^(i)}: {str: (B,)}
        """
        logger.info("Starting modality-attributed uncertainty quantification...")
        
        # 1. Generate samples from the full model p(Y|m', \Psi)
        # The sample_generator is responsible for using the ensemble, CDMs, DMW mechanism internally
        # to produce the final samples. It hides the complexity of K, M, z_common, etc.
        # Expected output shape: (B, K*M, C) or (B, N, C)
        full_samples = sample_generator(original_input) 
        logger.info(f"Generated full model samples with shape: {full_samples.shape}")

        # 2. Compute predictive statistics for the full model
        y_bar, cov_matrix, marginal_vars = self.compute_predictive_statistics(full_samples)
        logger.info("Computed predictive statistics for full model.")

        # 3. Perform ablation sampling and compute statistics for each ablated model
        ablated_predictive_means = {}
        for mod_name in self.modality_names:
            logger.info(f"Performing ablation for modality: {mod_name}")
            # a. Create ablated input m'^{(-i)}
            ablated_input = self.perform_ablation_sampling(original_input, mod_name)
            
            # b. Generate samples from the ablated model p(Y|m'^{(-i)}, \Psi)
            # Note: In a full implementation, the sample_generator might need to be told
            # which modality is ablated to adjust conditioning (e.g., s(m')).
            # For simplicity here, we assume the generator can handle the ablated input.
            # A more sophisticated approach would pass `ablated_modality_name` to the generator
            # or recompute s(m'^{(-i)}) internally.
            ablated_samples = sample_generator(ablated_input)
            logger.info(f"  Generated ablated samples with shape: {ablated_samples.shape}")
            
            # c. Compute predictive mean for the ablated model \bar{y}^{(-i)}
            y_bar_ablated, _, _ = self.compute_predictive_statistics(ablated_samples)
            ablated_predictive_means[mod_name] = y_bar_ablated
            logger.info(f"  Computed ablated predictive mean for {mod_name}.")

        # 4. Decompose uncertainty by computing U_{x^(i)}
        modality_uncertainties = self.compute_modality_attributed_uncertainty(
            predictive_mean=y_bar,
            ablated_predictive_means=ablated_predictive_means
        )
        logger.info("Computed modality-attributed uncertainties.")

        logger.info("Finished modality-attributed uncertainty quantification.")
        return y_bar, cov_matrix, marginal_vars, modality_uncertainties


# --- Example Usage ---
# This example assumes the existence of other components (MM-ViT, DMW, CDM Ensemble)
# and a way to integrate them into a `sample_generator` function.

if __name__ == "__main__":
    # Example configuration
    batch_size = 4
    num_classes = 2 # Binary classification (C)
    num_ensemble_members = 5 # K
    num_samples_per_member = 10 # M (smaller for demo)
    modality_names_example = ['image_features', 'text_features', 'tabular_features']
    
    # 1. Instantiate the quantifier
    uncertainty_quantifier = ModalityAttributedUncertaintyQuantifier(
        num_classes=num_classes,
        num_ensemble_members=num_ensemble_members,
        modality_names=modality_names_example,
        # Use default zero baseline
    )

    # 2. Define a mock sample generator function
    # This function would normally encapsulate the complex logic of the DyMoLaDiNe model:
    # MM-ViT -> features -> DMW (Rel_ψ, f_ω) -> CDM Ensemble sampling -> Final prediction samples
    # For this demo, it generates random samples.
    def mock_sample_generator(input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Mock function to simulate the sampling process.
        In a real scenario, this would use the trained DyMoLaDiNe model.
        """
        B = list(input_dict.values())[0].shape[0] # Get batch size from input
        N = num_ensemble_members * num_samples_per_member # Total samples
        C = num_classes
        
        # Generate random samples (this is just for demonstration)
        # In reality, these would come from the reverse diffusion process of the ensemble.
        samples = torch.randn(B, N, C)
        # Normalize to represent probabilities (simplex) if needed for certain metrics,
        # though for variance calculations, real-valued samples are fine.
        # samples = F.softmax(samples, dim=-1) 
        logger.info(f"  Mock sample generator produced samples of shape: ({B}, {N}, {C})")
        return samples

    # 3. Create dummy original input data
    dummy_original_input = {
        'image_features': torch.randn(batch_size, 768),
        'text_features': torch.randn(batch_size, 512),
        'tabular_features': torch.randn(batch_size, 256)
    }

    # 4. Run the uncertainty quantification
    try:
        y_bar, cov_matrix, marginal_vars, modality_uncertainties = uncertainty_quantifier.quantify(
            sample_generator=mock_sample_generator,
            original_input=dummy_original_input,
            # In a real scenario, you'd pass the actual z_common, s(m'), π_k(m')
            z_common_list=None, 
            reliability_scores=None,
            dynamic_weights=None,
            num_samples_per_member=num_samples_per_member
        )
        
        # 5. Inspect the results
        logger.info("--- Uncertainty Quantification Results ---")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Predictive mean (y_bar) shape: {y_bar.shape}") # (B, C)
        logger.info(f"Predictive covariance matrix shape: {cov_matrix.shape}") # (B, C, C)
        logger.info(f"Marginal predictive variances shape: {marginal_vars.shape}") # (B, C)
        
        logger.info("Sample predictive mean (first item):")
        logger.info(f"  {y_bar[0]}")
        
        logger.info("Sample marginal variances (first item):")
        logger.info(f"  {marginal_vars[0]}")
        
        logger.info("Modality-attributed uncertainties:")
        for mod_name, uncertainty in modality_uncertainties.items():
            logger.info(f"  U_{mod_name} (first item): {uncertainty[0].item():.4f}")
            
        # Example: Find the modality with highest attributed uncertainty for the first sample
        first_sample_uncertainties = {k: v[0].item() for k, v in modality_uncertainties.items()}
        most_uncertain_modality = max(first_sample_uncertainties, key=first_sample_uncertainties.get)
        logger.info(f"For the first sample, the modality contributing most to uncertainty is: {most_uncertain_modality} "
                    f"(U = {first_sample_uncertainties[most_uncertain_modality]:.4f})")

    except Exception as e:
        logger.error(f"Error during uncertainty quantification: {e}")
