import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityAttributedUncertaintyQuantifier:
    """
    Framework for Modality-Attributed Uncertainty Quantification.
    
    Generates Testing Samples from the full model and ablated models,
    computes total predictive uncertainty, and decomposes it
    by modality influence using ablation sampling.
    Aligns with concepts from LaDiNE (CPIW, CNPV) and extends them
    to attribute uncertainty to specific input modalities in a multi-modal setting.
    """
    def __init__(
        self,
        num_classes: int, # C
        modality_names: List[str], # e.g., ['image1', 'image2', 'tabular', 'text']
        baseline_generator: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None,
        sharpness_param: float = 0.1737 # ι from LaDiNE, can be dataset-specific
    ):
        """
        Initializes the uncertainty quantifier.

        Args:
            num_classes (int): Number of classes (C).
            modality_names (List[str]): List of modality names present in the input data.
            baseline_generator (Optional[Callable]): Function to generate a baseline
                for a modality. Takes (modality_name, original_tensor) and returns baseline.
                If None, uses zero tensor.
            sharpness_param (float, optional): Sharpness parameter (ι) for mapping
                Testing Samples to the probability simplex (LaDiNE Eq. 26). Defaults to 0.1737.
        """
        self.num_classes = num_classes
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        self.sharpness_param = sharpness_param
        
        if baseline_generator is None:
            # Default: zero tensor baseline (common choice)
            self.baseline_generator = lambda name, tensor: torch.zeros_like(tensor)
        else:
            self.baseline_generator = baseline_generator
            
        logger.info(f"Initialized ModalityAttributedUncertaintyQuantifier for {self.num_modalities} modalities: {modality_names}")

    def compute_predictive_statistics(
        self, 
        final_predictions: torch.Tensor, # Shape (B, N, C) or (B, C) if already averaged
        Testing Samples: Optional[torch.Tensor] = None # Shape (B, N, C) if needed for variance
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes predictive mean, covariance, and marginal variances.
        If Testing Samples are provided, computes variance-based metrics (CPIW, CNPV analogs).
        If only final_predictions are provided, treats them as the mean.

        Args:
            final_predictions (torch.Tensor): Final predictions, typically from ensemble averaging.
                                                Shape (B, C) or (B, N, C).
            Testing Samples (Optional[torch.Tensor]): Raw Monte Carlo Testing Samples from the model.
                                              Shape (B, N, C). Required for CPIW/CNPV.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - Predictive mean y_bar: (B, C)
                - Marginal predictive variances Var[Y_a|m'] or CPIW-like: (B, C) or None
                - CPIW-like metric per class: (B, C) or None
        """
        if final_predictions.dim() == 3:
             # Assume final_predictions is (B, N, C) - e.g., from one ensemble member's Testing Samples
            # Or it's already averaged Testing Samples (B, N, C) where N is Testing Samples per member
            # For consistency with LaDiNE CPIW/CNPV, we treat Testing Samples as the source
            if Testing Samples is not None and Testing Samples is not final_predictions:
                logger.warning("Both final_predictions (3D) and Testing Samples provided. Using Testing Samples for statistics.")
            effective_Testing Samples = final_predictions if Testing Samples is None else samples
            y_bar = torch.mean(effective_samples, dim=1) # (B, C)
        elif final_predictions.dim() == 2:
            # Assume final_predictions is the already computed mean (B, C)
            y_bar = final_predictions # (B, C)
            effective_samples = samples # Use provided samples if available
        else:
            raise ValueError(f"final_predictions must be 2D or 3D, got shape {final_predictions.shape}")

        marginal_vars = None
        cpiw_like = None

        if effective_samples is not None and effective_samples.dim() == 3:
            B, N, C = effective_samples.shape
            if N > 1:
                # --- Compute Marginal Predictive Variance (LaDiNE-style CNPV base) ---
                # LaDiNE Eq. (32) for CNPV is normalized variance of samples for a class.
                # CNPVa = (1/|Sa|) * sum(4 * (y_ia - y_bar_a)^2)
                # We compute a similar variance per class per sample.
                # Let's compute the variance per class across samples for each batch item.
                # This is analogous to marginal predictive variance.
                
                # Center samples
                centered_samples = effective_samples - y_bar.unsqueeze(1) # (B, N, C)
                # Compute squared deviations
                squared_deviations = centered_samples ** 2 # (B, N, C)
                # Sum over samples and normalize
                # This gives us the variance estimate for each class for each batch item.
                marginal_vars = torch.mean(squared_deviations, dim=1) # (B, C)
                # LaDiNE multiplies by 4 in CNPV formula, we can do that or keep raw variance.
                # marginal_vars = 4 * marginal_vars # Optional scaling to match LaDiNE CNPV style

                # --- Compute CPIW-like metric (LaDiNE Eq. 31) ---
                # CPIWa = Q97.5(Sa) - Q2.5(Sa)
                # We compute this per class per batch item.
                # PyTorch's quantile function can do this.
                # q_values shape: (B, 2, C) where 2 is for 2.5% and 97.5% quantiles
                q_values = torch.quantile(effective_samples, torch.tensor([0.025, 0.975], device=effective_samples.device), dim=1)
                cpiw_like = q_values[:, 1, :] - q_values[:, 0, :] # (B, C)
                # Note: This is CPIW per class, not a single scalar. LaDiNE reports it per class.
                
            else:
                logger.warning("Only one sample provided, cannot compute variance or CPIW.")
                marginal_vars = torch.zeros_like(y_bar)
                cpiw_like = torch.zeros_like(y_bar)
        else:
             # If no samples or samples not 3D, we cannot compute variance-based metrics
             # LaDiNE's CPIW and CNPV require the samples S.
            logger.info("Samples not provided or not 3D, skipping variance/CPIW computation.")
            marginal_vars = None # Cannot compute without samples
            cpiw_like = None # Cannot compute without samples
            
        return y_bar, marginal_vars, cpiw_like

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
        if ablated_modality_name not in self.modality_names:
             logger.warning(f"Ablating modality {ablated_modality_name} which is not in the quantifier's modality list {self.modality_names}. This might be intentional for nested ablations.")
            
        ablated_input = {}
        for mod_name, mod_tensor in original_input.items():
            if mod_name == ablated_modality_name:
                # Replace with baseline
                ablated_input[mod_name] = self.baseline_generator(mod_name, mod_tensor)
                logger.debug(f"  Ablated modality '{mod_name}' with baseline.")
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
        This is a first-order Sobol index approximation based on ablation.
        U_{x^(i)} = ||\bar{y} - \bar{y}^{(-i)}||_2^2

        Args:
            predictive_mean (torch.Tensor): \bar{y}, mean prediction from full model, shape (B, C).
            ablated_predictive_means (Dict[str, torch.Tensor]): Dict of \bar{y}^{(-i)} from ablated models, shape (B, C) each.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping modality names to their attributed
                                     uncertainty U_{x^(i)} of shape (B,).
        """
        modality_uncertainties = {}
        for mod_name, ablated_mean in ablated_predictive_means.items():
            # Compute squared L2 norm of difference (LaDiNE-style attribution)
            # U_x^(i) = ||\bar{y} - \bar{y}^{(-i)}||_2^2
            diff = predictive_mean - ablated_mean # (B, C)
            # Sum of squares across classes, for each item in batch
            # This gives a scalar uncertainty per sample per modality.
            uncertainty = torch.sum(diff**2, dim=1) # (B,)
            modality_uncertainties[mod_name] = uncertainty
            logger.debug(f"  Computed attributed uncertainty for '{mod_name}': shape {uncertainty.shape}")
            
        return modality_uncertainties

    def identify_highest_uncertainty_modality(
        self,
        modality_uncertainties: Dict[str, torch.Tensor]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Identifies the modality contributing the most uncertainty for each sample.

        Args:
            modality_uncertainties (Dict[str, torch.Tensor]): Uncertainties U_{x^(i)} for each modality.

        Returns:
            Tuple[List[str], torch.Tensor]: A list of modality names (one per sample)
                                            and a tensor of the corresponding maximum uncertainty values.
        """
        if not modality_uncertainties:
            return [], torch.tensor([])

        # Stack uncertainties into a single tensor (B, num_ablated_modalities)
        # We assume all modalities were ablated and are present in the dict.
        # Order should be consistent, e.g., based on self.modality_names
        # Let's filter and order based on self.modality_names for consistency.
        ordered_mod_names = [name for name in self.modality_names if name in modality_uncertainties]
        if not ordered_mod_names:
            logger.warning("No matching modalities found for uncertainty identification.")
            return [], torch.tensor([])
            
        uncertainties_list = [modality_uncertainties[name] for name in ordered_mod_names]
        uncertainties_tensor = torch.stack(uncertainties_list, dim=1) # (B, num_ablated_modalities)
        
        # Find the index of the maximum uncertainty for each sample
        max_indices = torch.argmax(uncertainties_tensor, dim=1) # (B,)
        max_values = torch.gather(uncertainties_tensor, 1, max_indices.unsqueeze(1)).squeeze(1) # (B,)
        
        # Map indices back to modality names
        most_uncertain_modalities = [ordered_mod_names[idx.item()] for idx in max_indices]
        
        return most_uncertain_modalities, max_values

    def quantify(
        self,
        # The core function that performs sampling is passed in.
        # This decouples the quantifier from the specific ensemble implementation.
        # It should take an input dict and return samples or final predictions.
        sample_generator: Callable[[Dict[str, torch.Tensor]], Tuple[torch.Tensor, Optional[List[torch.Tensor]]]],
        original_input: Dict[str, torch.Tensor], # m'
        # Note: Unlike other components, this one primarily works with the input m'
        # and the model's sampling interface. z_common, s(m'), π_k(m') are internal to the generator.
    ) -> Dict[str, Any]:
        """
        Performs the full modality-attributed uncertainty quantification pipeline.

        1. Generate samples/predictions from the full model.
        2. Compute predictive statistics (mean, potentially var/CPIW if samples are returned).
        3. For each modality, perform ablation sampling.
        4. Generate samples/predictions from each ablated model.
        5. Compute predictive means for ablated models.
        6. Decompose uncertainty by computing U_{x^(i)}.
        7. Identify the modality of highest uncertainty.

        Args:
            sample_generator (Callable): A function that takes an input dict and returns
                                         (final_predictions (B, C), optional raw_samples_list or tensor (B, N, C)).
                                         This function encapsulates the DyMoLaDiNE model's sampling logic.
            original_input (Dict[str, torch.Tensor]): The original multi-modal input m'.

        Returns:
            Dict[str, Any]: A dictionary containing all quantification results:
                - 'predictive_mean': (B, C)
                - 'marginal_vars': (B, C) or None
                - 'cpiw_like': (B, C) or None
                - 'modality_uncertainties': Dict[str, (B,)]
                - 'most_uncertain_modalities': List[str] (length B)
                - 'max_attributed_uncertainties': (B,)
                - 'ablated_predictive_means': Dict[str, (B, C)]
        """
        logger.info("Starting modality-attributed uncertainty quantification...")
        batch_size = list(original_input.values())[0].shape[0]
        
        # 1. Generate samples/predictions from the FULL model p(Y|m', \Psi)
        # The sample_generator is responsible for using the ensemble, CDMs, DMW mechanism internally
        # to produce the final predictions and optionally raw samples.
        # Expected outputs:
        # - final_predictions: (B, C) - The final class probabilities after ensemble averaging and simplex mapping.
        # - raw_samples: Optional - (B, N, C) or List[(B, M, C)] - Raw samples before averaging/simplex mapping.
        try:
            final_predictions_full, raw_samples_full = sample_generator(original_input)
            logger.info(f"Generated full model predictions with shape: {final_predictions_full.shape}")
            if raw_samples_full is not None:
                if isinstance(raw_samples_full, list):
                    logger.info(f"  Raw samples returned as list of length {len(raw_samples_full)}")
                else:
                    logger.info(f"  Raw samples returned as tensor with shape: {raw_samples_full.shape}")
        except Exception as e:
            logger.error(f"Error in sample_generator for full model: {e}")
            raise

        # 2. Compute predictive statistics for the FULL model
        y_bar_full, marginal_vars_full, cpiw_like_full = self.compute_predictive_statistics(
            final_predictions_full, raw_samples_full
        )
        logger.info("Computed predictive statistics for full model.")

        # 3. Perform ablation sampling and compute statistics for each ABLATED model
        ablated_predictive_means = {}
        for mod_name in self.modality_names:
            logger.info(f"Performing ablation for modality: {mod_name}")
            # a. Create ablated input m'^{(-i)}
            ablated_input = self.perform_ablation_sampling(original_input, mod_name)
            
            # b. Generate samples/predictions from the ABLATED model p(Y|m'^{(-i)}, \Psi)
            # The sample_generator should handle the ablated input internally.
            # It might recompute s(m'^{(-i)}) or use a flag, but that's its internal logic.
            try:
                final_predictions_ablated, _ = sample_generator(ablated_input)
                logger.info(f"  Generated ablated model predictions with shape: {final_predictions_ablated.shape}")
            except Exception as e:
                logger.error(f"Error in sample_generator for ablated model '{mod_name}': {e}")
                # Depending on requirements, you might want to skip this modality or raise the error.
                # For robustness, let's log and continue, marking this modality's mean as zero or NaN.
                # Here, we'll re-raise to halt the process if a core component fails.
                raise

            # c. Compute predictive mean for the ablated model \bar{y}^{(-i)}
            # Ablated model should also return final predictions (B, C)
            y_bar_ablated, _, _ = self.compute_predictive_statistics(final_predictions_ablated)
            ablated_predictive_means[mod_name] = y_bar_ablated
            logger.info(f"  Computed ablated predictive mean for '{mod_name}'.")

        # 4. Decompose uncertainty by computing U_{x^(i)}
        modality_uncertainties = self.compute_modality_attributed_uncertainty(
            predictive_mean=y_bar_full,
            ablated_predictive_means=ablated_predictive_means
        )
        logger.info("Computed modality-attributed uncertainties.")

        # 5. Identify the modality with highest attributed uncertainty for each sample
        most_uncertain_modalities, max_attributed_uncertainties = self.identify_highest_uncertainty_modality(
            modality_uncertainties
        )
        logger.info("Identified modalities contributing most to uncertainty.")

        logger.info("Finished modality-attributed uncertainty quantification.")
        
        return {
            'predictive_mean': y_bar_full, # (B, C)
            'marginal_vars': marginal_vars_full, # (B, C) or None
            'cpiw_like': cpiw_like_full, # (B, C) or None
            'modality_uncertainties': modality_uncertainties, # {str: (B,)}
            'most_uncertain_modalities': most_uncertain_modalities, # List[str] len B
            'max_attributed_uncertainties': max_attributed_uncertainties, # (B,)
            'ablated_predictive_means': ablated_predictive_means # {str: (B, C)}
        }


# --- Example Usage ---
# This example assumes the existence of other components (MM-ViT, DMW, CDM Ensemble)
# and a way to integrate them into a `sample_generator` function.

if __name__ == "__main__":
    # --- 1. Mock Components and Sample Generator ---
    # In a real scenario, these would be your trained DyMoLaDiNE models.
    from types import SimpleNamespace # For mock ensemble
    
    # Example configuration
    batch_size = 3
    num_classes = 2 # Binary classification (C)
    modality_names_example = ['image_features', 'text_features', 'tabular_features']
    num_samples_per_member = 10 # M (smaller for demo)
    num_ensemble_members = 3 # K
    
    # A mock sample generator function to simulate DyMoLaDiNE sampling
    # This would normally use the trained RobustMultiModalDiffusionEnsemble
    # and other components.
    def mock_sample_generator(input_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Mock function to simulate the DyMoLaDiNE sampling process.
        In a real scenario, this would use the trained model components.
        Returns final predictions and raw samples.
        """
        B = list(input_dict.values())[0].shape[0] # Get batch size from input
        C = num_classes
        
        # --- Simulate raw samples from ensemble (K members, M samples each) ---
        # This is the core output needed for uncertainty quantification.
        # Shape: List of K tensors, each (B, M, C)
        raw_samples_list = []
        for k in range(num_ensemble_members):
            # Generate random samples for this member (this is just for demonstration)
            # In DyMoLaDiNE, these come from the reverse diffusion process of the k-th CDM.
            member_samples = torch.randn(B, num_samples_per_member, C)
            # Map to simplex using LaDiNE Eq. (26) style (simplified here)
            # In practice, this mapping happens *after* averaging all K*M samples.
            # But for demo, we do it per member.
            # Let's NOT map to simplex here, return raw samples for variance calc.
            # The quantifier's compute_predictive_statistics will handle averaging and mapping.
            raw_samples_list.append(member_samples)
        
        # --- Simulate final prediction (what the model actually outputs) ---
        # This is the average of all K*M samples, mapped to simplex.
        # Shape: (B, C)
        # For mock, we'll just average the raw samples crudely and apply softmax.
        # Flatten list to (B, K*M, C)
        all_samples_flat = torch.cat(raw_samples_list, dim=1) # (B, K*M, C)
        # Average over samples
        avg_samples = torch.mean(all_samples_flat, dim=1) # (B, C)
        # Map to simplex (mock LaDiNE Eq. 26)
        logits_for_simplex = - (1.0 / 0.1737) * (avg_samples - 1.0) ** 2
        final_predictions = F.softmax(logits_for_simplex, dim=-1) # (B, C)
        
        logger.info(f"  Mock sample generator produced final predictions {final_predictions.shape} "
                    f"and raw samples list (len {len(raw_samples_list)}) of shape {raw_samples_list[0].shape}")
        # Return final predictions and the raw samples (as list)
        return final_predictions, raw_samples_list


    # --- 2. Create testing original input data ---
    testing_original_input = {
        'image_features': torch.randn(batch_size, 768),
        'text_features': torch.randn(batch_size, 512),
        'tabular_features': torch.randn(batch_size, 100) # Assuming 100 tabular features
    }

    # --- 3. Instantiate the quantifier ---
    uncertainty_quantifier = ModalityAttributedUncertaintyQuantifier(
        num_classes=num_classes,
        modality_names=modality_names_example,
        # Use default zero baseline
        sharpness_param=0.1737 # Example from LaDiNE
    )

    # --- 4. Run the uncertainty quantification ---
    try:
        results = uncertainty_quantifier.quantify(
            sample_generator=mock_sample_generator,
            original_input=testing_original_input,
        )
        
        # --- 5. Inspect the results ---
        logger.info("\n--- Uncertainty Quantification Results ---")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Modalities analyzed: {modality_names_example}")
        
        logger.info(f"\nPredictive mean (y_bar) shape: {results['predictive_mean'].shape}") # (B, C)
        logger.info(f"Testing Sample predictive mean (first item):\n{results['predictive_mean'][0]}")
        logger.info(f"  Sum of first prediction: {results['predictive_mean'][0].sum().item():.4f}")

        if results['marginal_vars'] is not None:
            logger.info(f"\nMarginal predictive variances (CNPV-like) shape: {results['marginal_vars'].shape}") # (B, C)
            logger.info(f"Testing Sample marginal vars (first item):\n{results['marginal_vars'][0]}")
        else:
            logger.info("\nMarginal predictive variances not computed (no raw Testing Samples provided to statistics function).")

        if results['cpiw_like'] is not None:
            logger.info(f"\nCPIW-like metric shape: {results['cpiw_like'].shape}") # (B, C)
            logger.info(f"Testing Sample CPIW-like (first item):\n{results['cpiw_like'][0]}")
        else:
             logger.info("\nCPIW-like metric not computed (no raw Testing Samples provided to statistics function).")

        logger.info("\nModality-attributed uncertainties:")
        for mod_name, uncertainty in results['modality_uncertainties'].items():
            logger.info(f"  U_{mod_name} (for batch): {uncertainty}") # (B,)

        logger.info(f"\nModality contributing most uncertainty per Testing Sample:")
        for i in range(batch_size):
            logger.info(f"  Testing Sample {i}: '{results['most_uncertain_modalities'][i]}' "
                        f"(U = {results['max_attributed_uncertainties'][i].item():.4f})")

        logger.info("\nAblated Predictive Means (first item for each modality):")
        for mod_name, ablated_mean in results['ablated_predictive_means'].items():
            logger.info(f"  y_bar^({mod_name})[0]: {ablated_mean[0]}")

        # Example: Detailed analysis for the first Testing Sample
        logger.info(f"\n--- Detailed Analysis for Testing Sample 0 ---")
        idx = 0
        logger.info(f"Final Prediction: {results['predictive_mean'][idx]}")
        logger.info(f"Most uncertain modality: {results['most_uncertain_modalities'][idx]} "
                    f"(U = {results['max_attributed_uncertainties'][idx].item():.4f})")
        logger.info("Attributed uncertainties:")
        for mod_name, uncertainty in results['modality_uncertainties'].items():
            logger.info(f"  U_{mod_name}: {uncertainty[idx].item():.4f}")
        logger.info("Ablated means:")
        for mod_name, ablated_mean in results['ablated_predictive_means'].items():
            diff_vector = results['predictive_mean'][idx] - ablated_mean[idx]
            diff_norm_sq = torch.sum(diff_vector ** 2).item()
            logger.info(f"  y_bar - y_bar^({mod_name}): ||diff||^2 = {diff_norm_sq:.4f}")


    except Exception as e:
        logger.error(f"Error during uncertainty quantification: {e}", exc_info=True)
