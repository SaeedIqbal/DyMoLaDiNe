import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReliabilityScorer(nn.Module):
    """
    Reliability Scoring Network (Rel_ψ).
    
    Predicts a reliability score vector s(m') = [s1, ..., sn, st, sc] for a given
    multi-modal input m'. Each s_i is in [0, 1] and represents the estimated
    reliability of modality i.
    """
    def __init__(
        self,
        modality_dims: Dict[str, int],  # e.g., {'image': 768, 'text': 512, 'tabular': 256}
        output_dim: int,                # n + 2 (number of modalities)
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.2
    ):
        """
        Initializes the Reliability Scorer.

        Args:
            modality_dims (Dict[str, int]): A dictionary mapping modality names to their feature dimensions.
            output_dim (int): The number of modalities (n + 2).
            hidden_dims (List[int], optional): Dimensions of the hidden layers. Defaults to [512, 256].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(ReliabilityScorer, self).__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        # Calculate total input dimension by summing dimensions of all modalities
        total_input_dim = sum(modality_dims.values())
        
        # Build the MLP layers
        layers = []
        input_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            
        # Final layer to output scores
        layers.append(nn.Linear(input_dim, output_dim))
        # Use Sigmoid to ensure outputs are in [0, 1]
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized ReliabilityScorer with input dim {total_input_dim}, output dim {output_dim}")

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to compute reliability scores.

        Args:
            modality_features (Dict[str, torch.Tensor]): A dictionary where keys are modality names
                                                         and values are feature tensors of shape (batch_size, dim).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, output_dim) containing reliability scores s(m') in [0, 1].
        """
        # Concatenate features from all modalities
        # Ensure consistent ordering
        ordered_features = [modality_features[key] for key in sorted(modality_features.keys())]
        x = torch.cat(ordered_features, dim=1)  # Shape: (batch_size, total_input_dim)
        
        # Pass through MLP
        scores = self.mlp(x)  # Shape: (batch_size, output_dim)
        return scores


class DynamicWeightingFunction(nn.Module):
    """
    Dynamic Weighting Function (f_ω).
    
    Computes dynamic mixture weights π_k(m'; ω) = Softmax(f_ω(s(m')))_k
    based on the reliability scores s(m').
    """
    def __init__(
        self,
        input_dim: int,  # Dimension of s(m'), i.e., n + 2
        num_ensemble_members: int, # K
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.1
    ):
        """
        Initializes the Dynamic Weighting Function.

        Args:
            input_dim (int): Dimension of the reliability score vector (n + 2).
            num_ensemble_members (int): Number of ensemble members (K).
            hidden_dims (List[int], optional): Dimensions of the hidden layers. Defaults to [128, 64].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(DynamicWeightingFunction, self).__init__()
        self.input_dim = input_dim
        self.num_ensemble_members = num_ensemble_members
        
        # Build the MLP layers
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        # Final layer to output logits for K ensemble members
        layers.append(nn.Linear(current_dim, num_ensemble_members))
        # Softmax is applied outside the model during use
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized DynamicWeightingFunction with input dim {input_dim}, output dim {num_ensemble_members}")

    def forward(self, reliability_scores: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute logits for mixture weights.

        Args:
            reliability_scores (torch.Tensor): Reliability scores s(m') of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits for mixture weights of shape (batch_size, num_ensemble_members).
                          Apply Softmax to get π_k(m'; ω).
        """
        logits = self.mlp(reliability_scores)  # Shape: (batch_size, num_ensemble_members)
        return logits


class DynamicModalityWeightingMechanism(nn.Module):
    """
    Dynamic Modality Weighting Mechanism.
    
    Orchestrates the Reliability Scorer and Dynamic Weighting Function.
    Integrates reliability scores into CDM conditioning.
    """
    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_ensemble_members: int, # K
        reliability_scorer_hidden_dims: List[int] = [512, 256],
        weighting_function_hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initializes the complete Dynamic Modality Weighting Mechanism.

        Args:
            modality_dims (Dict[str, int]): A dictionary mapping modality names to their feature dimensions.
            num_ensemble_members (int): Number of ensemble members (K).
            reliability_scorer_hidden_dims (List[int], optional): Hidden dims for Rel_ψ. Defaults to [512, 256].
            weighting_function_hidden_dims (List[int], optional): Hidden dims for f_ω. Defaults to [128, 64].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(DynamicModalityWeightingMechanism, self).__init__()
        self.num_ensemble_members = num_ensemble_members
        self.num_modalities = len(modality_dims)
        
        # 1. Initialize Reliability Scorer (Rel_ψ)
        self.reliability_scorer = ReliabilityScorer(
            modality_dims=modality_dims,
            output_dim=self.num_modalities, # n + 2, assuming text/tabular are part of n or handled separately
            hidden_dims=reliability_scorer_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # 2. Initialize Dynamic Weighting Function (f_ω)
        # The input to f_ω is the output of Rel_ψ
        self.weighting_function = DynamicWeightingFunction(
            input_dim=self.num_modalities,
            num_ensemble_members=num_ensemble_members,
            hidden_dims=weighting_function_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        logger.info(f"Initialized DynamicModalityWeightingMechanism with {num_ensemble_members} ensemble members and {self.num_modalities} modalities")

    def compute_reliability_scores(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the reliability score vector s(m') for a batch of inputs.

        Args:
            modality_features (Dict[str, torch.Tensor]): Dictionary of modality features.

        Returns:
            torch.Tensor: Reliability scores s(m') of shape (batch_size, num_modalities).
        """
        return self.reliability_scorer(modality_features)

    def compute_dynamic_weights(self, reliability_scores: torch.Tensor) -> torch.Tensor:
        """
        Computes the dynamic mixture weights π_k(m'; ω) using Softmax.

        Args:
            reliability_scores (torch.Tensor): Scores s(m') of shape (batch_size, num_modalities).

        Returns:
            torch.Tensor: Mixture weights π_k(m'; ω) of shape (batch_size, num_ensemble_members).
        """
        logits = self.weighting_function(reliability_scores)
        weights = F.softmax(logits, dim=1) # Apply Softmax along ensemble dimension
        return weights

    def prepare_cdm_conditioning(
        self,
        z_common: torch.Tensor,
        original_input: Dict[str, torch.Tensor], # m'
        reliability_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares the conditioning dictionary for the CDMs.
        Modifies p_θ_k(Y|z_common, m') to p_θ_k(Y|z_common, m', s(m')).

        Args:
            z_common (torch.Tensor): Common latent representation of shape (batch_size, d_z).
            original_input (Dict[str, torch.Tensor]): The original multi-modal input m'.
            reliability_scores (Optional[torch.Tensor], optional): Pre-computed s(m'). If None, computes it.
                                                              Shape (batch_size, num_modalities).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing conditioning variables for the CDM.
                                     Includes 'z_common', original modalities, and 'reliability_scores'.
        """
        if reliability_scores is None:
            # This requires access to modality features, which should be part of original_input
            # or derived from it. For simplicity, we assume features are passed in original_input
            # or the scorer can process original_input directly.
            # A more robust design might pass the MM-Enc features or individual encodings.
            # Here, we assume original_input contains the features needed by the scorer.
            # In a full integration, you'd pass the relevant embeddings.
            # Let's assume we pass the raw data and the scorer handles preprocessing internally,
            # or we pass pre-extracted features.
            # For this example, let's assume `original_input` already contains the features
            # compatible with the scorer's `forward` method.
            # A placeholder approach: if features not directly usable, one would need to
            # re-encode or pass them through the respective encoders (Enc^(i), Enc^(t), Enc^(c)).
            # For now, we'll pass `original_input` directly, acknowledging this is a simplification.
            # A better design would be to pass the extracted features from MM-Enc or individual encoders.
            # Let's revise the assumption: we expect `original_input` to contain pre-extracted features
            # for each modality that the ReliabilityScorer expects.
            # e.g., original_input = {'image_features': ..., 'text_features': ..., 'tabular_features': ...}
            # which matches the keys expected by ReliabilityScorer.
            
            # Check if features are already in the expected format
            try:
                reliability_scores = self.compute_reliability_scores(original_input)
            except Exception as e:
                logger.error(f"Failed to compute reliability scores from original_input: {e}")
                # Fallback or raise error
                raise ValueError("original_input must contain features compatible with ReliabilityScorer. "
                                 "Ensure keys match modality_dims keys and values are tensors of correct shape.") from e

        # Prepare conditioning dictionary for CDM
        # This is a conceptual representation. The actual CDM implementation will
        # need to handle how these are combined (e.g., concatenation, element-wise product).
        conditioning_dict = {
            'z_common': z_common, # Latent from MM-ViT + mapping network
            # Include original modalities if needed by CDM
            # The CDM conditioning logic (e.g., element-wise product with time embedding)
            # will handle how these are used.
            # 'original_input': original_input, # Optional: pass raw input if CDM needs it
            'reliability_scores': reliability_scores # The new conditioning variable s(m')
        }
        
        # Note: The actual integration into the CDM's forward pass (e.g., how s(m') modifies
        # the noise prediction or denoising process) is handled within the CDM class itself,
        # likely by modifying how the conditioning signal is processed (e.g., concatenating
        # s(m') to the time embedding or z_common before passing to the CDM's neural network).
        # This function provides the necessary components.
        
        return conditioning_dict

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor], # Features from MM-Enc or individual encoders
        z_common: torch.Tensor,
        original_input: Optional[Dict[str, torch.Tensor]] = None # Raw input m' if needed by CDM
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs a full forward pass of the Dynamic Modality Weighting Mechanism.

        Args:
            modality_features (Dict[str, torch.Tensor]): Dictionary of modality features for the scorer.
            z_common (torch.Tensor): Common latent representation of shape (batch_size, d_z).
            original_input (Optional[Dict[str, torch.Tensor]], optional): The original multi-modal input m'.
                                                                        Needed if CDM requires raw input or if
                                                                        scores need to be recomputed.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                - Reliability scores s(m') of shape (batch_size, num_modalities).
                - Dynamic mixture weights π_k(m'; ω) of shape (batch_size, num_ensemble_members).
                - Conditioning dictionary for CDMs.
        """
        # 1. Compute reliability scores s(m')
        s_m_prime = self.compute_reliability_scores(modality_features)
        
        # 2. Compute dynamic mixture weights π_k(m'; ω)
        pi_k_m_omega = self.compute_dynamic_weights(s_m_prime)
        
        # 3. Prepare conditioning for CDMs
        # If original_input is not provided, we assume the features passed are sufficient
        # and the CDM doesn't need the raw input. Otherwise, pass it.
        cdm_conditioning = self.prepare_cdm_conditioning(
            z_common=z_common,
            original_input=original_input if original_input is not None else modality_features,
            reliability_scores=s_m_prime # Pass pre-computed scores
        )
        
        return s_m_prime, pi_k_m_omega, cdm_conditioning


# --- Example Usage ---
if __name__ == "__main__":
    # Example configuration
    batch_size = 4
    modality_dims_example = {
        'image_features': 768,   # d_{e_i}
        'text_features': 512,    # d_{e_c}
        'tabular_features': 256  # d_{e_t}
    }
    num_ensemble_members_example = 5 # K
    latent_dim = 512 # d_z

    # Create dummy input data (features extracted by MM-ViT or individual encoders)
    dummy_modality_features = {
        'image_features': torch.randn(batch_size, modality_dims_example['image_features']),
        'text_features': torch.randn(batch_size, modality_dims_example['text_features']),
        'tabular_features': torch.randn(batch_size, modality_dims_example['tabular_features'])
    }
    
    # Dummy z_common (output of g_ϕ_k(ek))
    dummy_z_common = torch.randn(batch_size, latent_dim)
    
    # Dummy original input (raw data, if CDM needs it)
    dummy_original_input = {
        # In a real scenario, this might be the raw tensors
        # For this example, we'll pass the features again, or leave as None
        # if the CDM conditioning doesn't require it.
        # 'image': torch.randn(batch_size, 3, 224, 224),
        # 'text': torch.randint(0, 1000, (batch_size, 512)),
        # 'tabular': torch.randn(batch_size, 50)
        # For this demo, we pass features to show the mechanism
        'image_features': dummy_modality_features['image_features'],
        'text_features': dummy_modality_features['text_features'],
        'tabular_features': dummy_modality_features['tabular_features']
    }

    # Instantiate the mechanism
    dmw_mechanism = DynamicModalityWeightingMechanism(
        modality_dims=modality_dims_example,
        num_ensemble_members=num_ensemble_members_example,
        reliability_scorer_hidden_dims=[512, 256],
        weighting_function_hidden_dims=[128, 64],
        dropout_rate=0.2
    )

    # Run forward pass
    try:
        s_scores, weights, cdm_cond = dmw_mechanism(
            modality_features=dummy_modality_features,
            z_common=dummy_z_common,
            original_input=dummy_original_input
        )
        
        logger.info(f"Input batch size: {batch_size}")
        logger.info(f"Reliability scores (s(m')) shape: {s_scores.shape}") # (batch_size, 3)
        logger.info(f"Dynamic weights (π_k(m'; ω)) shape: {weights.shape}") # (batch_size, 5)
        logger.info(f"CDM conditioning keys: {list(cdm_cond.keys())}")
        logger.info(f"z_common shape in conditioning: {cdm_cond['z_common'].shape}")
        logger.info(f"reliability_scores shape in conditioning: {cdm_cond['reliability_scores'].shape}")
        logger.info(f"Sample reliability scores (first item): {s_scores[0]}")
        logger.info(f"Sample dynamic weights (first item, should sum to 1): {weights[0]}, sum={weights[0].sum().item():.4f}")
        
    except Exception as e:
        logger.error(f"Error during forward pass: {e}")
