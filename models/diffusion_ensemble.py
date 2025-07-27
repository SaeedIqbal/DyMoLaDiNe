import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextEmbeddingNetwork(nn.Module):
    """
    Context Embedding Network (c_ζ).
    
    Maps the conditioning tuple (z_common, m', s(m')) into the same space as Y_t.
    This embedding is used to center the forward diffusion process and can be
    integrated into the reverse process (noise predictor).
    
    For DyMoLaDiNE, this is a key component for conditioning on multi-modal inputs
    and dynamic reliability scores.
    """
    def __init__(
        self,
        z_common_dim: int,           # d_z
        reliability_scores_dim: int, # n + 2 (number of modalities)
        output_dim: int,             # Should match the dimension of Y_t (number of classes)
        modality_dims: Optional[Dict[str, int]] = None, # Dimensions of raw modalities in m' if used
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.1
    ):
        """
        Initializes the Context Embedding Network.

        Args:
            z_common_dim (int): Dimension of the common latent variable z_common.
            reliability_scores_dim (int): Dimension of the reliability score vector s(m').
            output_dim (int): Dimension of the output embedding (should match Y_t dim).
            modality_dims (Optional[Dict[str, int]]): Dimensions of raw modalities in m'.
                                                    If None, raw modalities are not used.
            hidden_dims (List[int], optional): Hidden layer dimensions. Defaults to [512, 256].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(ContextEmbeddingNetwork, self).__init__()
        self.z_common_dim = z_common_dim
        self.reliability_scores_dim = reliability_scores_dim
        self.modality_dims = modality_dims or {}
        self.output_dim = output_dim
        
        # Calculate total input dimension
        total_input_dim = z_common_dim + reliability_scores_dim
        if self.modality_dims:
            total_input_dim += sum(self.modality_dims.values())
            
        # Build MLP layers
        layers = []
        input_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            
        # Final layer to output embedding
        layers.append(nn.Linear(input_dim, output_dim))
        # No final activation as it's an embedding that can be positive or negative
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized ContextEmbeddingNetwork with input dim {total_input_dim}, output dim {output_dim}")

    def forward(
        self,
        z_common: torch.Tensor,
        reliability_scores: torch.Tensor,
        raw_modalities: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute the context embedding.

        Args:
            z_common (torch.Tensor): Common latent representation, shape (B, d_z).
            reliability_scores (torch.Tensor): Reliability scores s(m'), shape (B, n+2).
            raw_modalities (Optional[Dict[str, torch.Tensor]]): Raw modality tensors from m'.
                                                              Shape for each key: (B, dim).

        Returns:
            torch.Tensor: Context embedding, shape (B, output_dim).
        """
        # Concatenate inputs
        inputs = [z_common, reliability_scores]
        if raw_modalities and self.modality_dims:
            # Ensure consistent ordering
            ordered_features = [raw_modalities[key] for key in sorted(raw_modalities.keys()) if key in self.modality_dims]
            inputs.extend(ordered_features)
            
        x = torch.cat(inputs, dim=1) # Shape: (B, total_input_dim)
        embedding = self.mlp(x)      # Shape: (B, output_dim)
        return embedding


class NoisePredictorNetwork(nn.Module):
    """
    Noise Predictor Network (ε_{θ_k, ζ}).
    
    Estimates the noise ϵ added at step t, given the noisy sample Y_t, time t,
    and the conditioning context (z_common, m', s(m')).
    
    This network is central to the reverse diffusion process and incorporates
    the dynamic conditioning from DyMoLaDiNE.
    """
    def __init__(
        self,
        y_dim: int,                  # Dimension of Y_t (number of classes)
        time_emb_dim: int,           # Dimension of time embedding
        context_emb_dim: int,        # Dimension of context embedding from c_ζ
        hidden_dims: List[int] = [1024, 512, 256], # As in LaDiNE for label space diffusion
        dropout_rate: float = 0.1,
        time_embedding_type: str = "sinusoidal" # Or "linear"
    ):
        """
        Initializes the Noise Predictor Network.

        Args:
            y_dim (int): Dimension of the noisy label vector Y_t.
            time_emb_dim (int): Dimension of the time step embedding.
            context_emb_dim (int): Dimension of the context embedding.
            hidden_dims (List[int], optional): Hidden layer dimensions. Defaults to [1024, 512, 256].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            time_embedding_type (str, optional): Type of time embedding. Defaults to "sinusoidal".
        """
        super(NoisePredictorNetwork, self).__init__()
        self.y_dim = y_dim
        self.time_emb_dim = time_emb_dim
        self.context_emb_dim = context_emb_dim
        self.time_embedding_type = time_embedding_type
        
        # Time embedding
        if time_embedding_type == "sinusoidal":
            # Placeholder, actual sinusoidal embedding is done in forward
            self.time_mlp = nn.Linear(time_emb_dim, time_emb_dim)
        else: # linear
            self.time_mlp = nn.Linear(1, time_emb_dim)
        
        # Total conditioning dimension after combining Y_t, time, and context
        total_cond_dim = y_dim + time_emb_dim + context_emb_dim
        
        # Build MLP layers
        layers = []
        input_dim = total_cond_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU()) # Using SiLU as in many diffusion implementations
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            
        # Final layer to predict noise (same dimension as Y_t)
        layers.append(nn.Linear(input_dim, y_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized NoisePredictorNetwork with total cond dim {total_cond_dim}, output dim {y_dim}")

    def forward(
        self,
        y_t: torch.Tensor,
        timesteps: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to predict the noise.

        Args:
            y_t (torch.Tensor): Noisy label sample at time t, shape (B, y_dim).
            timesteps (torch.Tensor): Time steps, shape (B,).
            context_embedding (torch.Tensor): Context embedding from c_ζ, shape (B, context_emb_dim).

        Returns:
            torch.Tensor: Predicted noise ϵ, shape (B, y_dim).
        """
        B = y_t.shape[0]
        
        # Embed time steps
        if self.time_embedding_type == "sinusoidal":
            # Simple sinusoidal embedding (can be made more complex)
            half_dim = self.time_emb_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
            emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
            if self.time_emb_dim % 2 == 1:  # zero pad
                emb = F.pad(emb, (0, 1))
            t_emb = self.time_mlp(emb) # Apply linear layer
        else: # linear
            t_emb = self.time_mlp(timesteps.float().unsqueeze(-1) / 1000.0) # Normalize timesteps
        
        # Concatenate all conditioning information
        cond = torch.cat([y_t, t_emb, context_embedding], dim=-1) # Shape: (B, total_cond_dim)
        
        # Predict noise
        predicted_noise = self.mlp(cond) # Shape: (B, y_dim)
        return predicted_noise


class ConditionalDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model (CDM) p_{θ_k}(Y|z_common, m', s(m')).
    
    Implements the forward and reverse processes, and the training objective.
    Extends the LaDiNE CDM concept to DyMoLaDiNE's multi-modal context.
    """
    def __init__(
        self,
        num_classes: int,            # Number of classes (dimension of Y)
        z_common_dim: int,           # d_z
        reliability_scores_dim: int, # n + 2
        modality_dims: Optional[Dict[str, int]] = None,
        time_emb_dim: int = 128,
        context_hidden_dims: List[int] = [512, 256],
        noise_predictor_hidden_dims: List[int] = [1024, 512, 256], # Match LaDiNE
        num_timesteps: int = 1000,
        beta_schedule: str = "linear", # or "cosine"
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        """
        Initializes the Conditional Diffusion Model.

        Args:
            num_classes (int): Number of classes.
            z_common_dim (int): Dimension of z_common.
            reliability_scores_dim (int): Dimension of s(m').
            modality_dims (Optional[Dict[str, int]]): Raw modality dimensions for context.
            time_emb_dim (int, optional): Time embedding dimension. Defaults to 128.
            context_hidden_dims (List[int], optional): Hidden dims for context emb. Defaults to [512, 256].
            noise_predictor_hidden_dims (List[int], optional): Hidden dims for noise pred. Defaults to [1024, 512, 256].
            num_timesteps (int, optional): Number of diffusion steps T. Defaults to 1000.
            beta_schedule (str, optional): Type of beta schedule. Defaults to "linear".
            beta_start (float, optional): Start of beta schedule. Defaults to 1e-4.
            beta_end (float, optional): End of beta schedule. Defaults to 0.02.
        """
        super(ConditionalDiffusionModel, self).__init__()
        self.num_classes = num_classes
        self.z_common_dim = z_common_dim
        self.reliability_scores_dim = reliability_scores_dim
        self.modality_dims = modality_dims or {}
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # 1. Context Embedding Network (c_ζ)
        self.context_embedding_net = ContextEmbeddingNetwork(
            z_common_dim=z_common_dim,
            reliability_scores_dim=reliability_scores_dim,
            output_dim=num_classes, # Match Y_t dimension
            modality_dims=modality_dims,
            hidden_dims=context_hidden_dims
        )
        
        # 2. Noise Predictor Network (ε_{θ_k, ζ})
        self.noise_predictor_net = NoisePredictorNetwork(
            y_dim=num_classes,
            time_emb_dim=time_emb_dim,
            context_emb_dim=num_classes, # Output of context embedding
            hidden_dims=noise_predictor_hidden_dims
        )
        
        # 3. Register buffers for precomputed alpha and beta schedules
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule as in Nichol & Dhariwal 2021
            def cosine_beta_schedule(timesteps, s=0.008):
                steps = timesteps + 1
                x = torch.linspace(0, timesteps, steps)
                alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                return torch.clip(betas, 0.0001, 0.9999)
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers so they are moved to device with the model
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute useful terms for sampling and loss (from DDPM)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Useful terms for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # Clip to 0 for t=0
        posterior_variance = torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        
        # Coefficients for posterior mean q(y_{t-1} | y_t, y_0)
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        logger.info(f"Initialized ConditionalDiffusionModel with {num_timesteps} timesteps and {beta_schedule} schedule")

    def q_sample(
        self,
        y_0: torch.Tensor,
        t: torch.Tensor,
        context_embedding: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: sample q(y_t | y_0, context).
        Extends LaDiNE Eq. (8) to include context embedding.

        y_t = sqrt(alpha_bar_t) * y_0 + (1 - sqrt(alpha_bar_t)) * context + sqrt(1 - alpha_bar_t) * noise

        Args:
            y_0 (torch.Tensor): Original one-hot label, shape (B, num_classes).
            t (torch.Tensor): Timesteps, shape (B,).
            context_embedding (torch.Tensor): Context embedding c_ζ(...), shape (B, num_classes).
            noise (Optional[torch.Tensor], optional): Pre-generated noise. Defaults to None.

        Returns:
            torch.Tensor: Noisy sample y_t, shape (B, num_classes).
        """
        if noise is None:
            noise = torch.randn_like(y_0)
            
        # Get alpha_bar_t for each sample in the batch
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod.gather(-1, t).unsqueeze(-1) # (B, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).unsqueeze(-1) # (B, 1)
        
        # Compute y_t using the modified forward process
        # This aligns with the formulation where the mean is shifted by the context
        y_t = (sqrt_alpha_bar_t * y_0 +
               (1 - sqrt_alpha_bar_t) * context_embedding +
               sqrt_one_minus_alpha_bar_t * noise)
        return y_t

    def p_losses(
        self,
        y_0: torch.Tensor, # One-hot labels
        z_common: torch.Tensor,
        reliability_scores: torch.Tensor,
        raw_modalities: Optional[Dict[str, torch.Tensor]],
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the simplified variational bound loss (L_CDM,k).
        Extends LaDiNE Eq. (15) to the multi-modal conditional setting.

        L_CDM,k = E[||ϵ - ϵ_θ(y_t, z_common, m', s(m'), t)||^2]

        Args:
            y_0 (torch.Tensor): True one-hot labels, shape (B, num_classes).
            z_common (torch.Tensor): Common latent, shape (B, d_z).
            reliability_scores (torch.Tensor): Scores s(m'), shape (B, n+2).
            raw_modalities (Optional[Dict[str, torch.Tensor]]): Raw m'.
            t (torch.Tensor): Timesteps, shape (B,).
            noise (Optional[torch.Tensor], optional): Pre-generated noise. Defaults to None.

        Returns:
            torch.Tensor: Loss value (scalar).
        """
        if noise is None:
            noise = torch.randn_like(y_0)
        
        # 1. Compute context embedding c_ζ(z_common, m', s(m'))
        context_emb = self.context_embedding_net(z_common, reliability_scores, raw_modalities)
        
        # 2. Sample y_t using the forward process q(y_t | y_0, context)
        y_t = self.q_sample(y_0, t, context_emb, noise)
        
        # 3. Predict noise using ε_θ(y_t, t, context)
        predicted_noise = self.noise_predictor_net(y_t, t, context_emb)
        
        # 4. Compute MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def predict_start_from_noise(self, y_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict y_0 from y_t and predicted noise, using the context.
        This is a helper for the reverse process sampling.
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.gather(-1, t).unsqueeze(-1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod.gather(-1, t).unsqueeze(-1)
        
        # Predict y_0
        y_0_pred = sqrt_recip_alphas_cumprod_t * (y_t - sqrt_recipm1_alphas_cumprod_t * noise)
        
        # The context embedding was part of the forward process mean.
        # In the reverse, we don't subtract it back as it's part of the conditioning.
        # The noise prediction already accounts for the context's influence.
        # This function primarily inverts the noise scaling part of the forward process.
        return y_0_pred

    def q_posterior(self, y_0_pred: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the posterior q(y_{t-1} | y_t, y_0_pred).
        """
        posterior_mean_coef1_t = self.posterior_mean_coef1.gather(-1, t).unsqueeze(-1)
        posterior_mean_coef2_t = self.posterior_mean_coef2.gather(-1, t).unsqueeze(-1)
        
        posterior_mean = posterior_mean_coef1_t * y_0_pred + posterior_mean_coef2_t * y_t
        posterior_variance = self.posterior_variance.gather(-1, t).unsqueeze(-1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped.gather(-1, t).unsqueeze(-1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, y_t: torch.Tensor, t: torch.Tensor, context_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the reverse process p(y_{t-1} | y_t, context).
        """
        # Predict noise
        predicted_noise = self.noise_predictor_net(y_t, t, context_embedding)
        # Predict y_0 from noise
        y_0_pred = self.predict_start_from_noise(y_t, t, predicted_noise, context_embedding)
        # Clamp y_0 predictions (important for stability)
        y_0_pred.clamp_(-1., 1.)
        
        # Get posterior q(y_{t-1} | y_t, y_0_pred) which is used as p(y_{t-1} | y_t, context)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(y_0_pred, y_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    def forward(self, *args, **kwargs):
        """Wrapper for p_losses during training."""
        return self.p_losses(*args, **kwargs)


class RobustMultiModalDiffusionEnsemble(nn.Module):
    """
    Robust Multi-Modal Diffusion Ensemble.
    
    Orchestrates K Conditional Diffusion Models and computes the final mixture prediction.
    Extends LaDiNE's ensemble concept to DyMoLaDiNE's multi-modal setting with dynamic weights.
    p(Y|m', Ψ) = Σ_k π_k(m') ∫ p_θ_k(Y|z_common, m', s(m')) ...
    """
    def __init__(
        self,
        num_ensemble_members: int,   # K
        num_classes: int,
        z_common_dim: int,           # d_z
        reliability_scores_dim: int, # n + 2
        modality_dims: Optional[Dict[str, int]] = None,
        # CDM parameters
        time_emb_dim: int = 128,
        context_hidden_dims: List[int] = [512, 256],
        noise_predictor_hidden_dims: List[int] = [1024, 512, 256], # Match LaDiNE
        num_timesteps: int = 1000,
        beta_schedule: str = "linear", # or "cosine"
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        """
        Initializes the Robust Multi-Modal Diffusion Ensemble.

        Args:
            num_ensemble_members (int): Number of ensemble members K.
            num_classes (int): Number of classes.
            z_common_dim (int): Dimension of z_common.
            reliability_scores_dim (int): Dimension of s(m').
            modality_dims (Optional[Dict[str, int]]): Raw modality dimensions.
            ... (other CDM parameters)
        """
        super(RobustMultiModalDiffusionEnsemble, self).__init__()
        self.num_ensemble_members = num_ensemble_members
        self.num_classes = num_classes
        
        # Create K ensemble members (CDMs)
        self.cdm_members = nn.ModuleList([
            ConditionalDiffusionModel(
                num_classes=num_classes,
                z_common_dim=z_common_dim,
                reliability_scores_dim=reliability_scores_dim,
                modality_dims=modality_dims,
                time_emb_dim=time_emb_dim,
                context_hidden_dims=context_hidden_dims,
                noise_predictor_hidden_dims=noise_predictor_hidden_dims,
                num_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end
            ) for _ in range(num_ensemble_members)
        ])
        
        logger.info(f"Initialized RobustMultiModalDiffusionEnsemble with {num_ensemble_members} members")

    def compute_loss(
        self,
        y_0: torch.Tensor, # One-hot labels
        z_common_list: List[torch.Tensor], # List of z_common for each member
        reliability_scores: torch.Tensor,
        raw_modalities: Optional[Dict[str, torch.Tensor]],
        dynamic_weights: torch.Tensor # π_k(m'; ω) from DynamicWeightingFunction, shape (B, K)
    ) -> torch.Tensor:
        """
        Compute the total loss for the ensemble, weighted by dynamic weights.

        Args:
            y_0 (torch.Tensor): True one-hot labels, shape (B, num_classes).
            z_common_list (List[torch.Tensor]): List of z_common for each member, each (B, d_z).
            reliability_scores (torch.Tensor): Scores s(m'), shape (B, n+2).
            raw_modalities (Optional[Dict[str, torch.Tensor]]): Raw m'.
            dynamic_weights (torch.Tensor): Weights π_k(m'; ω), shape (B, K).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if len(z_common_list) != self.num_ensemble_members:
            raise ValueError(f"z_common_list must have {self.num_ensemble_members} elements, got {len(z_common_list)}")
            
        total_loss = 0.0
        batch_size = y_0.shape[0]
        
        for k in range(self.num_ensemble_members):
            # Sample random timesteps for this member
            t = torch.randint(0, self.cdm_members[k].num_timesteps, (batch_size,), device=y_0.device).long()
            
            # Compute loss for this member
            member_loss = self.cdm_members[k](
                y_0=y_0,
                z_common=z_common_list[k], # Each member might have its own z_common
                reliability_scores=reliability_scores,
                raw_modalities=raw_modalities,
                t=t
            ) # Scalar loss
            
            # Weight the loss by π_k(m'; ω)
            # Average the weight across the batch for this member
            avg_weight = torch.mean(dynamic_weights[:, k])
            weighted_loss = avg_weight * member_loss
            total_loss += weighted_loss
        
        return total_loss / self.num_ensemble_members # Average over ensemble members

    def sample(
        self,
        z_common_list: List[torch.Tensor],
        reliability_scores: torch.Tensor,
        raw_modalities: Optional[Dict[str, torch.Tensor]],
        dynamic_weights: torch.Tensor, # π_k(m'; ω)
        num_samples_per_member: int = 20, # M
        # Optional: Provide a custom sampler function for advanced techniques (e.g., DDIM)
        custom_sampler: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sample from the ensemble to get final predictions.
        Extends LaDiNE's sampling (Sec. III-C, Algorithm 1) to the ensemble and multi-modal context.

        Args:
            z_common_list (List[torch.Tensor]): List of z_common for each member.
            reliability_scores (torch.Tensor): Scores s(m').
            raw_modalities (Optional[Dict[str, torch.Tensor]]): Raw m'.
            dynamic_weights (torch.Tensor): Weights π_k(m'; ω), shape (B, K).
            num_samples_per_member (int, optional): Number of samples M per member. Defaults to 20.
            custom_sampler (Optional[Callable], optional): Custom sampling function.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 
                - Final predictions after averaging and mapping to simplex, shape (B, num_classes).
                - List of raw sample tensors from each member, each (B, M, num_classes).
        """
        if len(z_common_list) != self.num_ensemble_members:
             raise ValueError(f"z_common_list must have {self.num_ensemble_members} elements, got {len(z_common_list)}")

        batch_size = z_common_list[0].shape[0]
        all_samples = []
        
        for k in range(self.num_ensemble_members):
            cdm = self.cdm_members[k]
            z_common_k = z_common_list[k]
            
            # 1. Compute context embedding for this member (used throughout sampling)
            context_emb = cdm.context_embedding_net(z_common_k, reliability_scores, raw_modalities)
            
            # 2. Generate M samples for this member
            member_samples = [] # List to hold M samples for this member
            
            for _ in range(num_samples_per_member):
                # Sample y_T ~ N(context_emb, I) # Simplified prior centered at context
                # LaDiNE samples from N(z, I), but our forward process uses context_emb as mean.
                # So, the prior should also be centered at context_emb for consistency.
                y_t = torch.randn((batch_size, self.num_classes), device=z_common_k.device) + context_emb
                
                # Iteratively denoise from T to 1 (Algorithm 1 style, adapted)
                for t in reversed(range(cdm.num_timesteps)):
                    t_tensor = torch.full((batch_size,), t, device=y_t.device, dtype=torch.long)
                    
                    if custom_sampler:
                        # Use a potentially more advanced sampler (e.g., DDIM)
                        y_t = custom_sampler(cdm, y_t, t_tensor, context_emb)
                    else:
                        # Standard DDPM reverse step
                        # Predict noise
                        predicted_noise = cdm.noise_predictor_net(y_t, t_tensor, context_emb)
                        
                        # Compute y_{t-1} using the reverse process mean and variance
                        if t > 0:
                            # Get coefficients
                            posterior_mean, posterior_variance, _ = cdm.p_mean_variance(y_t, t_tensor, context_emb)
                            
                            # Add noise for stochastic sampling
                            noise = torch.randn_like(y_t)
                            # Sigma_t is sqrt(posterior_variance)
                            y_t = posterior_mean + torch.sqrt(posterior_variance) * noise
                        else:
                            # Final step, t=0, no noise
                            posterior_mean, _, _ = cdm.p_mean_variance(y_t, t_tensor, context_emb)
                            y_t = posterior_mean
                            # No noise added at t=0
            
                # y_t is now the denoised sample y_0 (real-valued vector)
                member_samples.append(y_t)
            
            # Stack samples for this member: (M, B, num_classes)
            member_samples_tensor = torch.stack(member_samples, dim=0)
            # Transpose to (B, M, num_classes)
            member_samples_tensor = member_samples_tensor.transpose(0, 1)
            all_samples.append(member_samples_tensor)
        
        # 3. Aggregate samples from all K members
        # all_samples is a list of K tensors, each (B, M, num_classes)
        # We have K * M samples per batch item
        
        # Flatten the list of samples to a single tensor (B, K*M, num_classes)
        flat_samples = torch.cat(all_samples, dim=1) # (B, K*M, num_classes)
        
        # 4. Average over all K*M samples for each item in the batch
        # Shape: (B, num_classes)
        averaged_predictions = torch.mean(flat_samples, dim=1)
        
        # 5. Map to probability simplex (as in LaDiNE, eq 26)
        # Pr(y=a|x) = exp(-ι^-1 * (y_a - 1)^2) / sum_i(exp(-ι^-1 * (y_i - 1)^2))
        # Using a fixed sharpness parameter ι for simplicity (can be made learnable or dataset-specific)
        # A reasonable value based on LaDiNE paper's findings
        iota = 0.1737 # Example value, can be a parameter
        logits_for_simplex = - (1.0 / iota) * (averaged_predictions - 1.0) ** 2
        final_predictions = F.softmax(logits_for_simplex, dim=-1) # (B, num_classes)
        
        return final_predictions, all_samples # Return final preds and raw samples for uncertainty


# --- Example Usage ---
if __name__ == "__main__":
    # Example configuration
    batch_size = 4
    num_classes = 2 # Binary classification
    num_ensemble_members = 3 # K (smaller for demo)
    z_common_dim = 512 # d_z
    reliability_scores_dim = 3 # n+2 (e.g., image, text, tabular)
    num_timesteps = 100 # Smaller for faster demo
    
    # testing data
    testing_y_0 = F.one_hot(torch.randint(0, num_classes, (batch_size,)), num_classes).float() # (B, C)
    testing_z_common_list = [torch.randn(batch_size, z_common_dim) for _ in range(num_ensemble_members)]
    testing_reliability_scores = torch.rand(batch_size, reliability_scores_dim) # (B, n+2)
    testing_raw_modalities = {
        'image_features': torch.randn(batch_size, 768),
        'text_features': torch.randn(batch_size, 512)
        # Note: These dims should match modality_dims passed to CDM if used in context emb
    } 
    testing_dynamic_weights = F.softmax(torch.randn(batch_size, num_ensemble_members), dim=1) # (B, K)

    # Instantiate the ensemble
    ensemble = RobustMultiModalDiffusionEnsemble(
        num_ensemble_members=num_ensemble_members,
        num_classes=num_classes,
        z_common_dim=z_common_dim,
        reliability_scores_dim=reliability_scores_dim,
        modality_dims={'image_features': 768, 'text_features': 512}, # Match testing_raw_modalities keys/dims
        num_timesteps=num_timesteps,
        beta_schedule="linear", # or "cosine"
        beta_start=1e-4,
        beta_end=0.02
    )

    # --- Training Loss Computation ---
    try:
        loss = ensemble.compute_loss(
            y_0=testing_y_0,
            z_common_list=testing_z_common_list,
            reliability_scores=testing_reliability_scores,
            raw_modalities=testing_raw_modalities,
            dynamic_weights=testing_dynamic_weights
        )
        logger.info(f"Computed training loss: {loss.item():.4f}")
        print(f"Computed training loss: {loss.item():.4f}")
    except Exception as e:
        logger.error(f"Error computing training loss: {e}")
        print(f"Error computing training loss: {e}")

    # --- Sampling for Prediction ---
    # --- Replace with your dataset path... and details-----------
    try:
        final_preds, raw_samples = ensemble.sample(
            z_common_list=testing_z_common_list,
            reliability_scores=testing_reliability_scores,
            raw_modalities=testing_raw_modalities,
            dynamic_weights=testing_dynamic_weights,
            num_samples_per_member=5 # M=5 for faster demo
        )
        logger.info(f"Sampled final predictions shape: {final_preds.shape}") # (B, num_classes)
        logger.info(f"Sampled raw samples list length: {len(raw_samples)}") # K
        logger.info(f"Sampled raw samples shape (first member): {raw_samples[0].shape}") # (B, M, num_classes)
        logger.info(f"Sample predictions (first item): {final_preds[0]}")
        logger.info(f"Sample predictions sum to 1: {final_preds[0].sum().item():.4f}")
        print(f"Sampled final predictions shape: {final_preds.shape}") # (B, num_classes)
        print(f"Sampled raw samples list length: {len(raw_samples)}") # K
        print(f"Sampled raw samples shape (first member): {raw_samples[0].shape}") # (B, M, num_classes)
        print(f"Sample predictions (first item): {final_preds[0]}")
        print(f"Sample predictions sum to 1: {final_preds[0].sum().item():.4f}")
        
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        print(f"Error during sampling: {e}")
