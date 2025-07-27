import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchEmbedding(nn.Module):
    """
    Converts a multi-modal dictionary of tensors into patch embeddings.
    Each modality is embedded separately and then concatenated.
    Assumes 2D image-like inputs for image modalities.
    """
    def __init__(
        self,
        modality_dims: Dict[str, int], # e.g., {'image1': 3, 'image2': 1, 'tabular': 100}
        modality_patch_sizes: Dict[str, int], # e.g., {'image1': 16, 'image2': 16, 'tabular': 1}
        modality_embed_dims: Dict[str, int], # e.g., {'image1': 384, 'image2': 384, 'tabular': 256}
        common_embed_dim: int, # The final dimension after projection (d_e)
        img_size: int = 224 # Assumed square image size for image modalities
    ):
        """
        Initializes the Patch Embedding layer.

        Args:
            modality_dims (Dict[str, int]): Input dimensions for each modality.
            modality_patch_sizes (Dict[str, int]): Patch size for each modality.
                                          For non-image, typically 1.
            modality_embed_dims (Dict[str, int]): Intermediate embedding dim for each modality.
            common_embed_dim (int): The shared embedding dimension (d_e) for fusion.
            img_size (int, optional): Size of the input image (assumed square). Defaults to 224.
        """
        super(PatchEmbedding, self).__init__()
        self.modality_dims = modality_dims
        self.modality_patch_sizes = modality_patch_sizes
        self.modality_embed_dims = modality_embed_dims
        self.common_embed_dim = common_embed_dim
        self.img_size = img_size

        # Create separate embedding layers for each modality
        self.modality_embeddings = nn.ModuleDict()
        self.modality_projections = nn.ModuleDict()
        self.modality_cls_tokens = nn.ParameterDict()
        self.modality_pos_embeddings = nn.ParameterDict()

        total_seq_len = 1 # Start with 1 for the global CLS token
        self.modality_seq_lens = {}

        for mod_name, input_dim in modality_dims.items():
            patch_size = modality_patch_sizes[mod_name]
            embed_dim = modality_embed_dims[mod_name]

            if 'image' in mod_name.lower() or (input_dim > 10 and patch_size > 1):
                # Handle image-like modalities
                # Assume input is (B, C, H, W)
                num_patches_h = img_size // patch_size
                num_patches_w = img_size // patch_size
                num_patches = num_patches_h * num_patches_w
                patch_dim = input_dim * patch_size * patch_size
                self.modality_embeddings[mod_name] = nn.Conv2d(
                    input_dim, embed_dim, kernel_size=patch_size, stride=patch_size
                )
            else:
                # Handle tabular/1D-like modalities
                # Assume input is (B, D)
                num_patches = 1 # Treat entire vector as one "patch"
                patch_dim = input_dim
                # Use a linear layer for 1D data
                self.modality_embeddings[mod_name] = nn.Linear(patch_dim, embed_dim)

            # Projection to common embedding space
            self.modality_projections[mod_name] = nn.Linear(embed_dim, common_embed_dim)

            # Learnable CLS token for this modality
            self.modality_cls_tokens[mod_name] = nn.Parameter(torch.randn(1, 1, common_embed_dim))

            # Positional embeddings: 1 (modality cls) + num_patches
            seq_length = 1 + num_patches
            self.modality_seq_lens[mod_name] = seq_length
            self.modality_pos_embeddings[mod_name] = nn.Parameter(torch.randn(1, seq_length, common_embed_dim))

            total_seq_len += seq_length

        # Global CLS token and its positional embedding
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, common_embed_dim))
        self.global_pos_embedding = nn.Parameter(torch.randn(1, total_seq_len, common_embed_dim))

        logger.info(f"Initialized PatchEmbedding for modalities: {list(modality_dims.keys())}")
        logger.info(f"  Total sequence length (including global CLS): {total_seq_len}")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to create patch embeddings.

        Args:
            x (Dict[str, torch.Tensor]): Dictionary of input tensors for each modality.
                                         Image: (B, C, H, W), Tabular: (B, D)

        Returns:
            torch.Tensor: Concatenated patch embeddings of shape (B, N, d_e),
                          where N is the total number of patches + global CLS.
        """
        batch_size = list(x.values())[0].shape[0]
        modality_embeddings_list = [self.global_cls_token.expand(batch_size, -1, -1)]

        for mod_name, mod_tensor in x.items():
            # 1. Extract patches and create initial embeddings
            if 'image' in mod_name.lower() or (mod_tensor.dim() == 4 and mod_tensor.shape[2] > 10):
                # Image-like: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
                patches = self.modality_embeddings[mod_name](mod_tensor)
                # Flatten patches: (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches)
                patches = patches.flatten(2)
                # Transpose to (B, num_patches, embed_dim)
                patches = patches.transpose(1, 2)
            else:
                # Tabular/1D-like: (B, D) -> (B, 1, embed_dim)
                patches = self.modality_embeddings[mod_name](mod_tensor).unsqueeze(1)

            # 2. Project to common embedding space
            patches = self.modality_projections[mod_name](patches) # (B, num_patches, d_e)

            # 3. Add modality-specific CLS token
            mod_cls_token = self.modality_cls_tokens[mod_name].expand(batch_size, -1, -1) # (B, 1, d_e)
            patches_with_cls = torch.cat((mod_cls_token, patches), dim=1) # (B, 1+num_patches, d_e)

            # 4. Add positional encoding for this modality block
            pos_emb = self.modality_pos_embeddings[mod_name] # (1, 1+num_patches, d_e)
            patches_with_cls += pos_emb

            modality_embeddings_list.append(patches_with_cls)

        # 5. Concatenate embeddings from all modalities
        # Resulting shape: (B, 1 + sum(1 + num_patches_i), d_e)
        concatenated_embeddings = torch.cat(modality_embeddings_list, dim=1)

        # 6. Add global positional encoding
        concatenated_embeddings += self.global_pos_embedding[:, :concatenated_embeddings.shape[1], :]

        return concatenated_embeddings # (B, N, d_e)


class CrossModalAttention(nn.Module):
    """
    Implements cross-attention to fuse information across modalities.
    Operates on the sequence of patch embeddings from PatchEmbedding.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initializes the Cross-Modal Attention layer.

        Args:
            embed_dim (int): The embedding dimension (d_e).
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(CrossModalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention.

        Args:
            query (torch.Tensor): Query tensor of shape (B, N_q, d_e).
            key (torch.Tensor): Key tensor of shape (B, N_k, d_e).
            value (torch.Tensor): Value tensor of shape (B, N_k, d_e).
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape (B, N_q, N_k).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor (B, N_q, d_e) and attention weights (B, H, N_q, N_k).
        """
        B, N_q, _ = query.shape
        _, N_k, _ = key.shape

        # Project and reshape Q, K, V
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N_q, head_dim)
        K = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N_k, head_dim)
        V = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N_k, head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, N_q, N_k)
        
        if attn_mask is not None:
            # Apply mask (set masked positions to large negative value)
            scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # (B, H, N_q, N_k)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        out = torch.matmul(attn_weights, V) # (B, H, N_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim) # (B, N_q, d_e)
        out = self.out_proj(out) # (B, N_q, d_e)

        return out, attn_weights


class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer, potentially incorporating cross-modal attention.
    Follows the standard architecture: MHA -> Add & Norm -> FFN -> Add & Norm
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float = 0.0
    ):
        """
        Initializes the Transformer Encoder Layer.

        Args:
            embed_dim (int): The embedding dimension (d_e).
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio for hidden dimension in FFN. Defaults to 4.0.
            dropout (float, optional): Dropout rate for FFN. Defaults to 0.1.
            attn_dropout (float, optional): Dropout rate for attention. Defaults to 0.1.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        # Placeholder for potential cross-modal attention mechanism within the layer
        # This could be more complex, e.g., attending from one modality to another
        # For simplicity here, we stick to standard self-attention within the full sequence.
        # The cross-modal interaction is primarily handled by the initial PatchEmbedding
        # and the structure of the input sequence.
        
        self.norm1 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, d_e).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, d_e).
        """
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x) # (B, N, d_e)
        x = x + self.drop_path(attn_out) # Residual connection
        x = self.norm1(x) # Layer norm

        # Feed-Forward Network
        ffn_out = self.mlp(x) # (B, N, d_e)
        x = x + self.drop_path(ffn_out) # Residual connection
        x = self.norm2(x) # Layer norm

        return x


class MultiModalEncoder(nn.Module):
    """
    The main encoder stack, consisting of multiple Transformer Encoder Layers.
    """
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path_rate: float = 0.0
    ):
        """
        Initializes the Multi-Modal Encoder.

        Args:
            embed_dim (int): The embedding dimension (d_e).
            depth (int): Number of encoder layers (L).
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio for hidden dimension in FFN. Defaults to 4.0.
            dropout (float, optional): Dropout rate for FFN. Defaults to 0.1.
            attn_dropout (float, optional): Dropout rate for attention. Defaults to 0.1.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
        """
        super(MultiModalEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder layers.

        Args:
            x (torch.Tensor): Input tensor from PatchEmbedding of shape (B, N, d_e).

        Returns:
            torch.Tensor: Encoded features of shape (B, N, d_e).
        """
        for layer in self.layers:
            x = layer(x) # (B, N, d_e)
        x = self.norm(x) # (B, N, d_e)
        return x


class MultiModalVisionTransformer(nn.Module):
    """
    Cross-Modal Invariant Feature Extractor.
    
    Combines Patch Embedding, Multi-Modal Encoder, and outputs invariant features
    from early layers, potentially the global CLS token.
    """
    def __init__(
        self,
        modality_dims: Dict[str, int],
        modality_patch_sizes: Dict[str, int],
        modality_embed_dims: Dict[str, int],
        common_embed_dim: int = 768, # d_e
        img_size: int = 224,
        encoder_depth: int = 12, # L
        encoder_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        invariant_layer_indices: List[int] = [2, 5, 8] # Indices of layers to extract invariant features from
    ):
        """
        Initializes the Multi-Modal Vision Transformer.

        Args:
            modality_dims (Dict[str, int]): Input dimensions for each modality.
            modality_patch_sizes (Dict[str, int]): Patch sizes.
            modality_embed_dims (Dict[str, int]): Intermediate embed dims.
            common_embed_dim (int, optional): Shared embedding dim (d_e). Defaults to 768.
            img_size (int, optional): Image size. Defaults to 224.
            encoder_depth (int, optional): Number of encoder layers (L). Defaults to 12.
            encoder_num_heads (int, optional): Num heads. Defaults to 12.
            mlp_ratio (float, optional): FFN ratio. Defaults to 4.0.
            dropout (float, optional): Dropout. Defaults to 0.1.
            attn_dropout (float, optional): Attention dropout. Defaults to 0.1.
            drop_path_rate (float, optional): Drop path rate. Defaults to 0.1.
            invariant_layer_indices (List[int], optional): Indices for invariant features.
                                                         Defaults to [2, 5, 8].
        """
        super(MultiModalVisionTransformer, self).__init__()
        self.common_embed_dim = common_embed_dim
        self.invariant_layer_indices = invariant_layer_indices

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            modality_dims=modality_dims,
            modality_patch_sizes=modality_patch_sizes,
            modality_embed_dims=modality_embed_dims,
            common_embed_dim=common_embed_dim,
            img_size=img_size
        )

        # 2. Multi-Modal Encoder
        self.encoder = MultiModalEncoder(
            embed_dim=common_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate
        )

        logger.info(f"Initialized MultiModalVisionTransformer with depth {encoder_depth} and invariant layers at {invariant_layer_indices}")

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass to extract features.

        Args:
            x (Dict[str, torch.Tensor]): Input multi-modal data.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final encoded features (B, N, d_e).
                - List of invariant features from specified layers, each (B, N, d_e).
                  Typically, the global CLS token features are used:
                  [features_layer_i[:, 0] for i in invariant_layer_indices].
        """
        # 1. Patch Embedding
        embedded_patches = self.patch_embed(x) # (B, N, d_e)
        B, N, _ = embedded_patches.shape

        # 2. Encode with intermediate feature extraction
        encoded_features = embedded_patches
        invariant_features = []

        for i, layer in enumerate(self.encoder.layers):
            encoded_features = layer(encoded_features) # (B, N, d_e)
            
            if i in self.invariant_layer_indices:
                # Extract the global CLS token representation as invariant feature
                # This aligns with LaDiNE's use of class tokens from early layers.
                invariant_feat = encoded_features[:, 0] # (B, d_e)
                invariant_features.append(invariant_feat)

        # Apply final norm
        encoded_features = self.encoder.norm(encoded_features) # (B, N, d_e)

        return encoded_features, invariant_features # (B, N, d_e), List[(B, d_e)]


# --- Example Usage ---
if __name__ == "__main__":
    # Example configuration for a multi-modal input: 2 images, 1 tabular
    batch_size = 4
    img_size = 224
    common_embed_dim = 768
    
    # Define modalities
    modality_dims_example = {
        'image1': 3,      # RGB image
        'image2': 1,      # Grayscale image
        'tabular': 50     # 50 features
    }
    modality_patch_sizes_example = {
        'image1': 16,
        'image2': 16,
        'tabular': 1 # Treat tabular as a single "patch"
    }
    modality_embed_dims_example = {
        'image1': 384,
        'image2': 384,
        'tabular': 256
    }
    
    # Create input data
    _input = {
        'image1': torch.randn(batch_size, 3, img_size, img_size),
        'image2': torch.randn(batch_size, 1, img_size, img_size),
        'tabular': torch.randn(batch_size, 50)
    }

    # Instantiate the MM-ViT
    mm_vit = MultiModalVisionTransformer(
        modality_dims=modality_dims_example,
        modality_patch_sizes=modality_patch_sizes_example,
        modality_embed_dims=modality_embed_dims_example,
        common_embed_dim=common_embed_dim,
        img_size=img_size,
        encoder_depth=6, # Smaller for demo
        encoder_num_heads=8, # Smaller for demo
        invariant_layer_indices=[1, 3, 5] # Extract from layers 1, 3, 5
    )

    # Run forward pass
    try:
        final_features, invariant_features_list = mm_vit(_input)
        
        logger.info("--- MM-ViT Forward Pass Results ---")
        logger.info(f"Input batch size: {batch_size}")
        logger.info(f"Final encoded features shape: {final_features.shape}") # (B, N, d_e)
        
        logger.info(f"Number of invariant feature sets extracted: {len(invariant_features_list)}")
        for i, inv_feat in enumerate(invariant_features_list):
            logger.info(f"  Invariant features from layer {mm_vit.invariant_layer_indices[i]} shape: {inv_feat.shape}") # (B, d_e)
            
        # Example: Use the last extracted invariant feature for downstream tasks
        # (e.g., as input to the mapping network g_ϕ_k in DyMoLaDiNE)
        last_invariant_feature = invariant_features_list[-1] # (B, d_e)
        logger.info(f"Last extracted invariant feature (for g_ϕ_k) shape: {last_invariant_feature.shape}")

    except Exception as e:
        logger.error(f"Error during MM-ViT forward pass: {e}")
