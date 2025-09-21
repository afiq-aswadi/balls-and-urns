"""
Model architectures and configurations for coinformer experiments.
"""
import torch
import transformer_lens
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import math


class PosEmbedType(Enum):
    """Types of positional embeddings."""
    LEARNED = "learned"      # default trainable embeddings
    LINEAR = "linear"        # linear function of position (i/n_ctx)
    LOG = "log"             # logarithmic function of position


@dataclass
class PosEmbedConfig:
    """Configuration for positional embeddings."""
    type: PosEmbedType = PosEmbedType.LEARNED
    trainable: bool = True  # whether to compute gradients
    scale: float = 1.0      # scaling factor for deterministic embeddings


@dataclass
class ModelConfig:
    """Configuration for coinformer model architectures."""
    # Architecture parameters
    d_model: int = 64
    d_head: int = 32
    n_heads: Optional[int] = None  # If None, defaults to d_model // d_head
    d_mlp: Optional[int] = None  # If None, defaults to 4 * d_model
    n_layers: int = 2
    n_ctx: int = 100
    act_fn: str = "relu"
    attn_only: bool = False
    
    # Vocabulary and embedding options
    use_bos_token: bool = False  # If True, d_vocab=3 (BOS,0,1); if False, d_vocab=2 (0,1)
    pos_embed_config: Optional[PosEmbedConfig] = field(default_factory=PosEmbedConfig)  # None means no pos embeds
    
    # Normalization
    normalization_type: Optional[str] = None
    
    @property
    def d_vocab(self) -> int:
        """Vocabulary size based on BOS token usage."""
        return 3 if self.use_bos_token else 2
    
    def to_transformer_lens_config(self) -> transformer_lens.HookedTransformerConfig:
        """Convert to transformer_lens configuration."""
        # Provide explicit fallbacks expected by TransformerLens
        computed_n_heads = self.n_heads if self.n_heads is not None else (self.d_model // self.d_head)
        computed_d_mlp = self.d_mlp if self.d_mlp is not None else (4 * self.d_model)
        return transformer_lens.HookedTransformerConfig(
            d_model=self.d_model,
            d_head=self.d_head,
            n_heads=computed_n_heads,
            d_mlp=computed_d_mlp,
            n_layers=self.n_layers,
            n_ctx=self.n_ctx,
            d_vocab=self.d_vocab,
            act_fn=self.act_fn,
            default_prepend_bos=self.use_bos_token,
            attn_only=self.attn_only,
            normalization_type=self.normalization_type,
        )


# Predefined model configurations
DEFAULT_CONFIG = ModelConfig()

ATTENTION_ONLY_CONFIG = ModelConfig(
    d_model=64,
    d_head=64,
    n_layers=1,
    attn_only=True
)

# Bottleneck configurations for dimension analysis
SMALL_D_MODEL_CONFIG = ModelConfig(d_model=32, d_head=16)
LARGE_D_MODEL_CONFIG = ModelConfig(d_model=128, d_head=64)
SMALL_D_HEAD_CONFIG = ModelConfig(d_model=64, d_head=16)

# BOS token configurations
BOS_TOKEN_CONFIG = ModelConfig(use_bos_token=True)
NO_POS_EMBED_CONFIG = ModelConfig(pos_embed_config=None)


def create_coinformer_model(config: ModelConfig) -> transformer_lens.HookedTransformer:
    """
    Create a coinformer model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized HookedTransformer model
    """
    tl_config = config.to_transformer_lens_config()
    model = transformer_lens.HookedTransformer(tl_config)
    
    # Handle positional embeddings based on configuration
    if config.pos_embed_config is None:
        # No positional embeddings
        deactivate_positional_embedding(model)
    elif config.pos_embed_config.type == PosEmbedType.LINEAR:
        set_linear_positional_embedding(model, config.pos_embed_config)
    elif config.pos_embed_config.type == PosEmbedType.LOG:
        set_log_positional_embedding(model, config.pos_embed_config)
    elif config.pos_embed_config.type == PosEmbedType.LEARNED:
        # Default learned embeddings, just set trainability
        model.pos_embed.W_pos.requires_grad = config.pos_embed_config.trainable
    
    return model


def deactivate_positional_embedding(model: transformer_lens.HookedTransformer) -> transformer_lens.HookedTransformer:
    """
    Deactivate the positional embedding in the model.
    This is done by setting the positional embedding weights to zero.
    """
    model.pos_embed.W_pos.data.fill_(0.0)
    model.pos_embed.W_pos.requires_grad = False
    return model


def set_linear_positional_embedding(model: transformer_lens.HookedTransformer, config: PosEmbedConfig) -> transformer_lens.HookedTransformer:
    """
    Set linear positional embeddings in the last dimension.
    Position i gets value (i / n_ctx) * scale in the last dimension.
    All other dimensions are set to 0.
    """
    n_ctx = model.cfg.n_ctx
    
    # Initialize all positional embeddings to zero
    model.pos_embed.W_pos.data.fill_(0.0)
    
    # Set the last dimension to linear function of position
    for pos in range(n_ctx):
        model.pos_embed.W_pos.data[pos, -1] = (pos / n_ctx) * config.scale
    
    # Set trainability
    model.pos_embed.W_pos.requires_grad = config.trainable
    
    return model


def set_log_positional_embedding(model: transformer_lens.HookedTransformer, config: PosEmbedConfig) -> transformer_lens.HookedTransformer:
    """
    Set logarithmic positional embeddings in the last dimension.
    Position i gets value log((i+1) / (n_ctx+1)) * scale in the last dimension.
    All other dimensions are set to 0.
    """
    n_ctx = model.cfg.n_ctx
    
    # Initialize all positional embeddings to zero
    model.pos_embed.W_pos.data.fill_(0.0)
    
    # Set the last dimension to log function of position
    for pos in range(n_ctx):
        # Use (pos+1) to avoid log(0), normalize to [0,1] range approximately
        normalized_pos = (pos + 1) / (n_ctx + 1)
        model.pos_embed.W_pos.data[pos, -1] = math.log(normalized_pos) * config.scale
    
    # Set trainability
    model.pos_embed.W_pos.requires_grad = config.trainable
    
    return model


def set_fixed_positional_embedding(model: transformer_lens.HookedTransformer) -> transformer_lens.HookedTransformer:
    """
    Set fixed positional embeddings that increment by 1 in the last dimension.
    The positional embeddings are non-trainable (gradients switched off).
    
    For a model with d_model dimensions, only the last dimension is used for position,
    with position 0 having value 0, position 1 having value 1, etc.
    All other dimensions are set to 0.
    """
    n_ctx = model.cfg.n_ctx
    
    # Initialize all positional embeddings to zero
    model.pos_embed.W_pos.data.fill_(0.0)
    
    # Set the last dimension to increment by position
    for pos in range(n_ctx):
        model.pos_embed.W_pos.data[pos, -1] = float(pos)
    
    # Turn off gradients for positional embeddings
    model.pos_embed.W_pos.requires_grad = False
    
    return model


def get_model_info(model: transformer_lens.HookedTransformer) -> dict:
    """Get information about a model's architecture."""
    config = model.cfg
    return {
        "d_model": config.d_model,
        "d_head": config.d_head,
        "n_layers": config.n_layers,
        "n_ctx": config.n_ctx,
        "d_vocab": config.d_vocab,
        "attn_only": config.attn_only,
        "use_bos_token": config.default_prepend_bos,
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def save_model(model: transformer_lens.HookedTransformer, filepath: str) -> None:
    """Save model state dict to file."""
    torch.save(model.state_dict(), filepath)


def load_model(config: ModelConfig, filepath: str) -> transformer_lens.HookedTransformer:
    """Load model from saved state dict."""
    model = create_coinformer_model(config)
    model.load_state_dict(torch.load(filepath))
    return model