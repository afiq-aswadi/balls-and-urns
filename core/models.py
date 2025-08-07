"""
Model architectures and configurations for coinformer experiments.
"""
import torch
import transformer_lens
from dataclasses import dataclass
from typing import Optional


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
    use_pos_embed: bool = True
    
    # Normalization
    normalization_type: Optional[str] = None
    
    @property
    def d_vocab(self) -> int:
        """Vocabulary size based on BOS token usage."""
        return 3 if self.use_bos_token else 2
    
    def to_transformer_lens_config(self) -> transformer_lens.HookedTransformerConfig:
        """Convert to transformer_lens configuration."""
        return transformer_lens.HookedTransformerConfig(
            d_model=self.d_model,
            d_head=self.d_head,
            n_heads=self.n_heads,  # Will default to d_model // d_head if None
            d_mlp=self.d_mlp,  # Will default to 4 * d_model if None
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
NO_POS_EMBED_CONFIG = ModelConfig(use_pos_embed=False)


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
    
    # Deactivate positional embedding if specified
    if not config.use_pos_embed:
        deactivate_positional_embedding(model)
    
    return model


def deactivate_positional_embedding(model: transformer_lens.HookedTransformer) -> transformer_lens.HookedTransformer:
    """
    Deactivate the positional embedding in the model.
    This is done by setting the positional embedding weights to zero.
    """
    model.pos_embed.W_pos.data.fill_(0.0)
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