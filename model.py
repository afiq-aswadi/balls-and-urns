import transformer_lens

coinformer_model_attn_only_config = transformer_lens.HookedTransformerConfig(
    d_model=64, #embedding dimension
    d_head=64,
    n_layers=1,
    n_ctx=100,
    d_vocab=2,
    default_prepend_bos=False,
    attn_only=True
)



coinformer_model_config = transformer_lens.HookedTransformerConfig(
    d_model=64, #embedding dimension
    d_head=32,
    n_layers=2,
    n_ctx=100,
    d_vocab=2,
    act_fn="relu",
    default_prepend_bos=False,
)

def deactivate_position(model):
    """
    Deactivate the positional embedding in the model.
    This is done by setting the positional embedding weights to zero.
    """
    model.pos_embed.W_pos.data.fill_(0.0)
    model.pos_embed.W_pos.requires_grad = False
    return model
