
def load_checkpoints_sorted(ckpt_dir: str):
    pattern = os.path.join(ckpt_dir, "small_transformer_epoch_*.pt")
    files = glob.glob(pattern)
    def epoch_num(fp):
        m = re.search(r"epoch_(\d+)_", os.path.basename(fp))
        return int(m.group(1)) if m else -1
    return sorted(files, key=epoch_num)

def filter_checkpoints_by_config(ckpt_paths: list, exp_cfg: ExperimentConfig, seed: int) -> list:
    """Keep only checkpoints whose saved config matches the current experiment config and seed."""
    matched = []
    for p in ckpt_paths:
        try:
            ck = torch.load(p, map_location=DEVICE)
            cfg = ck.get("config", {})
            mc = cfg.get("model_config", {})
            if (
                cfg.get("alpha") == exp_cfg.alpha and
                cfg.get("beta") == exp_cfg.beta and
                cfg.get("seed") == seed and
                cfg.get("seq_length") == exp_cfg.seq_length and
                mc.get("d_model") == exp_cfg.model_config.d_model and
                mc.get("d_head") == exp_cfg.model_config.d_head and
                mc.get("n_layers") == exp_cfg.model_config.n_layers and
                mc.get("use_bos_token") == exp_cfg.model_config.use_bos_token and
                mc.get("attn_only") == exp_cfg.model_config.attn_only and
                mc.get("d_mlp") == exp_cfg.model_config.d_mlp
            ):
                matched.append(p)
        except Exception:
            # Skip unreadable/legacy checkpoints
            continue
    # Keep chronological order
    return matched

def build_epoch_dataframe(checkpoints: list, model_cfg: ModelConfig) -> pd.DataFrame:
    records = []
    # Probe inputs
    seq_len = exp_config.seq_length
    batch = build_probe_batch(seq_len -1 , use_bos=model_cfg.use_bos_token).to(DEVICE)
    B, T = batch.shape
    # Pre-compute N and H
    pos_idx = torch.arange(T, device=batch.device).unsqueeze(0).expand(B, -1)
    # Exclude BOS for H count
    tokens_for_count = batch.clone()
    if model_cfg.use_bos_token:
        tokens_for_count[:, 0] = 0
    H_inclusive = torch.cumsum(tokens_for_count, dim=1)
    # Per-sequence source labels: 'zeros' (0), 'lower' (1), 'upper' (2)
    zeros_count = 1
    lower_count = seq_len - 1
    upper_count = seq_len
    source_seq = torch.empty((B,), dtype=torch.long, device=batch.device)
    source_seq[:zeros_count] = 0
    source_seq[zeros_count:zeros_count+lower_count] = 1
    source_seq[zeros_count+lower_count:] = 2
    source_mat = source_seq.unsqueeze(1).expand(-1, T)
    # Last token per sequence (exclude BOS): simply the last column of batch
    last_token_seq = batch[:, -1].clone()
    # Broadcast last token to all positions for convenient flattening
    last_token_mat = last_token_seq.unsqueeze(1).expand(-1, T)
    # Token observed at each (B, T) position (includes BOS at col 0 if present)
    token_at_pos = batch.clone()

    # We only keep positions t >= 1 when BOS is used
    valid_mask = torch.ones_like(pos_idx, dtype=torch.bool)
    if model_cfg.use_bos_token:
        valid_mask[:, 0] = False

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        epoch = ckpt.get("epoch", None)

        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"]) 
        model.eval()

        with torch.no_grad():
            _, cache = model.run_with_cache(batch)

        # Pre-MLP (after attention, before MLP) for layer 0
        pre_mlp = cache["resid_mid", 0]  # [B, T, d_model]
        # Post-MLP for layer 0
        post_mlp = cache["resid_post", 0]  # [B, T, d_model]

        # Flatten and collect
        for name, tensor in [("pre_mlp", pre_mlp), ("post_mlp", post_mlp)]:
            V = tensor[valid_mask].detach().cpu().numpy().reshape(-1, model.cfg.d_model)
            Ns = pos_idx[valid_mask].detach().cpu().numpy()
            Hs = H_inclusive[valid_mask].detach().cpu().numpy()
            Ss = source_mat[valid_mask].detach().cpu().numpy()
            Ls = last_token_mat[valid_mask].detach().cpu().numpy()
            Ts = token_at_pos[valid_mask].detach().cpu().numpy()
            for (x, y), n, h, s, l, tcur in zip(V, Ns, Hs, Ss, Ls, Ts):
                records.append({
                    "epoch": epoch,
                    "name": name,
                    "N": int(n),
                    "H": int(h),
                    "source": {0: "zeros", 1: "lower", 2: "upper"}[int(s)],
                    "final_token": int(l),
                    "token_at_N": int(tcur),
                    "x": float(x),
                    "y": float(y),
                })

    return pd.DataFrame.from_records(records)



def build_embed_unembed_trajectories(checkpoints: list, model_cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect token embedding and unembedding vectors across epochs.

    Returns two dataframes with columns: epoch, token_id, token_name, x, y
    """
    emb_records = []
    unemb_records = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        epoch = ckpt.get("epoch", None)

        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"]) 
        model.eval()

        with torch.no_grad():
            W_E = model.embed.W_E.detach().cpu().numpy()  # [d_vocab, d_model]
            W_U = model.unembed.W_U.detach().cpu().numpy()  # [d_model, d_vocab]

        d_vocab = model.cfg.d_vocab
        # Plot tokens 0,1 and BOS (id=2) if present
        tokens_of_interest = [0, 1] + ([2] if d_vocab > 2 else [])
        for tok in tokens_of_interest:
            if tok < d_vocab and model.cfg.d_model >= 2:
                x_e, y_e = float(W_E[tok, 0]), float(W_E[tok, 1])
                emb_records.append({
                    "epoch": epoch,
                    "token_id": tok,
                    "token_name": {0: "0", 1: "1", 2: "BOS"}.get(tok, str(tok)),
                    "x": x_e,
                    "y": y_e,
                })
                x_u, y_u = float(W_U[0, tok]), float(W_U[1, tok])
                unemb_records.append({
                    "epoch": epoch,
                    "token_id": tok,
                    "token_name": {0: "0", 1: "1", 2: "BOS"}.get(tok, str(tok)),
                    "x": x_u,
                    "y": y_u,
                })

    return pd.DataFrame.from_records(emb_records), pd.DataFrame.from_records(unemb_records)


def build_positional_embed_trajectories(checkpoints: list, model_cfg: ModelConfig, max_positions: int) -> pd.DataFrame:
    """Collect positional embedding vectors across epochs for positions [0, max_positions)."""
    records = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        epoch = ckpt.get("epoch", None)

        model = create_coinformer_model(model_cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"]) 
        model.eval()

        with torch.no_grad():
            W_pos = model.pos_embed.W_pos.detach().cpu().numpy()  # [n_ctx, d_model]

        n_ctx = W_pos.shape[0]
        d_model = W_pos.shape[1]
        limit = min(max_positions, n_ctx)
        if d_model < 2:
            continue
        for pos in range(limit):
            x, y = float(W_pos[pos, 0]), float(W_pos[pos, 1])
            records.append({
                "epoch": epoch,
                "position": pos,
                "x": x,
                "y": y,
            })
    return pd.DataFrame.from_records(records)
