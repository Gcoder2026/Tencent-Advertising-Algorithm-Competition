"""End-to-end smoke test: train tiny model on synthetic data, then infer.

Skipped when torch is missing (Anaconda base env). Runs in CI / on platform.
"""
import json
import os
import sys
import pytest


def test_train_and_infer_smoke(synth_data_root, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    # Make pcvr/ importable so `import infer` works.
    pcvr_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pcvr_root not in sys.path:
        sys.path.insert(0, pcvr_root)

    from configs.baseline import Config
    from src.data import get_pcvr_data, NUM_TIME_BUCKETS
    from src.model import PCVRHyFormer
    from src.trainer import Trainer
    from src.audit import audit_single_model

    cfg = Config(
        d_model=8, emb_dim=8, num_queries=1, num_hyformer_blocks=1, num_heads=2,
        batch_size=8, num_workers=0, buffer_batches=0,
        seq_max_lens={"seq_a": 16, "seq_b": 16},
        valid_ratio=0.2, num_epochs=1, patience=2,
        ns_tokenizer_type="rankmixer",
        user_ns_tokens=2, item_ns_tokens=2,
        use_bf16=False,            # CPU friendly
        warmup_steps=2, total_steps_hint=10, min_lr_factor=0.1,
        eval_every_n_steps=0,
    )
    train_loader, valid_loader, ds = get_pcvr_data(
        data_dir=synth_data_root["data_dir"],
        schema_path=synth_data_root["schema_path"],
        batch_size=cfg.batch_size, valid_ratio=cfg.valid_ratio,
        train_ratio=1.0, num_workers=0, buffer_batches=0,
        seed=cfg.seed, seq_max_lens=cfg.seq_max_lens,
    )
    user_specs = [(max(ds.user_int_vocab_sizes[o:o+l]), o, l)
                  for _, o, l in ds.user_int_schema.entries]
    item_specs = [(max(ds.item_int_vocab_sizes[o:o+l]), o, l)
                  for _, o, l in ds.item_int_schema.entries]
    user_ns = [[i] for i in range(len(ds.user_int_schema.entries))]
    item_ns = [[i] for i in range(len(ds.item_int_schema.entries))]
    model = PCVRHyFormer(
        user_int_feature_specs=user_specs, item_int_feature_specs=item_specs,
        user_dense_dim=ds.user_dense_schema.total_dim,
        item_dense_dim=ds.item_dense_schema.total_dim,
        seq_vocab_sizes=ds.seq_domain_vocab_sizes,
        user_ns_groups=user_ns, item_ns_groups=item_ns,
        d_model=cfg.d_model, emb_dim=cfg.emb_dim,
        num_queries=cfg.num_queries, num_hyformer_blocks=cfg.num_hyformer_blocks,
        num_heads=cfg.num_heads, seq_encoder_type=cfg.seq_encoder_type,
        hidden_mult=cfg.hidden_mult, dropout_rate=cfg.dropout_rate,
        seq_top_k=cfg.seq_top_k, seq_causal=cfg.seq_causal,
        action_num=cfg.action_num,
        num_time_buckets=NUM_TIME_BUCKETS if cfg.use_time_buckets else 0,
        rank_mixer_mode=cfg.rank_mixer_mode,
        use_rope=cfg.use_rope, rope_base=cfg.rope_base,
        emb_skip_threshold=cfg.emb_skip_threshold,
        seq_id_threshold=cfg.seq_id_threshold,
        ns_tokenizer_type=cfg.ns_tokenizer_type,
        user_ns_tokens=cfg.user_ns_tokens, item_ns_tokens=cfg.item_ns_tokens,
    )
    cfg.device = "cpu"
    cfg.ckpt_dir = str(tmp_path / "ckpts")
    cfg.schema_path = synth_data_root["schema_path"]
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    sample_uids = ["u0", "u1"]
    csv_path = str(tmp_path / "experiments.csv")
    trainer = Trainer(
        cfg=cfg, model=model,
        train_loader=train_loader, valid_loader=valid_loader,
        ckpt_out_dir=cfg.ckpt_dir, schema_path=cfg.schema_path,
        ns_groups_path=None, user_id_sample=sample_uids,
        experiments_csv=csv_path,
    )
    best = trainer.train()
    assert best is not None
    assert os.path.isdir(best)
    assert os.path.isfile(os.path.join(best, "model.pt"))
    assert os.path.isfile(csv_path)

    # Audit passes.
    rep = audit_single_model(best)
    assert rep["n_tensors"] > 0

    # Inference path (in-process to avoid env-var dance).
    monkeypatch.setenv("MODEL_OUTPUT_PATH", best)
    monkeypatch.setenv("EVAL_DATA_PATH", synth_data_root["data_dir"])
    monkeypatch.setenv("EVAL_RESULT_PATH", str(tmp_path / "result"))
    import infer  # noqa
    infer.main()
    out = os.path.join(str(tmp_path / "result"), "predictions.json")
    assert os.path.isfile(out)
    with open(out) as f:
        d = json.load(f)
    assert "predictions" in d
    assert len(d["predictions"]) > 0
