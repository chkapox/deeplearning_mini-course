# inference.py
import csv
from pathlib import Path

import torch
import torchaudio
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(cfg: DictConfig):
    # ── device ───────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda")
        if (cfg.trainer.device in ("auto", "cuda") and torch.cuda.is_available())
        else torch.device("cpu")
    )

    # ── dataset: форсим eval и без перемешивания ─────────────────────────────
    ds_cfg = cfg.datasets.test
    ds_cfg.split = cfg.datasets.test.split
    ds_cfg.shuffle_index = False
    dataset = instantiate(ds_cfg)

    # DataLoader: без shuffle / drop_last
    collate = instantiate(cfg.dataloader.collate_fn)
    loader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        shuffle=False,
        collate_fn=collate,
    )

    # ── модель и чекпоинт ────────────────────────────────────────────────────
    model = instantiate(cfg.model).to(device)
    ckpt_dir = ROOT_PATH / cfg.trainer.save_dir / cfg.writer.run_name
    ckpt_name = cfg.trainer.get("resume_from") or "model_best.pth"
    ckpt_path = ckpt_dir / ckpt_name
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    print(">>> checkpoint loaded from:", ckpt_path)
    print(">>> param abs-mean:", next(model.parameters()).abs().mean().item())
    model.eval()

    # ── ключи в порядке датасета ────────────────────────────────────────────
    keys_in_order = [Path(e["path"]).stem for e in dataset._index]

    # ── batch-трансформ для инференса (как на трене) ────────────────────────
    # можно отключить через +apply_tform=false
    apply_tform = bool(getattr(cfg, "apply_tform", True))
    tform = None
    if apply_tform:
        try:
            tform_cfg = cfg.transforms.batch_transforms.inference.data_object
            tform = instantiate(tform_cfg)
            if isinstance(tform, torch.nn.Module):
                tform = tform.to(device)
        except Exception:
            tform = None

    # ── тип скора ────────────────────────────────────────────────────────────
    # +score_type=p_bona | p_spoof | logit_diff | neg_logit_diff
    score_type = str(getattr(cfg, "score_type", "p_bona")).lower()
    assert score_type in {"p_bona", "p_spoof", "logit_diff", "neg_logit_diff"}

    # ── куда писать CSV ──────────────────────────────────────────────────────
    out_csv = Path(getattr(cfg, "out_csv", "preds.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── прогон и запись CSV: (utt_id, score) без заголовка ───────────────────
    total = 0
    idx = 0
    with torch.no_grad(), out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        for batch in loader:
            x = batch["data_object"].to(device)
            if tform is not None:
                x = tform(x)

            logits = model(data_object=x)["logits"]  # [B, 2]
            probs = torch.softmax(logits, dim=-1)

            # обучение: class 1 = bonafide, class 0 = spoof
            p_bona = probs[:, 1]
            p_spoof = probs[:, 0]
            ld = logits[:, 1] - logits[:, 0]  # >0 => bonafide

            if score_type == "p_bona":
                score = p_bona
            elif score_type == "p_spoof":
                score = p_spoof
            elif score_type == "logit_diff":
                score = ld
            else:  # neg_logit_diff
                score = -ld

            score = score.detach().cpu()
            bsz = score.shape[0]
            for j in range(bsz):
                writer.writerow([keys_in_order[idx + j], float(score[j])])
            idx += bsz
            total += bsz

    print(f"Saved CSV to: {out_csv} ({total} rows)")


if __name__ == "__main__":
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass
    main()
