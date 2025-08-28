import os
import json
import csv
import logging
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
from tqdm import tqdm

# ================================================
# 配置区域（直接修改这些变量）
ROOT_DIR = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\芙宁娜" # 要处理的音频文件所在的根目录
OUT_CSV = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\AudioClassification\emotions.csv"  # 输出结果 CSV
MODEL_ID = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\local_model\speech-emotion-recognition-with-openai-whisper-large-v3"  # HuggingFace 模型
BATCH_SIZE = 8          # 批量大小
MAX_DURATION = 30.0      # 截断/填充到的秒数，None 表示不裁剪
TOPK = 1                # 输出前 K 个情感类别
DEVICE = None           # "cuda", "cuda:0", "cpu"，None 自动检测
VERBOSE = 1             # 0=只警告, 1=info, 2=debug
LIMIT = None            # 调试用，只处理前 N 个文件
EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"]
# ================================================

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def find_audio_files(root: Path, exts):
    files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in {e.lower() for e in exts}:
            files.append(p)
    return files


def load_audio_mono(path: Path, sr: int, max_duration: float | None):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if max_duration is not None:
        max_len = int(sr * max_duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")
    return y.astype(np.float32, copy=False)


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits.float(), dim=-1)


def run_inference():
    # 设备
    device_str = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"使用设备: {device}")

    # 随机种子
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    np.random.seed(42)

    # 模型与特征提取器
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, do_normalize=True)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval().to(device)

    id2label = model.config.id2label
    sr = getattr(feature_extractor, "sampling_rate", 16000)

    # 查找文件
    root = Path(ROOT_DIR)
    files = find_audio_files(root, EXTS)
    if not files:
        raise FileNotFoundError(f"未找到音频文件: {ROOT_DIR}")
    if LIMIT:
        files = files[:LIMIT]

    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "pred_label", "pred_score", "topk_labels", "topk_scores"])
        writer.writeheader()

        pbar = tqdm(total=len(files), desc="情感分类", unit="file")
        for i in range(0, len(files), BATCH_SIZE):
            batch = files[i : i + BATCH_SIZE]
            waves = []
            valid_paths = []
            for p in batch:
                try:
                    y = load_audio_mono(p, sr=sr, max_duration=MAX_DURATION)
                    waves.append(y)
                    valid_paths.append(p)
                except Exception as e:
                    logging.error(f"读取失败 {p}: {e}")

            if not waves:
                pbar.update(len(batch))
                continue

            inputs = feature_extractor(waves, sampling_rate=sr, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = softmax_probs(outputs.logits)
            conf, pred_idx = torch.max(probs, dim=-1)

            topk_conf, topk_idx = torch.topk(probs, k=min(TOPK, probs.shape[-1]), dim=-1)

            for j, path in enumerate(valid_paths):
                row = {
                    "audio_path": str(path),
                    "pred_label": id2label.get(int(pred_idx[j].item()), str(int(pred_idx[j].item()))),
                    "pred_score": float(conf[j].item()),
                    "topk_labels": json.dumps([id2label.get(int(k.item()), str(int(k.item()))) for k in topk_idx[j]], ensure_ascii=False),
                    "topk_scores": json.dumps([float(c.item()) for c in topk_conf[j]]),
                }
                writer.writerow(row)

            pbar.update(len(batch))
        pbar.close()

    logging.info(f"结果已保存到: {out_csv}")


if __name__ == "__main__":
    setup_logging(VERBOSE)
    run_inference()
