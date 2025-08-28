# AudioClassification_emotion2vec.py
"""
兼容 emotion2vec+ (FunASR) 的批量情感分类脚本。
输入：ROOT_DIR 下的音频文件（支持 wav/flac/mp3/m4a/ogg/opus）
输出：CSV，字段 ["audio_path", "pred_label", "pred_score", "topk_labels", "topk_scores"]
"""

import os
import json
import csv
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ============== 配置区域（直接修改这些变量） ========================
ROOT_DIR = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\芙宁娜"
OUT_CSV = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\AudioClassification\emotions.csv"
MODEL_ID = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\local_model\emotion2vec_plus_large"   # 支持 HF / ModelScope 名称 或 本地路径
HUB = "hf"   # "hf" or "ms" （在中国大陆若无法访问 HF 可改成 "ms"）
BATCH_SIZE = 8          # 仅用于进度分块；FunASR 单文件推理（内部可能支持 scp 批量）
MAX_DURATION = 30.0     # 秒；如果为 None 则不裁剪
TOPK = 1                # 输出前 K 个情感类别
DEVICE = None           # None 自动检测 -> "cuda:0" / "cpu"
VERBOSE = 1             # 0=WARNING, 1=INFO, 2=DEBUG
LIMIT = None            # 调试用，只处理前 N 个文件
EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"]
# ===================================================================

# emotion2vec+ 默认 9-class mapping（如果你使用别的模型/label set，按需修改）
EMO_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown"
]


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


def find_audio_files(root: Path, exts: List[str]) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {e.lower() for e in exts}:
            files.append(p)
    files.sort()
    return files


def trim_or_pad_and_save_temp(y: np.ndarray, sr: int, max_duration: float) -> str:
    """
    裁剪/补零并写到临时 wav 文件，返回临时文件路径（wav）
    """
    max_len = int(sr * max_duration)
    if len(y) > max_len:
        y2 = y[:max_len]
    else:
        y2 = np.pad(y, (0, max_len - len(y)), mode="constant")
    tmp_path = os.path.join(tempfile.gettempdir(), f"tmp_trim_{uuid.uuid4().hex}.wav")
    sf.write(tmp_path, y2, sr, subtype="PCM_16")
    return tmp_path


def load_audio_for_trimming(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    返回 (audio_array, sr) 用于手动裁剪/补零
    """
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32, copy=False), sr


def parse_funasr_result(res: Any, topk: int = 1) -> Tuple[str, float, List[str], List[float]]:
    """
    尽量鲁棒地解析 FunASR(emotion2vec+) 的输出。
    支持以下常见形式：
      - dict with 'scores' (1D or 2D) and optional 'labels' / 'feats'
      - list of dict segments [{ 'scores': [...], 'labels': [...] }, ...]
      - fallback: list of segment-level (label, score) pairs -> choose max-scored segment
    返回：pred_label, pred_score, topk_labels, topk_scores
    """
    # Helper to convert to numpy safely
    def _to_np(x):
        try:
            arr = np.array(x, dtype=float)
            return arr
        except Exception:
            return None

    # Normalize container to list of segment-dicts
    segs = []
    if isinstance(res, dict):
        segs = [res]
    elif isinstance(res, (list, tuple)):
        if len(res) == 0:
            return "unknown", 0.0, [], []
        # assume list of dicts or single-element list
        if all(isinstance(x, dict) for x in res):
            segs = list(res)
        else:
            # fallback: wrap
            segs = [{"value": res}]
    else:
        return "unknown", 0.0, [], []

    # Try to collect per-segment per-class score arrays
    per_segment_probs = []
    segment_pred_labels = []
    segment_pred_scores = []

    for seg in segs:
        # typical keys: 'scores' (class-probs 1D or 2D), 'labels', 'feats'
        if "scores" in seg:
            s = _to_np(seg["scores"])
            if s is not None:
                # if shape (n_classes,) or (n_segments, n_classes)
                if s.ndim == 1:
                    # interpreted as per-class probs for this segment
                    per_segment_probs.append(s)
                elif s.ndim == 2:
                    # if it is 2D, treat each row as class probs and average rows
                    per_segment_probs.append(np.mean(s, axis=0))
                else:
                    # ignore weird shapes
                    pass
        elif "probs" in seg:
            s = _to_np(seg["probs"])
            if s is not None:
                if s.ndim == 1:
                    per_segment_probs.append(s)
                elif s.ndim == 2:
                    per_segment_probs.append(np.mean(s, axis=0))
        # fallback to segment-level label+score (no per-class distribution)
        if "labels" in seg and "scores" in seg and not per_segment_probs:
            # sometimes labels is predicted label per segment and scores is scalar confidence
            labs = seg.get("labels")
            scrs = seg.get("scores")
            # if labels and scrs are same-length arrays of scalar predictions
            if isinstance(labs, (list, tuple)) and isinstance(scrs, (list, tuple)) and len(labs) == len(scrs):
                # push each label+score
                for L, S in zip(labs, scrs):
                    segment_pred_labels.append(str(L))
                    segment_pred_scores.append(float(S))
        # sometimes seg has single 'label' and 'score'
        if "label" in seg and "score" in seg:
            segment_pred_labels.append(str(seg.get("label")))
            try:
                segment_pred_scores.append(float(seg.get("score") or 0.0))
            except Exception:
                segment_pred_scores.append(0.0)

    # If we have per-segment-per-class probabilities -> aggregate across segments (mean)
    if per_segment_probs:
        stacked = np.stack(per_segment_probs, axis=0)  # shape (n_segments, n_classes)
        agg = np.mean(stacked, axis=0)  # per-class aggregated prob
        # ensure length matches our EMO_LABELS else create numeric labels
        n_classes = agg.shape[0]
        if n_classes == len(EMO_LABELS):
            labels_names = EMO_LABELS
        else:
            # fallback numeric labels
            labels_names = [str(i) for i in range(n_classes)]
        topk_idx = np.argsort(-agg)[:topk]
        topk_scores = [float(agg[i]) for i in topk_idx]
        topk_labels = [labels_names[int(i)] for i in topk_idx]
        pred_label = topk_labels[0]
        pred_score = topk_scores[0]
        return pred_label, float(pred_score), topk_labels, topk_scores

    # Else if we have only segment_pred_labels + segment_pred_scores
    if segment_pred_labels and segment_pred_scores:
        # choose segment with highest score as file-level prediction
        idx = int(np.argmax(np.array(segment_pred_scores, dtype=float)))
        pred_label = segment_pred_labels[idx]
        pred_score = float(segment_pred_scores[idx])
        # topk: top-k segments by score
        order = np.argsort(-np.array(segment_pred_scores, dtype=float))[:topk]
        topk_labels = [segment_pred_labels[int(i)] for i in order]
        topk_scores = [float(segment_pred_scores[int(i)]) for i in order]
        return pred_label, pred_score, topk_labels, topk_scores

    # If nothing parsed, return unknown
    return "unknown", 0.0, [], []


def run_inference():
    # 1) 设备
    device_str = DEVICE
    if device_str is None:
        # funasr prefers "cuda:0" style
        try:
            import torch

            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_str = "cpu"
    logging.info(f"使用设备: {device_str}")

    # 2) import funasr (延后 import 以便给出友好报错)
    try:
        from funasr import AutoModel
    except Exception as e:
        logging.error("未能导入 funasr。请先安装：pip install -U funasr modelscope")
        raise

    # 3) 加载模型
    logging.info(f"加载 emotion2vec 模型: {MODEL_ID} (hub={HUB}) ...")
    try:
        model = AutoModel(model=MODEL_ID, hub=HUB, device=device_str)
    except TypeError:
        # 若 AutoModel signature 不支持 hub 或 device 参数，尝试更简单的构造
        model = AutoModel(model=MODEL_ID)
    logging.info(f"模型加载完成，model_path={getattr(model, 'model_path', MODEL_ID)}")

    # 4) 查找文件
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
            for p in batch:
                tmp_created = False
                tmp_path = None
                try:
                    # 若需要裁剪/补零，先 load -> trim/save 临时 wav
                    if MAX_DURATION is not None:
                        # FunASR 要求 16k 输入，官方示例使用 16k
                        target_sr = 16000
                        y, sr = load_audio_for_trimming(p, target_sr=target_sr)
                        tmp_path = trim_or_pad_and_save_temp(y, sr=sr, max_duration=MAX_DURATION)
                        tmp_created = True
                        input_for_model = tmp_path
                    else:
                        input_for_model = str(p)

                    # 调用 FunASR 推理
                    # granularity: "utterance" (utterance-level) or "frame"
                    # extract_embedding: False -> 只返回情感分布/标签；True -> 同时返回 feats（embedding）
                    res = model.generate(input_for_model, granularity="utterance", extract_embedding=False)
                    # parse 返回（鲁棒解析）
                    pred_label, pred_score, topk_labels, topk_scores = parse_funasr_result(res, topk=TOPK)

                    row = {
                        "audio_path": str(p),
                        "pred_label": pred_label,
                        "pred_score": float(pred_score),
                        "topk_labels": json.dumps(topk_labels, ensure_ascii=False),
                        "topk_scores": json.dumps(topk_scores),
                    }
                    writer.writerow(row)
                except Exception as e:
                    logging.exception(f"处理失败 {p}: {e}")
                finally:
                    if tmp_created and tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                pbar.update(1)
        pbar.close()

    logging.info(f"结果已保存到: {out_csv}")


if __name__ == "__main__":
    setup_logging(VERBOSE)
    run_inference()
