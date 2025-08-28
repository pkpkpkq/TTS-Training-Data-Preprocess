# AudioClassification_structbert.py
"""
兼容 StructBERT (ModelScope) 的批量情感分类脚本。
输入：ROOT_DIR 下的音频文件（支持 wav/flac/mp3/m4a/ogg/opus），但实际使用对应 .lab 文件的文本进行分类
输出：CSV，字段 ["audio_path", "pred_label", "pred_score", "topk_labels", "topk_scores"]
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

# ============== 配置区域（直接修改这些变量） ========================
ROOT_DIR = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\芙宁娜"
OUT_CSV = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\AudioClassification\emotions.csv"
MODEL_ID = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\local_model\StructBERT" 
MODEL_REVISION = "v1.0.0"  # 模型版本
BATCH_SIZE = 8          # 仅用于进度分块；ModelScope 支持批量推理
TOPK = 1                # 输出前 K 个情感类别
DEVICE = None           # None 自动检测 -> "cuda:0" / "cpu"
VERBOSE = 1             # 0=WARNING, 1=INFO, 2=DEBUG
LIMIT = None            # 调试用，只处理前 N 个文件
EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"]
# ===================================================================

# StructBERT 默认 8-class mapping（基于 README.md 中的情绪类别，包括无明显情绪）
EMO_LABELS = [
    "fearful", "angry", "disgusted", "like", "sad",
    "happy", "surprised", "neutral"
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


def read_lab_text(audio_path: Path) -> str:
    """
    对于给定的音频路径，查找对应 .lab 文件并读取文本。
    如果不存在或读取失败，返回空字符串。
    """
    lab_path = audio_path.with_suffix(".lab")
    if not lab_path.is_file():
        return ""
    try:
        with lab_path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.warning(f"读取 {lab_path} 失败: {e}")
        return ""


def parse_modelscope_result(res: Any, topk: int = 1) -> Tuple[str, float, List[str], List[float]]:
    """
    尽量鲁棒地解析 ModelScope(StructBERT) 的输出。
    支持常见形式：
      - dict with 'labels' (list of str) and 'scores' (list of float)
    返回：pred_label, pred_score, topk_labels, topk_scores
    """
    # Helper to convert to numpy safely
    def _to_np(x):
        try:
            arr = np.array(x, dtype=float)
            return arr
        except Exception:
            return None

    if not isinstance(res, dict):
        return "unknown", 0.0, [], []

    if "labels" in res and "scores" in res:
        labels = res.get("labels", [])
        scores = _to_np(res.get("scores"))
        if scores is not None and len(scores) == len(labels):
            # 假设 scores 已按降序排序，或我们排序
            if scores.ndim == 1:
                idx = np.argsort(-scores)
                sorted_labels = [labels[int(i)] for i in idx]
                sorted_scores = [float(scores[int(i)]) for i in idx]
                topk_labels = sorted_labels[:topk]
                topk_scores = sorted_scores[:topk]
                pred_label = topk_labels[0] if topk_labels else "unknown"
                pred_score = topk_scores[0] if topk_scores else 0.0
                return pred_label, pred_score, topk_labels, topk_scores

    # If nothing parsed, return unknown
    return "unknown", 0.0, [], []


def run_inference():
    # 1) 设备
    device_str = DEVICE
    if device_str is None:
        try:
            import torch

            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_str = "cpu"
    logging.info(f"使用设备: {device_str}")

    # 2) import modelscope (延后 import 以便给出友好报错)
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except Exception as e:
        logging.error("未能导入 modelscope。请先安装：pip install -U modelscope")
        raise

    # 3) 加载模型
    logging.info(f"加载 StructBERT 模型: {MODEL_ID} (revision={MODEL_REVISION}) ...")
    model = pipeline(Tasks.text_classification, MODEL_ID, model_revision=MODEL_REVISION, device=device_str)
    logging.info(f"模型加载完成，model_id={MODEL_ID}")

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
            batch_texts = []
            batch_audio_paths = []
            for p in batch:
                text = read_lab_text(p)
                if text:
                    batch_texts.append(text)
                    batch_audio_paths.append(str(p))
                else:
                    # 如果没有文本，标记为 unknown
                    row = {
                        "audio_path": str(p),
                        "pred_label": "unknown",
                        "pred_score": 0.0,
                        "topk_labels": json.dumps([], ensure_ascii=False),
                        "topk_scores": json.dumps([]),
                    }
                    writer.writerow(row)
                    pbar.update(1)

            if batch_texts:
                try:
                    # ModelScope 支持批量输入（list of str）
                    res_list = model(batch_texts)
                    if not isinstance(res_list, list):
                        res_list = [res_list]  # 单输入时可能不是列表

                    for res, audio_path in zip(res_list, batch_audio_paths):
                        pred_label, pred_score, topk_labels, topk_scores = parse_modelscope_result(res, topk=TOPK)
                        row = {
                            "audio_path": audio_path,
                            "pred_label": pred_label,
                            "pred_score": float(pred_score),
                            "topk_labels": json.dumps(topk_labels, ensure_ascii=False),
                            "topk_scores": json.dumps(topk_scores),
                        }
                        writer.writerow(row)
                        pbar.update(1)
                except Exception as e:
                    logging.exception(f"批量处理失败: {e}")
                    # 回退到 unknown
                    for audio_path in batch_audio_paths:
                        row = {
                            "audio_path": audio_path,
                            "pred_label": "unknown",
                            "pred_score": 0.0,
                            "topk_labels": json.dumps([], ensure_ascii=False),
                            "topk_scores": json.dumps([]),
                        }
                        writer.writerow(row)
                        pbar.update(1)
        pbar.close()

    logging.info(f"结果已保存到: {out_csv}")


if __name__ == "__main__":
    setup_logging(VERBOSE)
    run_inference()