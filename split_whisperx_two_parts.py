import os
import sys
import contextlib  # 用于管理资源，如文件句柄
import math
import shutil
import wave
import logging  # 用于日志记录
import time

# 仅依赖 whisperx 的对齐模型（不用完整 ASR），减少模型大小和推理时间
import torch
import whisperx

# 标点集合与优先级（优先级从前到后：高 -> 低）
PUNCT_PRIORITY = ["。", ".", "…", "；", ";", "？", "?", "！", "!", "，", ",", "、", "~"]
DEFAULT_MAX_DURATION = 25.0
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_SEGMENTS = 5


logger = logging.getLogger(__name__)

def get_wav_duration_seconds(path):
    """
    获取 WAV 文件的时长（秒）。
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            raise RuntimeError("采样率为0")
        duration = frames / float(rate)

    return duration

def split_wav_by_times(src_wav, split_times, outdir, base_name):
    """
    将 WAV 文件按多个指定时间点进行切分。
    返回一个包含所有输出 WAV 文件路径的列表。
    """
    with contextlib.closing(wave.open(src_wav, "rb")) as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)

    bytes_per_frame = nch * sw
    split_frames = [0] + [max(0, min(int(round(t * fr)), nframes)) for t in split_times] + [nframes]

    out_wav_paths = []
    for i in range(len(split_frames) - 1):
        start_frame = split_frames[i]
        end_frame = split_frames[i+1]

        if start_frame >= end_frame:
            continue

        start_byte = start_frame * bytes_per_frame
        end_byte = end_frame * bytes_per_frame

        part_data = pcm[start_byte:end_byte]

        out_path = os.path.join(outdir, f"{base_name}_part{i+1}.wav")
        with contextlib.closing(wave.open(out_path, "wb")) as wo:
            wo.setnchannels(nch)
            wo.setsampwidth(sw)
            wo.setframerate(fr)
            wo.writeframes(part_data)
        out_wav_paths.append(out_path)

    return out_wav_paths

def _get_potential_split_points(aligned_result, total_dur):
    """
    从对齐结果中提取所有可能的切分点及其优先级。
    """
    punct_to_priority = {p: i for i, p in enumerate(PUNCT_PRIORITY)}
    default_priority = len(PUNCT_PRIORITY)

    word_times = []
    for seg in aligned_result.get("segments", []):
        for w in seg.get("words", []) or []:
            if w.get("end") is not None and 0.2 < w["end"] < (total_dur - 0.2):
                word = w.get("word", "").strip()
                if not word:
                    continue

                priority = default_priority
                for p, prio_val in punct_to_priority.items():
                    if word.endswith(p):
                        priority = prio_val
                        break
                word_times.append({'time': w['end'], 'priority': priority, 'word': word})

    word_times.sort(key=lambda x: x['time'])
    return word_times

def _find_best_split_plan(points, total_dur, max_duration=25.0, max_cuts=4, min_duration=3.0):
    """
    使用动态规划寻找最佳切分方案。
    目标：1. 切分数最少 (<= max_cuts) 2. 总优先级最低 3. 每个分段时长 <= max_duration 且 >= min_duration
    """
    if not points:
        return None

    points.insert(0, {'time': 0, 'priority': -1, 'word': ''})

    dp = [[float('inf')] * len(points) for _ in range(max_cuts + 1)]
    path = [[-1] * len(points) for _ in range(max_cuts + 1)]
    dp[0][0] = 0

    for k in range(1, max_cuts + 1):
        for i in range(1, len(points)):
            t_i, p_i = points[i]['time'], points[i]['priority']
            for j in range(i):
                t_j = points[j]['time']
                segment_duration = t_i - t_j
                if not (min_duration <= segment_duration <= max_duration):
                    continue
                if dp[k-1][j] != float('inf'):
                    current_priority_sum = dp[k-1][j] + p_i
                    if current_priority_sum < dp[k][i]:
                        dp[k][i] = current_priority_sum
                        path[k][i] = j

    best_k, best_last_idx, min_total_priority = -1, -1, float('inf')
    for k in range(max_cuts + 1):
        for i in range(len(points)):
            last_segment_duration = total_dur - points[i]['time']
            if dp[k][i] != float('inf') and min_duration <= last_segment_duration <= max_duration:
                if best_k == -1 or k < best_k or (k == best_k and dp[k][i] < min_total_priority):
                    best_k, best_last_idx, min_total_priority = k, i, dp[k][i]
        if best_k != -1 and best_k == k:
            break

    if best_k == -1:
        return None

    split_times = []
    k, i = best_k, best_last_idx
    while k > 0 and i != -1:
        split_times.append(points[i]['time'])
        i = path[k][i]
        k -= 1

    split_times.reverse()
    return split_times

def split_audio_and_lab(audio_path, lab_path, outdir, lang="zh", align_model=None, metadata=None, device=None):
    """DEPRECATED: This function is part of the old structure. Use the AudioSplitter class instead."""
    """
    Args:
        audio_path (str): 输入音频文件的路径。
        lab_path (str): 输入文本文件的路径。
        outdir (str): 输出文件存放的目录。
        lang (str): 语言代码，默认为 "zh" (中文)。
        device (str, optional): 指定运行模型的设备（"cuda" 或 "cpu"）。如果为 None，则自动检测。
        min_duration (float, optional): 切分后每个片段的最小秒数，默认为 3.0。
    返回: (out_wavs) 包含所有切分后音频路径的列表，或在异常情况下抛出异常。
    """
    audio_path = os.path.abspath(audio_path)
    lab_path = os.path.abspath(lab_path)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    with open(lab_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    total_dur = get_wav_duration_seconds(audio_path)
    max_duration = DEFAULT_MAX_DURATION
    min_duration = DEFAULT_MIN_DURATION
    max_segments = DEFAULT_MAX_SEGMENTS

    if total_dur <= max_duration:
        logger.info("[split_audio_and_lab] Audio is short enough, no split needed for %s.", audio_path)
        print(f"[INFO] Audio is short enough, no split needed for {os.path.basename(audio_path)}.")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_wav = os.path.join(outdir, base + "_part1.wav")
        out_lab = os.path.join(outdir, base + "_part1.lab")
        if not os.path.exists(out_wav):
            shutil.copy(audio_path, out_wav)
            shutil.copy(lab_path, out_lab)
        return [out_wav]

    split_times, aligned = None, None
    if total_dur > max_duration * max_segments:
        logger.warning("[split_audio_and_lab] Audio is too long (%.2fs) to be split into %d segments of max %.2fs. It will be split into %d equal parts.", total_dur, max_segments, max_duration, max_segments)
        print(f"[WARN] Audio is too long ({total_dur:.2f}s) to be split into {max_segments} segments of max {max_duration}s. It will be split into {max_segments} equal parts.")
        num_segs = max_segments
        seg_dur = total_dur / num_segs
        split_times = [seg_dur * (i + 1) for i in range(num_segs - 1)]
    else:
        local_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        segments = [{"start": 0.0, "end": float(total_dur), "text": transcript}]
        try:
            if align_model is None or metadata is None:
                logger.info("[split_audio_and_lab] Loading alignment model (single-call mode).")
                print("[INFO] Loading alignment model (single-call mode).")
                align_model, metadata = whisperx.load_align_model(language_code=lang, device=local_device)
            else:
                logger.info("[split_audio_and_lab] Using provided alignment model (reused by caller).")

            audio = whisperx.load_audio(audio_path)

            align_start_time = time.time()
            aligned = whisperx.align(segments, align_model, metadata, audio, local_device, return_char_alignments=False)
            align_duration = time.time() - align_start_time
            logger.info("Alignment for '%s' took %.2f seconds.", os.path.basename(audio_path), align_duration)
            
            potential_points = _get_potential_split_points(aligned, total_dur)

            plan_start_time = time.time()
            split_times = _find_best_split_plan(potential_points, total_dur, max_duration, max_segments - 1, min_duration)
            plan_duration = time.time() - plan_start_time
            logger.info("Split planning for '%s' took %.2f seconds.", os.path.basename(audio_path), plan_duration)

            if split_times:
                logger.info("[split_audio_and_lab] Found optimal split plan with %d cuts: %s", len(split_times), [round(t, 2) for t in split_times])
                print(f"[INFO] Found optimal split plan with {len(split_times)} cuts for {os.path.basename(audio_path)}.")
            else:
                logger.warning("[split_audio_and_lab] Could not find an optimal split plan for %s. Falling back.", audio_path)
                print(f"[WARN] Could not find an optimal split plan for {os.path.basename(audio_path)}. Falling back.")

        except Exception as e:
            logger.warning("[split_audio_and_lab] Alignment or planning failed for %s: %s. Falling back.", audio_path, e)
            print(f"[WARN] Alignment or planning failed for {os.path.basename(audio_path)}: {e}. Falling back.", file=sys.stderr)

    if split_times is None:
        # Fallback to even splitting
        n_min = math.ceil(total_dur / max_duration)
        n_max = math.floor(total_dur / min_duration) if min_duration > 0 else max_segments

        possible_n_segs = [n for n in range(int(n_min), int(n_max) + 1) if n <= max_segments]

        if not possible_n_segs or possible_n_segs[0] <= 1:
            logger.warning("[split_audio_and_lab] Cannot find a valid number of segments for even split that satisfies all duration constraints for %s. The audio will not be split.", audio_path)
            print(f"[WARN] Cannot find a valid even split for {os.path.basename(audio_path)}. The audio will not be split.")
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_wav = os.path.join(outdir, base + "_part1.wav")
            out_lab = os.path.join(outdir, base + "_part1.lab")
            if not os.path.exists(out_wav):
                shutil.copy(audio_path, out_wav)
                shutil.copy(lab_path, out_lab)
            return [out_wav]
        num_segs = possible_n_segs[0]
        split_times = [(total_dur / num_segs) * (i + 1) for i in range(num_segs - 1)]
        aligned = None

    base = os.path.splitext(os.path.basename(audio_path))[0]
    try:
        out_wavs = split_wav_by_times(audio_path, split_times, outdir, base)
    except Exception as e:
        logger.error("[split_audio_and_lab] Failed to split audio %s: %s", audio_path, e)
        raise

    all_words = (aligned or {}).get("word_segments", [])
    split_boundaries = [0] + split_times + [total_dur]
    for i in range(len(out_wavs)):
        start_t, end_t = split_boundaries[i], split_boundaries[i+1]
        text_part = ""
        if all_words:
            part_words = [w['word'] for w in all_words if start_t <= w.get('start', -1) < end_t]
            text_part = "".join(part_words).strip()

        if not text_part:
            logger.info("Word-based text split failed. Falling back to proportional text split.")
            start_char = int(len(transcript) * (start_t / total_dur))
            end_char = int(len(transcript) * (end_t / total_dur))
            text_part = transcript[start_char:end_char].strip()

        out_lab = os.path.splitext(out_wavs[i])[0] + ".lab"
        with open(out_lab, "w", encoding="utf-8") as f:
            f.write(text_part)

    map_path = os.path.join(outdir, "切分记录.txt")
    try:
        with open(map_path, "a", encoding="utf-8") as mf:
            out_basenames = ", ".join(os.path.basename(p) for p in out_wavs)
            split_times_str = ", ".join(f"{t:.3f}s" for t in split_times)
            mf.write(f"{out_basenames} <- {os.path.basename(audio_path)}, splits=[{split_times_str}]\n")
    except Exception:
        pass

    return out_wavs

class AudioSplitter:
    """
    A class to handle splitting long audio files based on whisperx alignment.
    It encapsulates model loading, splitting logic, and batch processing.
    """

    def __init__(self, lang="zh", device=None, min_duration=3.0, logger_arg=None):
        """
        Initializes the AudioSplitter.

        Args:
            lang (str): Language code for the alignment model.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
            min_duration (float): Minimum duration for a split segment.
            logger_arg (logging.Logger, optional): External logger to use.
        """
        self.lang = lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_duration = min_duration
        self.logger = logger_arg or logger
        self.align_model = None
        self.metadata = None

    def _load_model(self):
        """Loads the whisperx alignment model if it hasn't been loaded yet."""
        if self.align_model is None or self.metadata is None:
            self.logger.info("Loading alignment model...")
            print(f"[INFO] Loading alignment model for language '{self.lang}' on device '{self.device}'.")
            try:
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=self.lang, device=self.device
                )
            except Exception as e:
                self.logger.error(f"Failed to load alignment model: {e}")
                print(f"[ERROR] Failed to load alignment model: {e}", file=sys.stderr)
                raise

    def _split_single(self, audio_path, lab_path, outdir):
        """
        Processes a single audio file and its corresponding text file for splitting.
        """
        audio_path = os.path.abspath(audio_path)
        lab_path = os.path.abspath(lab_path)
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)

        with open(lab_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        total_dur = get_wav_duration_seconds(audio_path)

        if total_dur <= DEFAULT_MAX_DURATION:
            self.logger.info(f"Audio is short enough, no split needed for {audio_path}.")
            print(f"[INFO] Audio is short enough, no split needed for {os.path.basename(audio_path)}.")
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_wav = os.path.join(outdir, base + "_part1.wav")
            out_lab = os.path.join(outdir, base + "_part1.lab")
            if not os.path.exists(out_wav):
                shutil.copy(audio_path, out_wav)
                shutil.copy(lab_path, out_lab)
            return [out_wav]

        split_times, aligned = None, None
        if total_dur > DEFAULT_MAX_DURATION * DEFAULT_MAX_SEGMENTS:
            self.logger.warning(f"Audio is too long ({total_dur:.2f}s) to be split into {DEFAULT_MAX_SEGMENTS} segments of max {DEFAULT_MAX_DURATION}s. It will be split into {DEFAULT_MAX_SEGMENTS} equal parts.")
            print(f"[WARN] Audio is too long ({total_dur:.2f}s) to be split into {DEFAULT_MAX_SEGMENTS} segments of max {DEFAULT_MAX_DURATION}s. It will be split into {DEFAULT_MAX_SEGMENTS} equal parts.")
            seg_dur = total_dur / DEFAULT_MAX_SEGMENTS
            split_times = [seg_dur * (i + 1) for i in range(DEFAULT_MAX_SEGMENTS - 1)]
        else:
            segments = [{"start": 0.0, "end": float(total_dur), "text": transcript}]
            try:
                audio = whisperx.load_audio(audio_path)
                align_start_time = time.time()
                aligned = whisperx.align(segments, self.align_model, self.metadata, audio, self.device, return_char_alignments=False)
                align_duration = time.time() - align_start_time
                self.logger.info(f"Alignment for '{os.path.basename(audio_path)}' took {align_duration:.2f} seconds.")
                
                potential_points = _get_potential_split_points(aligned, total_dur)
                plan_start_time = time.time()
                split_times = _find_best_split_plan(potential_points, total_dur, DEFAULT_MAX_DURATION, DEFAULT_MAX_SEGMENTS - 1, self.min_duration)
                plan_duration = time.time() - plan_start_time
                self.logger.info(f"Split planning for '{os.path.basename(audio_path)}' took {plan_duration:.2f} seconds.")

                if split_times:
                    self.logger.info(f"Found optimal split plan with {len(split_times)} cuts: {[round(t, 2) for t in split_times]}")
                    print(f"[INFO] Found optimal split plan with {len(split_times)} cuts for {os.path.basename(audio_path)}.")
                else:
                    self.logger.warning(f"Could not find an optimal split plan for {audio_path}. Falling back.")
                    print(f"[WARN] Could not find an optimal split plan for {os.path.basename(audio_path)}. Falling back.")
            except Exception as e:
                self.logger.warning(f"Alignment or planning failed for {audio_path}: {e}. Falling back.")
                print(f"[WARN] Alignment or planning failed for {os.path.basename(audio_path)}: {e}. Falling back.", file=sys.stderr)

        if split_times is None:
            # Fallback to even splitting
            n_min = math.ceil(total_dur / DEFAULT_MAX_DURATION)
            n_max = math.floor(total_dur / self.min_duration) if self.min_duration > 0 else DEFAULT_MAX_SEGMENTS
            possible_n_segs = [n for n in range(int(n_min), int(n_max) + 1) if n <= DEFAULT_MAX_SEGMENTS]

            if not possible_n_segs or possible_n_segs[0] <= 1:
                self.logger.warning(f"Cannot find a valid even split for {audio_path}. The audio will not be split.")
                print(f"[WARN] Cannot find a valid even split for {os.path.basename(audio_path)}. The audio will not be split.")
                base = os.path.splitext(os.path.basename(audio_path))[0]
                out_wav = os.path.join(outdir, base + "_part1.wav")
                out_lab = os.path.join(outdir, base + "_part1.lab")
                if not os.path.exists(out_wav):
                    shutil.copy(audio_path, out_wav)
                    shutil.copy(lab_path, out_lab)
                return [out_wav]
            
            num_segs = possible_n_segs[0]
            split_times = [(total_dur / num_segs) * (i + 1) for i in range(num_segs - 1)]
            aligned = None

        base = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            out_wavs = split_wav_by_times(audio_path, split_times, outdir, base)
        except Exception as e:
            self.logger.error(f"Failed to split audio {audio_path}: {e}")
            raise

        all_words = (aligned or {}).get("word_segments", [])
        split_boundaries = [0] + split_times + [total_dur]
        for i in range(len(out_wavs)):
            start_t, end_t = split_boundaries[i], split_boundaries[i+1]
            text_part = ""
            if all_words:
                part_words = [w['word'] for w in all_words if start_t <= w.get('start', -1) < end_t]
                text_part = "".join(part_words).strip()

            if not text_part:
                self.logger.info("Word-based text split failed. Falling back to proportional text split.")
                start_char = int(len(transcript) * (start_t / total_dur))
                end_char = int(len(transcript) * (end_t / total_dur))
                text_part = transcript[start_char:end_char].strip()

            out_lab = os.path.splitext(out_wavs[i])[0] + ".lab"
            with open(out_lab, "w", encoding="utf-8") as f:
                f.write(text_part)

        map_path = os.path.join(outdir, "切分记录.txt")
        try:
            with open(map_path, "a", encoding="utf-8") as mf:
                out_basenames = ", ".join(os.path.basename(p) for p in out_wavs)
                split_times_str = ", ".join(f"{t:.3f}s" for t in split_times)
                mf.write(f"{out_basenames} <- {os.path.basename(audio_path)}, splits=[{split_times_str}]\n")
        except Exception:
            pass

        return out_wavs

    def split(self, pairs, outdir):
        """
        Processes a batch of audio/text pairs for splitting.
        """
        self._load_model()
        results = {}
        for (audio_path, lab_path) in pairs:
            try:
                out_wavs = self._split_single(audio_path, lab_path, outdir)
                results[audio_path] = out_wavs
            except Exception as e:
                self.logger.warning(f"Split failed for {audio_path}: {e}")
                print(f"[WARN] Split failed for {audio_path}: {e}", file=sys.stderr)
                results[audio_path] = None
        return results

def split_batch(pairs, outdir, lang="zh", logger_arg=None):
    """
    Batch processing function for splitting audio and text file pairs.
    This function is a convenient wrapper around the AudioSplitter class.

    Args:
        pairs (list): A list of (audio_path, lab_path) tuples.
        outdir (str): The directory to save output files.
        lang (str): Language code for the alignment model.
        logger_arg (logging.Logger, optional): An external logger instance.
        min_duration (float): Minimum duration for a split segment.

    Returns:
        dict: A dictionary mapping original audio paths to a list of split WAV paths, or None on failure.
    """
    splitter = AudioSplitter(lang=lang, logger_arg=logger_arg)
    return splitter.split(pairs, outdir)