import os
import sys
import wave
import contextlib
from datetime import timedelta
import logging
import shutil
import re
import csv
import json
from pathlib import Path

# 导入修改后的分割函数（包含批量接口）
from split_whisperx_two_parts import split_batch
# 导入情感分类功能
from AudioClassification_structbert import run_inference as classify_emotions
import AudioClassification_structbert as emotion_model

def main():
    """主入口函数"""
    # ================================================
    # 配置区域
    ROOT_DIR = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\芙宁娜"
    # 新增一个布尔变量，True表示使用标注文本命名，False表示保留原始文件名（或生成新格式）
    USE_LAB_TEXT_AS_FILENAME = False
    # 新增一个布尔变量，True表示对音频进行情感分类并按情感分文件夹，False则不进行分类
    ENABLE_EMOTION_CLASSIFICATION = True
    REPLACE_MAP = {
        "|": "", "\n": " ", "\r": "", "「": "", "」": "", " / ":"，", "（":"", "）":"", "(":"", ")":"", "#":"", 
        "{M#他}{F#她}":"他", "{M#他们}{F#她们}":"他们", "{M他}{F她}":"他", "{M他们}{F她们}":"他们", "{NICKNAME}":"旅行者"
        }
    # ================================================

    try:
        processor = DatasetProcessor(Path(ROOT_DIR), REPLACE_MAP, USE_LAB_TEXT_AS_FILENAME, ENABLE_EMOTION_CLASSIFICATION)
        processor.run()
    except Exception as e:
        print(f"运行出错: {e}", file=sys.stderr)
        # 如果日志记录器已配置，则记录异常
        logger = logging.getLogger("tts_preprocess")
        if logger.handlers:
            logger.exception("发生未处理的异常")
        sys.exit(1)

def sanitize_filename(filename):
    """清理文件名中的无效字符，并限制长度"""
    # Windows invalid chars: <>:"/\|?*
    # Also remove leading/trailing spaces and dots which can be problematic.
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return sanitized.strip(' .')[:200] # Limit length to be safe

def read_lab_utf8(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def get_wav_duration_seconds(path):
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            raise RuntimeError("采样率为0")
        duration = frames / float(rate)
    return duration

def format_hms(total_seconds):
    total_secs = int(round(total_seconds))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    seconds = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_hms_filename(total_seconds):
    return format_hms(total_seconds).replace(":", "-")

def read_wav_params_and_frames(path):
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
    return {
        "nchannels": params.nchannels,
        "sampwidth": params.sampwidth,
        "framerate": params.framerate,
        "comptype": params.comptype,
        "nframes": params.nframes
    }, frames

def merge_wavs(output_path, wav_infos, silence_duration=0.5):
    if not wav_infos:
        raise ValueError("没有 wav 可合并")
    base = wav_infos[0]["params"]
    nch = base["nchannels"]
    sw = base["sampwidth"]
    fr = base["framerate"]
    for info in wav_infos[1:]:
        p = info["params"]
        if p["nchannels"] != nch or p["sampwidth"] != sw or p["framerate"] != fr or p["comptype"] != base["comptype"]:
            raise RuntimeError(f"音频参数不匹配，无法合并: {info['path']}")
    silence_frames = int(round(fr * silence_duration))
    silence_bytes = (b'\x00' * sw * nch) * silence_frames
    with contextlib.closing(wave.open(str(output_path), "wb")) as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(fr)
        for idx, info in enumerate(wav_infos):
            wf.writeframes(info["frames"])
            if idx != len(wav_infos) - 1:
                wf.writeframes(silence_bytes)

def call_split_script_batch(pairs, outdir, logger=None):
    """
    批量调用 split_batch，pairs 是 [(wav_path, lab_path), ...]
    返回字典 {wav_path: [out1.wav, out2.wav, ...]}
    """
    try:
        results = split_batch(pairs, outdir, lang="zh", logger_arg=logger)
        return results
    except Exception as e:
        print(f"[WARN] Batch split function failed: {e}", file=sys.stderr)
        # 返回所有失败项
        return {a: None for (a, _) in pairs}

class DatasetProcessor:
    """封装整个数据集处理流程的类"""

    # --- Constants for configuration and magic strings ---
    SHORT_DUR_THRESHOLD = 3.0
    LONG_DUR_THRESHOLD = 25.0
    TEMP_DIR_NAME = "_temp_processing"
    REVIEW_DIR_NAME = "待审阅_切分文件"
    MERGE_LOG_FILENAME = "！合并记录.txt"
    SPLIT_LOG_FILENAME = "！切分记录.txt"
    LOG_FILENAME_PREFIX = "！"
    # --- End of Constants ---

    def __init__(self, root_dir, replace_map, use_lab_text_as_filename=False, enable_emotion_classification=True):
        """
        初始化处理器。
        :param root_dir: 输入数据集的根目录。
        :param replace_map: 标注文本的替换规则字典。
        :param use_lab_text_as_filename: 是否使用标注文本作为文件名。
        :param enable_emotion_classification: 是否启用情感分类。
        """
        self.root_dir = root_dir.resolve()
        if not self.root_dir.is_dir():
            raise ValueError(f"指定的根目录不存在或不是目录: {self.root_dir}")

        self.use_lab_text_as_filename = use_lab_text_as_filename
        self.replace_map = replace_map
        self.enable_emotion_classification = enable_emotion_classification
        self.folder_name = self.root_dir.name
        if not self.folder_name:
            raise ValueError(f"无法从根目录派生文件夹名称: {self.root_dir}")

        # 路径和日志记录器将在 setup 阶段初始化
        self.output_dir = None
        self.temp_dir = None
        self.auto_merge_dir = None
        self.auto_split_dir = None
        self.log_path = None
        self.review_dir = None
        self.out_list_path = None
        self.logger = None
        self.emotion_results = {}  # 存储情感分类结果

    def _setup_paths_and_logging(self):
        """设置所有输出路径并配置日志记录。"""
        self.output_dir = self.root_dir.parent / "output" / self.folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 所有输出将被保存到: {self.output_dir}")

        # 为中间文件（切分、合并）创建临时目录
        self.temp_dir = self.output_dir / self.TEMP_DIR_NAME
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        self.auto_merge_dir = self.temp_dir
        self.auto_split_dir = self.temp_dir

        # 为人工审阅切分文件创建目录
        self.review_dir = self.output_dir / self.REVIEW_DIR_NAME
        if self.review_dir.exists():
            shutil.rmtree(self.review_dir)
        self.review_dir.mkdir()

        # 日志文件名前加!置顶，.list文件输出到父级output目录
        self.log_path = self.output_dir / f"{self.LOG_FILENAME_PREFIX}{self.folder_name}.log"
        self.out_list_path = self.output_dir.parent / f"{self.folder_name}.list"

        self.logger = logging.getLogger("tts_preprocess")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self.logger.info("="*50)
        self.logger.info("开始新的处理流程")
        self.logger.info("配置信息:")
        self.logger.info(f"  - 输入目录: {self.root_dir}")
        self.logger.info(f"  - 输出目录: {self.output_dir}")
        self.logger.info(f"  - 输出列表文件: {self.out_list_path}")
        self.logger.info(f"  - 使用标注文本命名: {self.use_lab_text_as_filename}")
        self.logger.info(f"  - 启用情感分类: {self.enable_emotion_classification}")
        self.logger.info(f"  - 替换规则: {self.replace_map}")
        self.logger.info("="*50)

    def _run_emotion_classification(self, input_dir):
        """运行情感分类并返回结果"""
        print(f"[INFO] 开始对目录 {input_dir} 中的文件进行情感分类...")
        self.logger.info(f"开始对目录 {input_dir} 中的文件进行情感分类")
        
        temp_csv = self.temp_dir / "emotions_temp.csv"
        
        # [SUGGESTION] The following interaction with a module via `sys.modules` is brittle.
        # A better approach is to modify `classify_emotions` to accept parameters directly,
        # e.g., `classify_emotions(input_dir=self.root_dir, output_csv=temp_csv)`.
        # 检查是否在AudioClassification模块中定义了这些全局变量
        emotion_model.ROOT_DIR = str(input_dir)
        emotion_model.OUT_CSV = str(temp_csv)
        
        try:
            # 运行情感分类
            classify_emotions()
            
            # 读取分类结果
            emotion_results = {}
            if temp_csv.exists():
                # 使用 Path 对象打开文件
                with temp_csv.open('r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        emotion_results[row['audio_path']] = row['pred_label']
            
            return emotion_results
        except Exception as e:
            self.logger.error(f"情感分类失败: {e}")
            raise RuntimeError("情感分类函数执行失败") from e

    def _find_audio_entries(self):
        """查找、配对并读取所有音频条目及其元数据。"""
        file_entries = [p for p in self.root_dir.iterdir() if p.is_file()]

        stems = {}
        for path in file_entries:
            name = path.stem
            ext_lower = path.suffix.lower()
            if ext_lower in (".wav", ".lab"):
                stems.setdefault(name, {})[ext_lower] = path

        self.logger.info(f"在根目录中找到 {len(list(self.root_dir.iterdir()))} 个文件/文件夹。")
        self.logger.info(f"找到 {len(stems)} 个带有 .wav 或 .lab 扩展名的唯一文件主干。")

        matched = []
        skipped = 0
        read_failures = []
        for stem, extmap in stems.items():
            if ".wav" not in extmap or ".lab" not in extmap:
                if ".wav" in extmap:
                    self.logger.info(f"跳过 '{stem}': 找到 .wav 文件但没有匹配的 .lab 文件。")
                elif ".lab" in extmap:
                    self.logger.info(f"跳过 '{stem}': 找到 .lab 文件但没有匹配的 .wav 文件。")
                skipped += 1
                continue

            # At this point, we know both .wav and .lab files exist.
            wav_path = extmap[".wav"]
            lab_path = extmap[".lab"]

            try:
                lab_text = read_lab_utf8(lab_path).strip()
                # 策略变更：在读取文本时立即应用替换规则，确保后续流程中的文本都是干净的
                for k, v in self.replace_map.items():
                    lab_text = lab_text.replace(k, v)

            except Exception as e:
                read_failures.append(f"{lab_path} (读取 .lab): {e}")
                self.logger.warning(f"读取 .lab {lab_path} 失败: {e}")
                skipped += 1
                continue

            try:
                duration = get_wav_duration_seconds(wav_path)
            except Exception as e:
                read_failures.append(f"{wav_path} (获取时长): {e}")
                self.logger.warning(f"获取 {wav_path} 的时长失败: {e}")
                skipped += 1
                continue

            matched.append({
                "stem": stem, "wav_path": wav_path, "lab_path": lab_path,
                "lab_text": lab_text, "duration": duration,
                "source_type": "original" # 标记来源
            })

        if read_failures:
            print(f"[WARN] 文件读取期间遇到 {len(read_failures)} 个错误。详情请查看日志文件: {self.log_path}", file=sys.stderr)

        self.logger.info(f"匹配到 {len(matched)} 对 .wav 和 .lab 文件。")
        if skipped > 0:
            self.logger.warning(f"跳过了 {skipped} 个文件（由于缺少配对或读取错误）。")

        return matched

    def _process_long_files(self, long_list):
        """处理长音频，进行自动切分。"""
        if not long_list:
            return []

        # After changes in _find_audio_entries, all entries will have a lab_path.
        long_list_to_split = long_list

        self.logger.info(f"  - 在 {len(long_list)} 个长文件中, {len(long_list_to_split)} 个已准备好进行自动切分。")

        unsplit_entries = []
        if long_list_to_split:
            print(f"[INFO] 准备切分 {len(long_list_to_split)} 个长音频文件。")
            self.logger.info(f"准备对 {len(long_list_to_split)} 个长音频进行批量切分。")

            pairs = [(str(m["wav_path"]), str(m["lab_path"])) for m in long_list_to_split]
            batch_results = call_split_script_batch(pairs, self.auto_split_dir, logger=self.logger)

            split_failures = []
            for m in long_list_to_split:
                wavp = m["wav_path"] # This is a Path object
                split_wav_paths = batch_results.get(str(wavp))

                if split_wav_paths:
                    try:
                        split_wav_paths_p = [Path(p) for p in split_wav_paths]
                        if not all(p.is_file() for p in split_wav_paths_p):
                            raise FileNotFoundError("部分切分后的wav文件未找到。")

                        for split_wav_path in split_wav_paths_p:
                            split_lab_path = split_wav_path.with_suffix(".lab")
                            if not split_lab_path.is_file():
                                raise FileNotFoundError(f"未找到切分后的 lab 文件: {split_lab_path}")
                            
                            # 移动到审阅目录
                            shutil.move(split_wav_path, self.review_dir / split_wav_path.name)
                            shutil.move(split_lab_path, self.review_dir / split_lab_path.name)

                        self.logger.info(f"切分成功并将文件移至待审阅文件夹: {wavp} -> {', '.join(p.name for p in split_wav_paths_p)}")
                    except Exception as e:
                        split_failures.append(wavp.name)
                        self.logger.warning(f"处理 {wavp} 的切分结果失败 (移动文件时出错): {e}")
                        unsplit_entries.append({"wav_path": wavp, "lab_text": m["lab_text"], "duration": m["duration"], "source_type": "original_long"})
                else:
                    split_failures.append(wavp.name)
                    self.logger.warning(f"{wavp} 切分失败或输出缺失。")
                    unsplit_entries.append({"wav_path": wavp, "lab_text": m["lab_text"], "duration": m["duration"], "source_type": "original_long"})
            
            if split_failures:
                print(f"[WARN] {len(split_failures)} 个长音频文件自动切分失败，已按原样添加。请考虑手动切分。详情请查看日志。", file=sys.stderr)

        return unsplit_entries

    def _process_short_files(self, short_list):
        """处理短音频，进行自动合并。优先合并相同情感的音频，然后跨情感合并以满足时长要求。"""
        if not short_list:
            return [], []

        # Stage 1: Same-emotion grouping
        emotion_groups = {}
        for item in short_list:
            emotion = item["emotion"]
            emotion_groups.setdefault(emotion, []).append(item)

        groups_to_merge = []
        leftovers = []
        for emotion, items in emotion_groups.items():
            i = 0
            while i < len(items):
                remaining = len(items) - i
                if remaining >= 3:
                    groups_to_merge.append(items[i:i+3])
                    i += 3
                elif remaining == 2:
                    groups_to_merge.append(items[i:i+2])
                    i += 2
                else:  # remaining == 1
                    leftovers.append(items[i])
                    i += 1
        
        self.logger.info(f"同情感合并：形成 {len(groups_to_merge)} 组，剩余 {len(leftovers)} 个文件待跨情感合并。")

        # Stage 2: Cross-emotion grouping for leftovers
        final_unmerged_entries = []
        if len(leftovers) > 1:
            self.logger.info(f"开始对 {len(leftovers)} 个剩余短文件进行跨情感合并。")
            leftovers.sort(key=lambda x: x['duration'], reverse=True)
            
            while leftovers:
                group = [leftovers.pop(0)]
                duration = group[0]['duration']

                # Greedily add more files to this group to exceed the threshold
                i = 0
                while i < len(leftovers):
                    item_to_add = leftovers[i]
                    if duration < self.SHORT_DUR_THRESHOLD and (duration + item_to_add['duration']) < self.LONG_DUR_THRESHOLD:
                        group.append(leftovers.pop(i))
                        duration += item_to_add['duration']
                    else:
                        i += 1
                
                if len(group) > 1:
                    groups_to_merge.append(group)
                else:
                    final_unmerged_entries.append(group[0])
        else:
            final_unmerged_entries.extend(leftovers)

        # Stage 3: Merging
        if not groups_to_merge:
            unmerged = [{"wav_path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"], "source_type": "original_short"} for item in short_list]
            return unmerged, []

        print(f"[INFO] 准备将短音频文件合并成 {len(groups_to_merge)} 组。")

        processed_entries = []
        merge_map_lines = []
        merge_read_failures, merge_failures = 0, 0

        for group_idx, group in enumerate(groups_to_merge, start=1):
            wav_infos, ok = [], True
            for item in group:
                try:
                    params, frames = read_wav_params_and_frames(item["wav_path"])
                    wav_infos.append({"params": params, "frames": frames, "path": str(item["wav_path"]), "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"]})
                except Exception as e:
                    self.logger.warning(f"读取用于合并的 wav 文件失败 {item['wav_path']}: {e}")
                    merge_read_failures += 1
                    ok = False
                    break

            if not ok:
                merge_failures += 1
                self.logger.warning("回退: 因读取错误，将该组的原始文件单独添加到列表。")
                processed_entries.extend([{"wav_path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"], "source_type": "original_short"} for item in group])
                continue

            # Check for cross-emotion merge and set emotion for the new file
            emotions_in_group = {info['emotion'] for info in wav_infos}
            emotion = group[-1]["emotion"] # Per request: use the emotion of the LAST item
            if len(emotions_in_group) > 1:
                self.logger.warning(f"跨情感合并短音频: 组中包含的情感有 {emotions_in_group}。合并后的文件将使用情感 '{emotion}'。")

            merge_name = f"{self.folder_name}_{emotion}_merge_{group_idx}_{format_hms_filename(sum(i['duration'] for i in group))}.wav"
            merge_path = self.auto_merge_dir / merge_name
            try:
                merge_wavs(merge_path, wav_infos, silence_duration=0.5)
                merge_lab = "".join(i["lab_text"] for i in wav_infos)
                processed_entries.append({"wav_path": merge_path, "lab_text": merge_lab, "duration": sum(i["duration"] for i in wav_infos), "emotion": emotion, "source_type": "merged"})
                merge_map_lines.append(f"{merge_path} <- " + ", ".join(str(i["path"]) for i in wav_infos))
            except Exception as e:
                merge_failures += 1
                self.logger.warning(f"第 {group_idx} 组合并失败: {e}")
                self.logger.warning("回退: 因合并失败，将该组的原始文件单独添加到列表。")
                processed_entries.extend([{"wav_path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"], "source_type": "original_short"} for item in group])

        if merge_read_failures > 0:
            print(f"[WARN] {merge_read_failures} 个短音频文件读取失败，无法合并。详情请查看日志。", file=sys.stderr)
        if merge_failures > 0:
            print(f"[WARN] {merge_failures} 组短音频文件合并失败，已将它们单独添加。详情请查看日志。", file=sys.stderr)

        for item in final_unmerged_entries:
            processed_entries.append({"wav_path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"], "emotion": item["emotion"], "source_type": "original_short"})
        return processed_entries, merge_map_lines

    def _wait_for_review_and_process(self):
        """等待用户审阅切分文件，然后处理它们。"""
        if not any(self.review_dir.iterdir()):
            self.logger.info("待审阅文件夹为空，跳过人工审阅步骤。")
            return []

        print("="*50)
        print(f"[INFO] {len(list(self.review_dir.glob('*.wav')))} 对切分文件已生成并等待审阅。")
        input(f"\n请检查文件夹 '{self.review_dir}' 中的文件。\n"
              f"您可以删除不需要的文件对，或修改 .lab 文件的内容。\n"
              f"审阅完成后，请按 Enter 键继续处理...")
        print("="*50)
        print("[INFO] 审阅完成，正在处理审阅后的文件...")
        self.logger.info("用户审阅完成，开始处理审阅后的文件。")

        # 重新扫描审阅文件夹
        lab_files = sorted(self.review_dir.glob("*.lab"))
        
        reviewed_entries = []
        for lab_path in lab_files:
            wav_path = lab_path.with_suffix(".wav")

            if not wav_path.exists():
                self.logger.warning(f"审阅后发现 {lab_path.name} 但缺少对应的 .wav 文件，已跳过。")
                continue

            try:
                lab_text = read_lab_utf8(lab_path).strip()
                # 再次应用替换规则，以防用户在审阅时添加了需要替换的字符
                for k, v in self.replace_map.items():
                    lab_text = lab_text.replace(k, v)
                duration = get_wav_duration_seconds(wav_path)
                
                reviewed_entries.append({
                    "wav_path": wav_path,
                    "lab_text": lab_text,
                    "duration": duration,
                    "source_type": "reviewed_split"
                })
                self.logger.info(f"已处理审阅后的文件: {wav_path.name}")
            except Exception as e:
                self.logger.error(f"处理审阅后的文件 {wav_path} 失败: {e}")
        return reviewed_entries

    def _finalize_output(self, final_entries, merge_map_lines, initial_count):
        """最终确定输出文件：按情感分类，重命名、复制并写入所有摘要和列表文件。"""
        print("="*50)
        print("[INFO] 正在最终确定输出文件：按情感分类，重命名并复制到输出目录...")
        self.logger.info("正在最终确定输出文件：按情感分类，重命名并复制到输出目录。")

        # 按情感分组
        initial_emotion_groups = {}
        for entry in final_entries:
            emotion = entry.get("emotion", "unknown")
            if emotion not in initial_emotion_groups:
                initial_emotion_groups[emotion] = []
            initial_emotion_groups[emotion].append(entry)

        # 为每个情感创建子目录
        emotion_dirs = {}
        for emotion in initial_emotion_groups.keys():
            emotion_dir = self.output_dir / emotion
            emotion_dir.mkdir(exist_ok=True)
            emotion_dirs[emotion] = emotion_dir

        final_output_entries = []
        processed_filenames = set()
        merge_counter = 1

        for emotion, entries in initial_emotion_groups.items():
            emotion_dir = emotion_dirs[emotion]
            for i, entry in enumerate(entries, start=1):
                source_path = entry["wav_path"]
                lab_text = entry["lab_text"]
                source_type = entry["source_type"]

                if self.use_lab_text_as_filename:
                    target_basename = sanitize_filename(lab_text) + ".wav"
                elif source_type == "merged":
                    target_basename = f"{self.folder_name}_{emotion}_auto_merge_{merge_counter}.wav"
                    merge_counter += 1
                elif source_type in ("original", "original_short", "original_long", "reviewed_split"):
                    target_basename = source_path.name
                else:
                    self.logger.warning(f"未知文件来源类型: {source_type} for {source_path}。将使用其当前文件名。")
                    target_basename = source_path.name

                temp_basename, counter = target_basename, 1
                while temp_basename in processed_filenames:
                    base, ext = os.path.splitext(target_basename)
                    temp_basename = f"{base}_{counter}{ext}"
                    counter += 1
                target_basename = temp_basename
                processed_filenames.add(target_basename)

                target_path = emotion_dir / target_basename
                try:
                    # 中间文件（来自切分或合并）位于临时目录中，应移动它们。
                    # 原始文件（正常长度）应被复制，以保持源数据集不变。
                    if source_type in ("merged", "reviewed_split"):
                        shutil.move(source_path, target_path)
                    else:
                        shutil.copy(source_path, target_path)

                    final_output_entries.append({
                        "wav_abs": str(target_path.resolve()),
                        "lab_text": lab_text,
                        "duration": entry["duration"],
                        "emotion": emotion
                    })
                except Exception as e:
                    print(f"[ERROR] 复制或移动 {source_path} 到 {target_path} 失败: {e}", file=sys.stderr)
                    self.logger.error(f"复制或移动 {source_path} 到 {target_path} 失败: {e}")

        if merge_map_lines:
            merge_map_path = self.output_dir / self.MERGE_LOG_FILENAME
            try:
                with merge_map_path.open("a", encoding="utf-8") as mf:
                    mf.write("\n".join(merge_map_lines) + "\n")
            except Exception as e:
                self.logger.warning(f"写入合并记录失败: {e}")

        # 移动并附加切分脚本生成的日志文件
        temp_split_log_path = self.temp_dir / "切分记录.txt" # This name comes from the external script
        if temp_split_log_path.exists():
            final_split_log_path = self.output_dir / self.SPLIT_LOG_FILENAME
            try:
                with temp_split_log_path.open("r", encoding="utf-8") as temp_f:
                    content = temp_f.read()
                with final_split_log_path.open("a", encoding="utf-8") as final_f:
                    final_f.write(content)
                temp_split_log_path.unlink()
                self.logger.info(f"已将切分记录附加到: {final_split_log_path}")
            except Exception as e:
                self.logger.warning(f"处理切分记录文件 {temp_split_log_path} 失败: {e}")

        # After processing, regroup the final entries for summary and list file writing
        final_emotion_groups = {}
        for entry in final_output_entries:
            emotion = entry["emotion"]
            final_emotion_groups.setdefault(emotion, []).append(entry)

        total_duration_seconds = sum(e['duration'] for e in final_output_entries)
        total_duration_hms = format_hms(total_duration_seconds)
        summary_text = (
            f"处理摘要:\n"
            f"  - 初始找到的音频文件: {initial_count}\n"
            f"  - 最终创建的输出文件: {len(final_output_entries)}\n"
            f"  - 最终音频总时长: {total_duration_hms} ({total_duration_seconds:.2f} 秒)\n"
            f"  - 情感分类: {', '.join([f'{e}: {len(v)}' for e, v in final_emotion_groups.items()])}"
        )
        print("="*50)
        print(summary_text)
        print("="*50)
        self.logger.info("="*50)
        self.logger.info("处理摘要:")
        self.logger.info(f"  - 初始找到的音频文件: {initial_count}")
        self.logger.info(f"  - 最终创建的输出文件: {len(final_output_entries)}")
        self.logger.info(f"  - 最终音频总时长: {total_duration_hms} ({total_duration_seconds:.2f} 秒)")
        self.logger.info(f"  - 情感分类: {', '.join([f'{e}: {len(v)}' for e, v in final_emotion_groups.items()])}")
        self.logger.info("="*50)

        duration_filename = f"!{format_hms_filename(total_duration_seconds)}.txt"
        duration_filepath = self.output_dir / duration_filename
        try:
            duration_filepath.touch()
            print(f"[INFO] 已创建时长文件: {duration_filepath}")
            self.logger.info(f"已创建时长文件: {duration_filepath}")
        except Exception as e:
            print(f"[WARN] 创建时长文件失败: {e}", file=sys.stderr)
            self.logger.warning(f"创建时长文件 {duration_filepath} 失败: {e}")

        # 为每个情感目录创建.list文件
        for emotion, entries in final_emotion_groups.items():
            emotion_dir = emotion_dirs[emotion]
            emotion_list_path = emotion_dir / f"{emotion}.list"
            try:
                with emotion_list_path.open("w", encoding="utf-8") as ol:
                    for ent in entries:
                        ol.write(f"{ent['wav_abs']}|{self.folder_name}|ZH|{ent['lab_text']}\n")
                self.logger.info(f"已将情感列表写入 {emotion_list_path} (条目数={len(entries)})")
            except Exception as e:
                print(f"[ERROR] 写入情感列表文件失败: {e}", file=sys.stderr)
                self.logger.error(f"写入情感列表 {emotion_list_path} 失败: {e}")

        # 创建总的.list文件
        try:
            with self.out_list_path.open("w", encoding="utf-8") as ol:
                for ent in final_output_entries:
                    ol.write(f"{ent['wav_abs']}|{self.folder_name}|ZH|{ent['lab_text']}\n")
            self.logger.info(f"已将总列表写入 {self.out_list_path} (条目数={len(final_output_entries)})")
        except Exception as e:
            print(f"[ERROR] 写入总列表文件失败: {e}", file=sys.stderr)
            self.logger.error(f"写入总列表 {self.out_list_path} 失败: {e}")

    def run(self):
        """执行完整的数据集处理流程。"""
        print(f"[INFO] 开始处理数据集: {self.root_dir}")
        self._setup_paths_and_logging()

        # 1. 查找并解析所有音频文件
        all_entries = self._find_audio_entries()
        initial_count = len(all_entries)

        # 2. 按时长分类
        short_list = [m for m in all_entries if m["duration"] <= self.SHORT_DUR_THRESHOLD]
        normal_list = [m for m in all_entries if self.SHORT_DUR_THRESHOLD < m["duration"] < self.LONG_DUR_THRESHOLD]
        long_list = [m for m in all_entries if m["duration"] >= self.LONG_DUR_THRESHOLD]
        self.logger.info(f"音频文件分类: {len(short_list)} 短 (<=3s), {len(normal_list)} 正常 (3-25s), {len(long_list)} 长 (>=25s)。")

        # 3. 处理长音频 (切分)
        unsplit_long_entries = self._process_long_files(long_list)

        # 4. 等待人工审阅切分文件
        reviewed_entries = self._wait_for_review_and_process()

        # 5. 集中进行情感分类
        all_files_to_process = normal_list + short_list + unsplit_long_entries + reviewed_entries
        if self.enable_emotion_classification and all_files_to_process:
            # 创建临时目录并复制文件
            classification_dir = self.temp_dir / "classification_input"
            classification_dir.mkdir(exist_ok=True)
            self.logger.info(f"创建临时分类目录: {classification_dir}")

            for entry in all_files_to_process:
                source_path = entry["wav_path"]
                target_path = classification_dir / source_path.name
                if not target_path.exists():
                    shutil.copy(source_path, target_path)

            # 运行分类
            self.emotion_results = self._run_emotion_classification(classification_dir)

            # 应用分类结果
            for entry in all_files_to_process:
                temp_path_str = str(classification_dir / entry["wav_path"].name)
                emotion = self.emotion_results.get(temp_path_str, "unknown")
                entry["emotion"] = emotion
            self.logger.info(f"情感分类完成，为 {len(all_files_to_process)} 个文件分配了标签。")
        else:
            self.logger.info("情感分类已禁用或无文件需要分类，为所有文件设置默认情感标签。")
            for entry in all_files_to_process:
                entry["emotion"] = "default"

        # 6. 按顺序处理并收集最终条目
        final_entries_before_copy = []

        # 6.1 添加正常长度、未切分的长音频和审阅过的音频
        final_entries_before_copy.extend(normal_list)
        final_entries_before_copy.extend(unsplit_long_entries)
        final_entries_before_copy.extend(reviewed_entries)
        print(f"[INFO] 已添加 {len(normal_list) + len(unsplit_long_entries) + len(reviewed_entries)} 个非合并音频文件。")

        # 6.2 处理短音频 (合并)
        processed_short_entries, merge_map_lines = self._process_short_files(short_list)
        final_entries_before_copy.extend(processed_short_entries)

        # 7. 最终确定输出
        self._finalize_output(final_entries_before_copy, merge_map_lines, initial_count)

        # 8. 清理临时文件
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理处理过程中生成的临时目录。"""
        if self.temp_dir and self.temp_dir.is_dir():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"已成功清理临时目录: {self.temp_dir}")
                print(f"[INFO] 已清理合并/切分临时工作目录。")
            except Exception as e:
                self.logger.error(f"清理临时目录 {self.temp_dir} 失败: {e}")
                print(f"[ERROR] 清理临时目录 {self.temp_dir} 失败: {e}", file=sys.stderr)
        if self.review_dir and self.review_dir.is_dir():
            try:
                shutil.rmtree(self.review_dir)
                self.logger.info(f"已成功清理审阅目录: {self.review_dir}")
                print(f"[INFO] 已清理审阅目录。")
            except Exception as e:
                self.logger.error(f"清理审阅目录 {self.review_dir} 失败: {e}")
                print(f"[ERROR] 清理审阅目录 {self.review_dir} 失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()