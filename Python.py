import os
import sys
import wave
import contextlib
from datetime import timedelta
import logging
import shutil
import re

# 导入修改后的分割函数（包含批量接口）
from split_whisperx_two_parts import split_batch

def main():
    """主入口函数"""
    # ================================================
    # 配置区域
    ROOT_DIR = r"D:\Justin\0software\GPT-SoVITS-v2pro-20250604\input\七七" 
    # 新增一个布尔变量，True表示使用标注文本命名，False表示保留原始文件名（或生成新格式）
    USE_LAB_TEXT_AS_FILENAME = True
    REPLACE_MAP = {
        "|": "", "\n": " ", "\r": "", "「": "", "」": "", " / ":"，", "（":"", "）":"", "(":"", ")":"", "#":"", 
        "{M#他}{F#她}":"他", "{M#他们}{F#她们}":"他们", "{M他}{F她}":"他", "{M他们}{F她们}":"他们", "{NICKNAME}":"旅行者"
        }
    # ================================================

    try:
        processor = DatasetProcessor(ROOT_DIR, REPLACE_MAP, USE_LAB_TEXT_AS_FILENAME)
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
    with contextlib.closing(wave.open(path, "rb")) as wf:
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
    with contextlib.closing(wave.open(path, "rb")) as wf:
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
    with contextlib.closing(wave.open(output_path, "wb")) as wf:
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
        results = split_batch(pairs, outdir, lang="zh", logger_arg=logger, min_duration=3.0)
        return results
    except Exception as e:
        print(f"[WARN] Batch split function failed: {e}", file=sys.stderr)
        # 返回所有失败项
        return {a: None for (a, _) in pairs}

class DatasetProcessor:
    """封装整个数据集处理流程的类"""

    def __init__(self, root_dir, replace_map, use_lab_text_as_filename=True):
        """
        初始化处理器。
        :param root_dir: 输入数据集的根目录。
        :param replace_map: 标注文本的替换规则字典。
        :param use_lab_text_as_filename: 是否使用标注文本作为文件名。
        """
        self.root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"指定的根目录不存在或不是目录: {self.root_dir}")

        self.use_lab_text_as_filename = use_lab_text_as_filename
        self.replace_map = replace_map
        self.folder_name = os.path.basename(self.root_dir.rstrip(os.sep))
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

    def _setup_paths_and_logging(self):
        """设置所有输出路径并配置日志记录。"""
        self.output_dir = os.path.join(os.path.dirname(self.root_dir), "output", self.folder_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] 所有输出将被保存到: {self.output_dir}")

        # 为中间文件（切分、合并）创建临时目录
        self.temp_dir = os.path.join(self.output_dir, "_temp_processing")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        self.auto_merge_dir = self.temp_dir
        self.auto_split_dir = self.temp_dir

        # 为人工审阅切分文件创建目录
        self.review_dir = os.path.join(self.output_dir, "待审阅_切分文件")
        if os.path.exists(self.review_dir):
            shutil.rmtree(self.review_dir)
        os.makedirs(self.review_dir)

        # 日志文件名前加!置顶，.list文件输出到父级output目录
        self.log_path = os.path.join(self.output_dir, f"！{self.folder_name}.log")
        parent_output_dir = os.path.dirname(self.output_dir)
        self.out_list_path = os.path.join(parent_output_dir, f"{self.folder_name}.list")

        self.logger = logging.getLogger("tts_preprocess")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
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
        self.logger.info(f"  - 替换规则: {self.replace_map}")
        self.logger.info("="*50)

    def _find_audio_entries(self):
        """查找、配对并读取所有音频条目及其元数据。"""
        entries = os.listdir(self.root_dir)
        file_entries = [f for f in entries if os.path.isfile(os.path.join(self.root_dir, f))]

        stems = {}
        for fname in file_entries:
            name, ext = os.path.splitext(fname)
            ext_lower = ext.lower()
            if ext_lower in (".wav", ".lab"):
                stems.setdefault(name, {})[ext_lower] = fname

        self.logger.info(f"在根目录中找到 {len(entries)} 个文件/文件夹。")
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
            wav_path = os.path.join(self.root_dir, extmap[".wav"])
            lab_path = os.path.join(self.root_dir, extmap[".lab"])

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
                "lab_text": lab_text, "duration": duration
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

            pairs = [(m["wav_path"], m["lab_path"]) for m in long_list_to_split]
            batch_results = call_split_script_batch(pairs, self.auto_split_dir, logger=self.logger)

            split_failures = []
            for m in long_list_to_split:
                wavp = m["wav_path"]
                split_wav_paths = batch_results.get(wavp)

                if split_wav_paths:
                    try:
                        if not all(os.path.isfile(p) for p in split_wav_paths):
                            raise FileNotFoundError("部分切分后的wav文件未找到。")

                        for split_wav_path in split_wav_paths:
                            split_lab_path = os.path.splitext(split_wav_path)[0] + ".lab"
                            if not os.path.isfile(split_lab_path):
                                raise FileNotFoundError(f"未找到切分后的 lab 文件: {split_lab_path}")
                            
                            # 移动到审阅目录
                            shutil.move(split_wav_path, os.path.join(self.review_dir, os.path.basename(split_wav_path)))
                            shutil.move(split_lab_path, os.path.join(self.review_dir, os.path.basename(split_lab_path)))

                        self.logger.info(f"切分成功并将文件移至待审阅文件夹: {wavp} -> {', '.join(os.path.basename(p) for p in split_wav_paths)}")
                    except Exception as e:
                        split_failures.append(os.path.basename(wavp))
                        self.logger.warning(f"处理 {wavp} 的切分结果失败 (移动文件时出错): {e}")
                        unsplit_entries.append({"wav_abs": os.path.abspath(wavp), "lab_text": m["lab_text"], "duration": m["duration"]})
                else:
                    split_failures.append(os.path.basename(wavp))
                    self.logger.warning(f"{wavp} 切分失败或输出缺失。")
                    unsplit_entries.append({"wav_abs": os.path.abspath(wavp), "lab_text": m["lab_text"], "duration": m["duration"]})
            
            if split_failures:
                print(f"[WARN] {len(split_failures)} 个长音频文件自动切分失败，已按原样添加。请考虑手动切分。详情请查看日志。", file=sys.stderr)

        return unsplit_entries

    def _process_short_files(self, short_list):
        """处理短音频，进行自动合并。"""
        if not short_list:
            return [], []

        groups = []
        s = short_list[:]
        while len(s) >= 3:
            groups.append([s.pop(0), s.pop(0), s.pop(0)])
        if len(s) == 2:
            groups.append([s.pop(0), s.pop(0)])
        
        unmerged_entries = []
        if len(s) == 1:
            last = groups.pop() if groups else None
            if last and len(last) == 3:
                a, b, c = last
                d = s.pop(0)
                groups.append([a, b])
                groups.append([c, d])
            else:
                unmerged_entries.append({"wav_abs": os.path.abspath(s[0]["wav_path"]), "lab_text": s[0]["lab_text"], "duration": s[0]["duration"]})

        if groups:
            print(f"[INFO] 准备将 {len(short_list)} 个短音频文件合并成 {len(groups)} 组。")

        processed_entries = []
        merge_map_lines = []
        merge_read_failures, merge_failures = 0, 0

        for group_idx, group in enumerate(groups, start=1):
            wav_infos, ok = [], True
            for item in group:
                try:
                    params, frames = read_wav_params_and_frames(item["wav_path"])
                    wav_infos.append({"params": params, "frames": frames, "path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"]})
                except Exception as e:
                    self.logger.warning(f"读取用于合并的 wav 文件失败 {item['wav_path']}: {e}")
                    merge_read_failures += 1
                    ok = False
                    break
            
            if not ok:
                merge_failures += 1
                self.logger.warning("回退: 因读取错误，将该组的原始文件单独添加到列表。")
                processed_entries.extend([{"wav_abs": os.path.abspath(item["wav_path"]), "lab_text": item["lab_text"], "duration": item["duration"]} for item in group])
                continue

            merge_name = f"{self.folder_name}_merge_{group_idx}_{format_hms_filename(sum(i['duration'] for i in wav_infos))}.wav"
            merge_path = os.path.join(self.auto_merge_dir, merge_name)
            try:
                merge_wavs(merge_path, wav_infos, silence_duration=0.5)
                merge_lab = "".join(i["lab_text"] for i in wav_infos)
                processed_entries.append({"wav_abs": os.path.abspath(merge_path), "lab_text": merge_lab, "duration": sum(i["duration"] for i in wav_infos)})
                merge_map_lines.append(f"{merge_path} <- " + ", ".join(i["path"] for i in wav_infos))
                self.logger.info(f"已将 {len(wav_infos)} 个文件合并到 {merge_path}")
            except Exception as e:
                merge_failures += 1
                self.logger.warning(f"第 {group_idx} 组合并失败: {e}")
                self.logger.warning("回退: 因合并失败，将该组的原始文件单独添加到列表。")
                processed_entries.extend([{"wav_abs": os.path.abspath(item["wav_path"]), "lab_text": item["lab_text"], "duration": item["duration"]} for item in group])

        if merge_read_failures > 0:
            print(f"[WARN] {merge_read_failures} 个短音频文件读取失败，无法合并。详情请查看日志。", file=sys.stderr)
        if merge_failures > 0:
            print(f"[WARN] {merge_failures} 组短音频文件合并失败，已将它们单独添加。详情请查看日志。", file=sys.stderr)

        processed_entries.extend(unmerged_entries)
        return processed_entries, merge_map_lines

    def _wait_for_review_and_process(self):
        """等待用户审阅切分文件，然后处理它们。"""
        if not os.listdir(self.review_dir):
            self.logger.info("待审阅文件夹为空，跳过人工审阅步骤。")
            return []

        print("="*50)
        print(f"[INFO] {len(os.listdir(self.review_dir)) // 2} 对切分文件已生成并等待审阅。")
        input(f"\n请检查文件夹 '{self.review_dir}' 中的文件。\n"
              f"您可以删除不需要的文件对，或修改 .lab 文件的内容。\n"
              f"审阅完成后，请按 Enter 键继续处理...")
        print("="*50)
        print("[INFO] 审阅完成，正在处理审阅后的文件...")
        self.logger.info("用户审阅完成，开始处理审阅后的文件。")

        # 重新扫描审阅文件夹
        lab_files = sorted([f for f in os.listdir(self.review_dir) if f.endswith(".lab")])
        
        reviewed_entries = []
        for lab_filename in lab_files:
            stem = os.path.splitext(lab_filename)[0]
            wav_filename = stem + ".wav"
            wav_path = os.path.join(self.review_dir, wav_filename)
            lab_path = os.path.join(self.review_dir, lab_filename)

            if not os.path.exists(wav_path):
                self.logger.warning(f"审阅后发现 {lab_filename} 但缺少对应的 .wav 文件，已跳过。")
                continue

            try:
                lab_text = read_lab_utf8(lab_path).strip()
                # 再次应用替换规则，以防用户在审阅时添加了需要替换的字符
                for k, v in self.replace_map.items():
                    lab_text = lab_text.replace(k, v)
                duration = get_wav_duration_seconds(wav_path)
                
                reviewed_entries.append({
                    "wav_abs": os.path.abspath(wav_path),
                    "lab_text": lab_text,
                    "duration": duration
                })
                self.logger.info(f"已处理审阅后的文件: {wav_filename}")
            except Exception as e:
                self.logger.error(f"处理审阅后的文件 {wav_path} 失败: {e}")
        return reviewed_entries

    def _finalize_output(self, final_entries, merge_map_lines, initial_count):
        """最终确定输出文件：重命名、复制并写入所有摘要和列表文件。"""
        print("="*50)
        print("[INFO] 正在最终确定输出文件：重命名并复制到输出目录的根目录...")
        self.logger.info("正在最终确定输出文件：重命名并复制到输出目录的根目录。")

        final_output_entries = []
        processed_filenames = set()
        merge_counter = 1

        for entry in final_entries:
            source_path = entry["wav_abs"]
            lab_text = entry["lab_text"]

            if self.use_lab_text_as_filename:
                target_basename = sanitize_filename(lab_text) + ".wav"
            else:
                source_filename = os.path.basename(source_path)
                if source_path.startswith(os.path.abspath(self.auto_merge_dir)):
                    # 已合并的音频
                    target_basename = f"{self.folder_name}_auto_merge_{merge_counter}.wav"
                    merge_counter += 1
                elif source_path.startswith(os.path.abspath(self.review_dir)):
                    # 已切分的音频
                    target_basename = source_filename
                elif source_path.startswith(os.path.abspath(self.root_dir)):
                    # 正常长度或未处理的音频
                    target_basename = source_filename
                else:
                    self.logger.warning(f"未知文件来源: {source_path}。将使用其当前文件名。")
                    target_basename = source_filename

            temp_basename, counter = target_basename, 1
            while temp_basename in processed_filenames:
                base, ext = os.path.splitext(target_basename)
                temp_basename = f"{base}_{counter}{ext}"
                counter += 1
            target_basename = temp_basename
            processed_filenames.add(target_basename)

            target_path = os.path.join(self.output_dir, target_basename)
            try:
                # 中间文件（来自切分或合并）位于临时目录中，应移动它们。
                # 原始文件（正常长度）应被复制，以保持源数据集不变。
                if source_path.startswith(os.path.abspath(self.temp_dir)) or \
                   source_path.startswith(os.path.abspath(self.review_dir)):
                    shutil.move(source_path, target_path)
                else:
                    shutil.copy(source_path, target_path)
                final_output_entries.append({
                    "wav_abs": os.path.abspath(target_path),
                    "lab_text": lab_text,
                    "duration": entry["duration"]
                })
            except Exception as e:
                print(f"[ERROR] 复制或移动 {source_path} 到 {target_path} 失败: {e}", file=sys.stderr)
                self.logger.error(f"复制或移动 {source_path} 到 {target_path} 失败: {e}")

        if merge_map_lines:
            merge_map_path = os.path.join(self.output_dir, "！合并记录.txt")
            try:
                with open(merge_map_path, "a", encoding="utf-8") as mf:
                    mf.write("\n".join(merge_map_lines) + "\n")
            except Exception as e:
                self.logger.warning(f"写入合并记录失败: {e}")

        # 移动并附加切分脚本生成的日志文件
        split_log_filename = "切分记录.txt"
        temp_split_log_path = os.path.join(self.temp_dir, split_log_filename)
        if os.path.exists(temp_split_log_path):
            final_split_log_path = os.path.join(self.output_dir, "！" + split_log_filename)
            try:
                with open(temp_split_log_path, "r", encoding="utf-8") as temp_f:
                    content = temp_f.read()
                with open(final_split_log_path, "a", encoding="utf-8") as final_f:
                    final_f.write(content)
                os.remove(temp_split_log_path)
                self.logger.info(f"已将切分记录附加到: {final_split_log_path}")
            except Exception as e:
                self.logger.warning(f"处理切分记录文件 {temp_split_log_path} 失败: {e}")

        total_duration_seconds = sum(e['duration'] for e in final_output_entries)
        total_duration_hms = format_hms(total_duration_seconds)
        summary_text = (
            f"处理摘要:\n"
            f"  - 初始找到的音频文件: {initial_count}\n"
            f"  - 最终创建的输出文件: {len(final_output_entries)}\n"
            f"  - 最终音频总时长: {total_duration_hms} ({total_duration_seconds:.2f} 秒)"
        )
        print("="*50)
        print(summary_text)
        print("="*50)
        self.logger.info("="*50)
        self.logger.info("处理摘要:")
        self.logger.info(f"  - 初始找到的音频文件: {initial_count}")
        self.logger.info(f"  - 最终创建的输出文件: {len(final_output_entries)}")
        self.logger.info(f"  - 最终音频总时长: {total_duration_hms} ({total_duration_seconds:.2f} 秒)")
        self.logger.info("="*50)

        duration_filename = f"!{format_hms_filename(total_duration_seconds)}.txt"
        duration_filepath = os.path.join(self.output_dir, duration_filename)
        try:
            with open(duration_filepath, "w") as f: pass
            print(f"[INFO] 已创建时长文件: {duration_filepath}")
            self.logger.info(f"已创建时长文件: {duration_filepath}")
        except Exception as e:
            print(f"[WARN] 创建时长文件失败: {e}", file=sys.stderr)
            self.logger.warning(f"创建时长文件 {duration_filepath} 失败: {e}")

        try:
            with open(self.out_list_path, "w", encoding="utf-8") as ol:
                for ent in final_output_entries:
                    ol.write(f"{ent['wav_abs']}|{self.folder_name}|ZH|{ent['lab_text']}\n")
            self.logger.info(f"已将列表写入 {self.out_list_path} (条目数={len(final_output_entries)})")
        except Exception as e:
            print(f"[ERROR] 写入列表文件失败: {e}", file=sys.stderr)
            self.logger.error(f"写入列表 {self.out_list_path} 失败: {e}")

    def run(self):
        """执行完整的数据集处理流程。"""
        print(f"[INFO] 开始处理数据集: {self.root_dir}")
        self._setup_paths_and_logging()

        # 1. 查找并解析所有音频文件
        all_entries = self._find_audio_entries()

        # 2. 按时长分类
        short_list = [m for m in all_entries if m["duration"] <= 3.0]
        normal_list = [m for m in all_entries if 3.0 < m["duration"] < 25.0]
        long_list = [m for m in all_entries if m["duration"] >= 25.0]
        self.logger.info(f"音频文件分类: {len(short_list)} 短 (<=3s), {len(normal_list)} 正常 (3-25s), {len(long_list)} 长 (>=25s)。")

        # 3. 按顺序处理并收集最终条目
        final_entries_before_copy = []

        # 3.1 添加正常长度的音频
        for m in normal_list:
            final_entries_before_copy.append({"wav_abs": os.path.abspath(m["wav_path"]), "lab_text": m["lab_text"], "duration": m["duration"]})
        print(f"[INFO] 已添加 {len(normal_list)} 个正常长度的音频文件。")
        self.logger.info(f"已添加 {len(normal_list)} 个正常长度的条目。")

        # 3.2 处理长音频
        unsplit_long_entries = self._process_long_files(long_list)
        final_entries_before_copy.extend(unsplit_long_entries)

        # 3.3 处理短音频
        processed_short_entries, merge_map_lines = self._process_short_files(short_list)
        final_entries_before_copy.extend(processed_short_entries)

        # 4. 等待人工审阅切分文件并处理
        reviewed_entries = self._wait_for_review_and_process()
        final_entries_before_copy.extend(reviewed_entries)

        # 5. 最终确定输出
        self._finalize_output(final_entries_before_copy, merge_map_lines, len(all_entries))

        # 6. 清理临时文件
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理处理过程中生成的临时目录。"""
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"已成功清理临时目录: {self.temp_dir}")
                print(f"[INFO] 已清理合并/切分临时工作目录。")
            except Exception as e:
                self.logger.error(f"清理临时目录 {self.temp_dir} 失败: {e}")
                print(f"[ERROR] 清理临时目录 {self.temp_dir} 失败: {e}", file=sys.stderr)
        if self.review_dir and os.path.isdir(self.review_dir):
            try:
                shutil.rmtree(self.review_dir)
                self.logger.info(f"已成功清理审阅目录: {self.review_dir}")
                print(f"[INFO] 已清理审阅目录。")
            except Exception as e:
                self.logger.error(f"清理审阅目录 {self.review_dir} 失败: {e}")
                print(f"[ERROR] 清理审阅目录 {self.review_dir} 失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()