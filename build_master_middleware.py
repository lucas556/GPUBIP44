#!/usr/bin/env python3
import argparse
import struct
import sys
import time
from typing import List

import pybind_mnemonic2master as m2m  # 你已经编译好的 GPU 模块


MAGIC = b"MSTRI\x00\x00\x01"  # 8 bytes: 简单的格式标识 + 版本号 v1


def count_mnemonics(path: str) -> int:
    """
    第一遍扫描文件，只统计有效助记词行数量：
    - 每行一条助记词
    - 跳过空行、以 # 开头的行
    """
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            cnt += 1
    return cnt


def format_sec(sec: float) -> str:
    """把秒格式化成更友好的字符串。"""
    if sec < 1:
        return f"{sec * 1000:.0f} ms"
    if sec < 60:
        return f"{sec:.1f} s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m} min {s} s"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


def process_in_batches(
    mnemonic_file: str,
    out_path: str,
    batch_size: int,
    threads_per_block: int,
) -> None:
    """
    第二遍扫描文件，按批处理送给 GPU，写出 master I 中间件文件。
    BIP39 passphrase 固定为 ""。
    """
    total = count_mnemonics(mnemonic_file)
    if total == 0:
        print("没有有效助记词，退出。", file=sys.stderr)
        return

    print(f"总助记词数量: {total}", file=sys.stderr)
    print(f"输出文件: {out_path}", file=sys.stderr)
    print(f"批大小: {batch_size}, threads_per_block: {threads_per_block}", file=sys.stderr)
    print(f"BIP39 passphrase 固定为: 空字符串 \"\"", file=sys.stderr)

    start_time = time.time()
    last_log_time = start_time

    with open(out_path, "wb") as fout:
        # 写 header: magic + count(u32 little-endian)
        fout.write(MAGIC)
        fout.write(struct.pack("<I", total))

        processed = 0
        batch: List[str] = []

        with open(mnemonic_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue

                batch.append(s)
                if len(batch) >= batch_size:
                    batch_start = time.time()

                    # GPU 计算：passphrase 固定为 ""
                    buf = m2m.derive_master_I_raw(
                        batch,
                        "",  # passphrase 固定为空
                        threads_per_block,
                    )
                    fout.write(buf)

                    batch_end = time.time()
                    batch_elapsed = batch_end - batch_start
                    processed += len(batch)
                    batch_count = len(batch)
                    batch.clear()

                    total_elapsed = batch_end - start_time
                    overall_speed = processed / total_elapsed if total_elapsed > 0 else 0.0
                    batch_speed = batch_count / batch_elapsed if batch_elapsed > 0 else 0.0

                    remaining = total - processed
                    eta = remaining / overall_speed if overall_speed > 0 else float("inf")

                    print(
                        (
                            f"[{processed:>12d} / {total:<12d}] "
                            f"batch={batch_count}  "
                            f"batch_time={format_sec(batch_elapsed):>8}  "
                            f"batch_speed={batch_speed:>10.0f} mnemo/s  "
                            f"overall_speed={overall_speed:>10.0f} mnemo/s  "
                            f"ETA={format_sec(eta):>10}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

            # 处理最后一个不足 batch 的尾巴
            if batch:
                batch_start = time.time()

                buf = m2m.derive_master_I_raw(
                    batch,
                    "",
                    threads_per_block,
                )
                fout.write(buf)

                batch_end = time.time()
                batch_elapsed = batch_end - batch_start
                batch_count = len(batch)
                processed += len(batch)
                batch.clear()

                total_elapsed = batch_end - start_time
                overall_speed = processed / total_elapsed if total_elapsed > 0 else 0.0
                batch_speed = batch_count / batch_elapsed if batch_elapsed > 0 else 0.0

                remaining = total - processed
                eta = remaining / overall_speed if overall_speed > 0 else 0.0

                print(
                    (
                        f"[{processed:>12d} / {total:<12d}] "
                        f"batch={batch_count}  "
                        f"batch_time={format_sec(batch_elapsed):>8}  "
                        f"batch_speed={batch_speed:>10.0f} mnemo/s  "
                        f"overall_speed={overall_speed:>10.0f} mnemo/s  "
                        f"ETA={format_sec(eta):>10}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    total_time = time.time() - start_time
    final_speed = total / total_time if total_time > 0 else 0.0
    print(
        f"写入完成，总耗时 {format_sec(total_time)}，平均速度 {final_speed:.0f} mnemo/s",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "从助记词文本生成 master I 中间件文件 (GPU BIP39+BIP32 master IL||IR).\n"
            "注意：BIP39 passphrase 固定为 \"\"，不再提供命令行参数。"
        )
    )
    parser.add_argument("mnemonic_file", help="助记词列表文件（每行一条助记词）")
    parser.add_argument("output_file", help="输出的中间件文件路径，例如 master_i.bin")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500000,
        help="批处理大小，默认 500000（按实际显存调整）",
    )
    parser.add_argument(
        "--threads-per-block",
        type=int,
        default=128,
        help="CUDA threads per block，默认 128",
    )

    args = parser.parse_args()

    process_in_batches(
        mnemonic_file=args.mnemonic_file,
        out_path=args.output_file,
        batch_size=args.batch_size,
        threads_per_block=args.threads_per_block,
    )


if __name__ == "__main__":
    main()
