#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU producer + CPU consumers + 主进程写 CSV 的完整管线版（xprv-only）：

- 主进程：
    - 从文件按批读取助记词 -> 投递到 mnem_queue
    - 从 result_queue 收结果 -> 写 CSV，并输出吞吐日志

- GPU 进程 (producer)：
    - 从 mnem_queue 取 (batch_idx, mnems)
    - 调用 pybind_mnemonic2xroot.derive_bip32_root_I_raw -> buf (n*64 bytes)
    - 把 (batch_idx, mnems, buf) 放入 cpu_queue
    - 完成后向 cpu_queue 投递 N 个 None 作为结束信号

- N 个 CPU 进程 (consumers)：
    - 从 cpu_queue 取 (batch_idx, mnems, buf)
    - 调用 pybind_xroot2xkey.xroots_to_xprv -> [xprv, ...]
    - 组装 rows = [(mnemonic, xprv), ...]
    - 把 (batch_idx, rows, count) 放入 result_queue

这样 GPU 和 CPU 可以真正并行工作，整体吞吐接近 max(GPU, CPU)。
"""

import sys
import csv
import time
import logging
from typing import List, Tuple

import multiprocessing as mp


# ---------------------- 读文件: 按批 yield 助记词 ----------------------


def iter_mnemonic_batches(path: str, batch_size: int):
    """
    流式读取文件，每次 yield (batch_idx, [mnemonic, ...])
    - 忽略空行和以 # 开头的行
    """
    batch: List[str] = []
    batch_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch_idx, batch
                batch_idx += 1
                batch = []
        if batch:
            yield batch_idx, batch


# ---------------------- GPU 进程：mnems -> I(buf) ----------------------


def gpu_worker(
    mnem_queue: mp.Queue,
    cpu_queue: mp.Queue,
    passphrase: str,
    threads_per_block: int,
    num_cpu_workers: int,
):
    """
    专门做 GPU PBKDF2+HMAC("Bitcoin seed") 的进程：
      - 从 mnem_queue 取 (batch_idx, mnems)
      - 调用 pybind_mnemonic2xroot.derive_bip32_root_I_raw
      - 把 (batch_idx, mnems, buf) 放入 cpu_queue
      - 输入结束时，从 mnem_queue 收到 None，之后向 cpu_queue 投递 num_cpu_workers 个 None
    """
    import pybind_mnemonic2xroot as m2r  # 只在子进程 import

    while True:
        item = mnem_queue.get()
        if item is None:
            # 所有输入结束，通知 CPU worker 停止
            for _ in range(num_cpu_workers):
                cpu_queue.put(None)
            break

        batch_idx, mnems = item
        t0 = time.time()

        buf: bytes = m2r.derive_bip32_root_I_raw(
            mnems,
            passphrase,
            threads_per_block,
        )
        expected_len = len(mnems) * 64
        if len(buf) != expected_len:
            raise RuntimeError(
                f"[GPU] batch {batch_idx}: derive_bip32_root_I_raw 返回 {len(buf)} != {expected_len}"
            )

        dt = time.time() - t0
        logging.debug(
            "[GPU] batch %d: %d mnemonics in %.2fs (%.0f mnems/s)",
            batch_idx,
            len(mnems),
            dt,
            len(mnems) / dt if dt > 0 else 0.0,
        )

        # 丢给 CPU 队列处理 xroot -> xprv
        cpu_queue.put((batch_idx, mnems, buf))


# ---------------------- CPU 进程：buf -> xprv ----------------------


def cpu_worker(
    cpu_queue: mp.Queue,
    result_queue: mp.Queue,
    version_xprv: int,
    depth: int,
    child_num: int,
):
    """
    纯 CPU 消费者：
      - 从 cpu_queue 取 (batch_idx, mnems, buf)
      - 调用 xroots_to_xprv -> ["xprv", ...]
      - 组装 rows = [(mnemonic, xprv), ...]
      - 放入 result_queue: (batch_idx, rows, count)
    """
    import pybind_xroot2xkey as x2k  # 只在子进程 import

    while True:
        item = cpu_queue.get()
        if item is None:
            break

        batch_idx, mnems, buf = item
        t0 = time.time()

        xprvs: List[str] = x2k.xroots_to_xprv(
            buf,
            version_xprv,
            depth,
            child_num,
        )

        if len(xprvs) != len(mnems):
            raise RuntimeError(
                f"[CPU] batch {batch_idx}: xroots_to_xprv 返回 {len(xprvs)} 条，与输入 {len(mnems)} 不一致"
            )

        rows: List[Tuple[str, str]] = list(zip(mnems, xprvs))

        dt = time.time() - t0
        logging.debug(
            "[CPU] batch %d: %d mnemonics in %.2fs (%.0f mnems/s)",
            batch_idx,
            len(mnems),
            dt,
            len(mnems) / dt if dt > 0 else 0.0,
        )

        result_queue.put((batch_idx, rows, len(rows)))


# ---------------------- 主流程：文件 -> 队列 -> CSV ----------------------


def process_file_pipeline(
    in_path: str,
    out_path: str,
    passphrase: str = "",
    batch_size: int = 400_000,
    threads_per_block: int = 128,
    num_cpu_workers: int = 2,
    max_inflight_batches: int = 4,
    version_xprv: int = 0x0488ADE4,
    depth: int = 0,
    child_num: int = 0,
):
    """
    完整流水线：
      - in_path:    输入助记词文件，每行一条
      - out_path:   输出 CSV (mnemonic,xprv)
      - passphrase: BIP39 passphrase（一般为空字符串）
      - batch_size: 每批助记词数量
      - threads_per_block: 传给 CUDA 的 TPB
      - num_cpu_workers: CPU 进程数
      - max_inflight_batches: 流水线中最多挂起多少个批次，控制内存
      - version_xprv / depth / child_num: BIP32 xprv 参数
    """

    logging.info(
        "开始处理: in=%s out=%s passphrase='%s' batch_size=%d TPB=%d cpu_workers=%d inflight=%d",
        in_path,
        out_path,
        passphrase,
        batch_size,
        threads_per_block,
        num_cpu_workers,
        max_inflight_batches,
    )

    ctx = mp.get_context("spawn")  # 推荐 spawn，避免 GPU 上下文继承问题
    mnem_queue: mp.Queue = ctx.Queue(maxsize=max_inflight_batches * 2)
    cpu_queue: mp.Queue = ctx.Queue(maxsize=max_inflight_batches * 2)
    result_queue: mp.Queue = ctx.Queue(maxsize=max_inflight_batches * 2)

    # 启动 GPU 进程
    gpu_proc = ctx.Process(
        target=gpu_worker,
        args=(mnem_queue, cpu_queue, passphrase, threads_per_block, num_cpu_workers),
        name="GPU-Worker",
    )
    gpu_proc.start()

    # 启动 CPU 进程们
    cpu_procs: List[mp.Process] = []
    for i in range(num_cpu_workers):
        p = ctx.Process(
            target=cpu_worker,
            args=(cpu_queue, result_queue, version_xprv, depth, child_num),
            name=f"CPU-Worker-{i}",
        )
        p.start()
        cpu_procs.append(p)

    total_processed = 0
    submitted_batches = 0
    finished_batches = 0
    t_start = time.time()

    # 记录每个 batch 提交时间，用于计算 end-to-end latency
    submit_time: dict[int, float] = {}

    # 主进程打开输出 CSV，一边投递、一边收结果、一边写
    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["mnemonic", "xprv"])

        # 1) 一边读文件一边投递到 GPU 队列
        for batch_idx, mnems in iter_mnemonic_batches(in_path, batch_size):
            # backpressure：如果挂起批次过多，先取一个结果写掉
            while submitted_batches - finished_batches >= max_inflight_batches:
                b_idx, rows, count = result_queue.get()
                now = time.time()
                finished_batches += 1
                total_processed += count

                dt_batch = now - submit_time.pop(b_idx, t_start)
                dt_total = now - t_start
                speed_batch = count / dt_batch if dt_batch > 0 else 0.0
                speed_total = total_processed / dt_total if dt_total > 0 else 0.0

                for row in rows:
                    writer.writerow(row)

                logging.info(
                    "[batch %d] %d mnemonics in %.2fs (%.0f mnems/s), total=%d (%.0f mnems/s)",
                    b_idx,
                    count,
                    dt_batch,
                    speed_batch,
                    total_processed,
                    speed_total,
                )

            # 正常投递新批次到 GPU
            submit_time[batch_idx] = time.time()
            mnem_queue.put((batch_idx, mnems))
            submitted_batches += 1

        # 所有输入读完，告诉 GPU worker：没有新数据了
        mnem_queue.put(None)

        # 2) 把剩余的结果都收完
        while finished_batches < submitted_batches:
            b_idx, rows, count = result_queue.get()
            now = time.time()
            finished_batches += 1
            total_processed += count

            dt_batch = now - submit_time.pop(b_idx, t_start)
            dt_total = now - t_start
            speed_batch = count / dt_batch if dt_batch > 0 else 0.0
            speed_total = total_processed / dt_total if dt_total > 0 else 0.0

            for row in rows:
                writer.writerow(row)

            logging.info(
                "[batch %d] %d mnemonics in %.2fs (%.0f mnems/s), total=%d (%.0f mnems/s)",
                b_idx,
                count,
                dt_batch,
                speed_batch,
                total_processed,
                speed_total,
            )

    # 等 GPU / CPU 子进程退出
    gpu_proc.join()
    for p in cpu_procs:
        p.join()

    elapsed = time.time() - t_start
    avg_speed = total_processed / elapsed if elapsed > 0 else 0.0
    logging.info(
        "全部完成: total=%d, elapsed=%.2fs, avg_speed=%.0f mnems/s",
        total_processed,
        elapsed,
        avg_speed,
    )


# ---------------------- CLI 入口 ----------------------


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        print(
            f"用法: {argv[0]} IN_MNEMONICS_TXT OUT_CSV [PASSPHRASE] "
            "[CPU_WORKERS] [BATCH_SIZE] [TPB] [INFLIGHT]\n\n"
            "  - IN_MNEMONICS_TXT : 每行一个助记词的文本文件\n"
            "  - OUT_CSV          : 输出 CSV 路径\n"
            "  - PASSPHRASE       : (可选) BIP39 passphrase，默认空字符串\n"
            "  - CPU_WORKERS      : (可选) CPU 进程数，默认 2\n"
            "  - BATCH_SIZE       : (可选) 每批助记词数量，默认 400000\n"
            "  - TPB              : (可选) CUDA threads_per_block，默认 128\n"
            "  - INFLIGHT         : (可选) 流水线挂起批次数上限，默认 4\n\n"
            "示例:\n"
            f"  OMP_NUM_THREADS=8 {argv[0]} mnemo_256_100000000.txt xroot_keys.csv '' 2 600000 128 4\n"
        )
        return

    in_path = argv[1]
    out_path = argv[2]
    passphrase = argv[3] if len(argv) > 3 else ""
    cpu_workers = int(argv[4]) if len(argv) > 4 else 2
    batch_size = int(argv[5]) if len(argv) > 5 else 400_000
    tpb = int(argv[6]) if len(argv) > 6 else 128
    inflight = int(argv[7]) if len(argv) > 7 else 4

    process_file_pipeline(
        in_path=in_path,
        out_path=out_path,
        passphrase=passphrase,
        batch_size=batch_size,
        threads_per_block=tpb,
        num_cpu_workers=cpu_workers,
        max_inflight_batches=inflight,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main(sys.argv)