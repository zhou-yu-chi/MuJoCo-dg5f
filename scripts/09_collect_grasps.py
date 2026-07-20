#!/usr/bin/env python3
"""
09_collect_grasps.py
---------------------
Stage-3 批次收集：多進程重複呼叫 08_try_grasp.py 的 run()，一直跑到你按
Ctrl+C，或是達到 --duration / --attempts / --target-successes 任一停止條件
為止。每一次嘗試（不管成功或失敗）都會寫成一行 JSON，append 進該 worker
自己的 log 檔——之後要拿去訓練，直接濾出 "success": true 的那些行即可，
裡面已經帶了 doc/plan.md 階段 3 要求的 object_pose／palm_pose／joint_angles。

每個 worker 各寫各的檔案（不共用同一個檔案），避免多個 process 同時寫同一
支檔案互相打架；一個 CPU core 對應一個 MuJoCo instance（plan.md 階段 3 的
建議做法），彼此之間除了讀同一份場景 XML 之外完全獨立。

Usage:
    python 09_collect_grasps.py --assets ../assets --object 002_master_chef_can
    python 09_collect_grasps.py --assets ../assets --object 025_mug \
        --workers 6 --duration 3600
    python 09_collect_grasps.py --assets ../assets --object 025_mug \
        --target-successes 2000
"""

import argparse
import contextlib
import importlib.util
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

# 08_try_grasp.py 的檔名以數字開頭，不能用一般的 import 語法，用
# importlib 動態載入，直接沿用它的 run()/build_single_object_scene()，
# 避免兩份程式碼各自維護、彼此漂移。
_TG_PATH = Path(__file__).resolve().parent / "08_try_grasp.py"
_spec = importlib.util.spec_from_file_location("try_grasp", _TG_PATH)
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)


def worker_main(
    worker_id: int,
    scene_path: Path,
    obj_name: str,
    seed_start: int,
    n_workers: int,
    stop_time,
    max_attempts,
    target_successes,
    log_path: Path,
    success_counter,
    attempt_counter,
):
    """單一 worker 的主迴圈：載入一次模型，接下來不斷重跑 try_grasp，直到
    任一停止條件成立。每個 worker 用等差數列 seed_start + worker_id +
    attempt*n_workers 取種子，保證全部 worker 之間不會用到同一個種子。"""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    attempt = 0
    with open(log_path, "a", buffering=1) as f:
        while True:
            if max_attempts is not None and attempt >= max_attempts:
                break
            if stop_time is not None and time.time() >= stop_time:
                break
            if target_successes is not None and success_counter.value >= target_successes:
                break

            seed = seed_start + worker_id + attempt * n_workers
            rng = np.random.default_rng(seed)
            mujoco.mj_resetData(model, data)  # 每次嘗試都要從乾淨的初始狀態開始，
            # 不然上一次嘗試「拿起+推力測試」留下的手掌/物體殘留姿態會污染下一次

            t0 = time.time()
            try:
                # run() 內部一大堆 print() 是給互動除錯用的，批次瘋狂跑的時候
                # 全部塞進 stdout 只會洗版，這裡整段消音，只留下面自己統計的
                # 那一行進度報告。
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    result = tg.run(model, data, obj_name, rng, viewer=None)
            except Exception as exc:  # noqa: BLE001 -- 單一 seed 炸掉不能讓整個 worker 死掉
                result = {
                    "object": obj_name,
                    "success": False,
                    "fail_reason": f"exception: {exc!r}",
                    "sample_p": None,
                    "sample_n": None,
                    "object_pose": None,
                    "palm_pose": None,
                    "joint_angles": None,
                }

            record = {
                "worker": worker_id,
                "seed": int(seed),
                "attempt": attempt,
                "duration_s": round(time.time() - t0, 3),
                "timestamp": time.time(),
                **result,
            }
            f.write(json.dumps(record) + "\n")

            with attempt_counter.get_lock():
                attempt_counter.value += 1
            if result["success"]:
                with success_counter.get_lock():
                    success_counter.value += 1

            attempt += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--object", required=True)
    ap.add_argument("--workers", type=int, default=None,
                     help="平行跑幾個 MuJoCo instance，預設 CPU 核心數 - 1")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--duration", type=float, default=None,
                     help="跑幾秒後停止（不給就跟 --attempts/--target-successes 一樣，沒給任何一個就跑到 Ctrl+C 為止）")
    ap.add_argument("--attempts", type=int, default=None,
                     help="全部 worker 加起來總共跑幾次嘗試後停止")
    ap.add_argument("--target-successes", type=int, default=None,
                     help="全部 worker 加起來收集到幾筆成功抓取後停止")
    ap.add_argument("--log-dir", type=Path, default=None,
                     help="log 檔輸出目錄，預設 <repo>/logs")
    args = ap.parse_args()

    assets_dir = args.assets.resolve()
    # 場景 XML 只在主行程建一次，所有 worker 共用同一份檔案，避免多個
    # process 同時搶著寫同一個檔名互相干擾（見 08_try_grasp.py 的
    # build_single_object_scene）。
    scene_path = tg.build_single_object_scene(assets_dir, args.object)

    log_dir = args.log_dir or (Path(__file__).resolve().parents[1] / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    n_workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    stop_time = time.time() + args.duration if args.duration else None
    attempts_per_worker = (args.attempts // n_workers) if args.attempts else None

    ctx = mp.get_context("spawn")  # 不要 fork：避免子行程繼承到父行程裡半初始化的原生函式庫狀態
    success_counter = ctx.Value("i", 0)
    attempt_counter = ctx.Value("i", 0)

    duration_str = f"{args.duration}s" if args.duration else "None"
    print(f"開始收集：object={args.object}  workers={n_workers}  "
          f"log_dir={log_dir}  停止條件: "
          f"duration={duration_str} attempts={args.attempts} target_successes={args.target_successes}"
          f"{'（都沒設，會一直跑到 Ctrl+C）' if not (args.duration or args.attempts or args.target_successes) else ''}")

    procs = []
    for w in range(n_workers):
        log_path = log_dir / f"{args.object}_{run_tag}_worker{w}.jsonl"
        p = ctx.Process(
            target=worker_main,
            args=(w, scene_path, args.object, args.seed_start, n_workers,
                  stop_time, attempts_per_worker, args.target_successes,
                  log_path, success_counter, attempt_counter),
            daemon=True,
        )
        p.start()
        procs.append(p)

    t_start = time.time()
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(5)
            n_att = attempt_counter.value
            n_succ = success_counter.value
            elapsed = time.time() - t_start
            rate = n_att / elapsed if elapsed > 0 else 0.0
            pct = 100 * n_succ / n_att if n_att else 0.0
            print(f"[{elapsed / 60:6.1f} min] attempts={n_att:6d}  successes={n_succ:5d} "
                  f"({pct:5.1f}%)  {rate:5.2f} attempts/s")
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在通知所有 worker 停止（目前這一次嘗試跑完就會停）...")
        for p in procs:
            p.terminate()

    for p in procs:
        p.join()

    n_att = attempt_counter.value
    n_succ = success_counter.value
    print(f"\n收集結束：總共 {n_att} 次嘗試，{n_succ} 次成功（{100 * n_succ / max(n_att, 1):.1f}%）")
    print(f"log 檔在 {log_dir}/{args.object}_{run_tag}_worker*.jsonl，"
          f"每一行是一次嘗試，success=true 的那些行已經帶好 object_pose/palm_pose/joint_angles，"
          f"可以直接濾出來當訓練資料。")


if __name__ == "__main__":
    main()
