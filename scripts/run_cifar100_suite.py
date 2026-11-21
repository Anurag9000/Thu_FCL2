#!/usr/bin/env python3
"""Run the registered CIFAR-100 algorithms on one selected CUDA device.

This orchestrates the repository's existing entry point and JSON configs; it
does not rewrite method code or hyperparameters. Run from any directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "hparams" / "cifar100"
ENTRY = ROOT / "system" / "main.py"

# `PreciseFCL` is the actual CLI registry name for the AFFCL server.
CONFIGS = {
    "FedAvg": "FedAvg_cifar100.json",
    "FedALA": "FedALA_cifar100.json",
    "FedAS": "FedAS_cifar100.json",
    "FedDBE": "FedDBE_cifar100.json",
    "FedL2P": "FedL2P_cifar100.json",
    "FedSSI": "FedSSI_cifar100.json",
    "FedSTGM": "FedSTGM_cifar100.json",
    "FedTARGET": "FedTARGET_cifar100.json",
    "FedWeIT": "FedWeIT_cifar100.json",
    "PreciseFCL (AFFCL)": "AFFCL_cifar100.json",
    "ReFedPlus": "ReFedPlus_cifar100.json",
    "FedFCIL": "FedFCIL_cifar100.json",
    "GLFC": "GLFC_cifar100.json",
    "LANDER": "LANDER_cifar100.json",
    "PILORA (internal federated path)": "PILORA_cifar100.json",
}

REGISTERED = {"FedAvg", "FedALA", "FedAS", "FedDBE", "FedL2P", "FedSSI",
              "FedSTGM", "FedTARGET", "FedWeIT", "PreciseFCL", "PILORA",
              "ReFedPlus", "FedFCIL", "GLFC", "LANDER"}

LOCK = threading.Lock()
CHILDREN: set[subprocess.Popen] = set()
INTERRUPTED = False


def config_jobs() -> list[dict]:
    jobs = []
    for label, filename in CONFIGS.items():
        path = CONFIG_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        config = json.loads(path.read_text())
        if config.get("dataset", "").upper() != "CIFAR100":
            raise ValueError(f"{path} is not configured for CIFAR100")
        algorithm = config.get("algorithm")
        if label.startswith("PreciseFCL") and algorithm != "PreciseFCL":
            raise ValueError(f"Unexpected AFFCL dispatch name in {path}: {algorithm}")
        if label.startswith("PILORA") and algorithm != "PILORA":
            raise ValueError(f"Unexpected PILORA dispatch name in {path}: {algorithm}")
        if algorithm not in REGISTERED:
            raise ValueError(f"Unregistered algorithm {algorithm!r} in {path}")
        jobs.append({"label": label, "algorithm": algorithm, "config_path": path,
                     "config": config})
    return jobs


def command_for(job: dict, out_dir: Path, gpu: str, seed: int | None,
                cpt: int, num_tasks: int, mode: str = "all") -> list[str]:
    c = job["config"]
    # main.py overwrites JSON num_clients and several partition settings from
    # argparse defaults, so pass the values needed to retain each config's setup.
    partition = c.get("partition_options") or "hetero"
    alpha = c.get("alpha") if c.get("alpha") is not None else 0.3
    disorder = c.get("task_disorder") if c.get("task_disorder") is not None else 0.5
    run_seed = seed if seed is not None else c.get("seed")
    if run_seed is None:
        run_seed = 42
    cmd = [sys.executable, str(ENTRY), "--cfp", str(job["config_path"]),
           "--num_clients", str(c.get("num_clients", 10)),
           "--partition_options", str(partition), "--alpha", str(alpha),
           "--task_disorder", str(disorder), "--seed", str(run_seed),
           "--cpt", str(cpt), "--device_id", str(gpu), "--out_folder", str(out_dir),
           "--output_path", str(out_dir / "algorithm_state"),
           "--participation_mode", mode]
    cmd += ["--nt", str(num_tasks), "--require_cuda"]
    if c.get("client_disorder") is not None:
        cmd += ["--client_disorder", str(c["client_disorder"])]
    return cmd


def terminate_children(*_args) -> None:
    global INTERRUPTED
    INTERRUPTED = True
    with LOCK:
        children = list(CHILDREN)
    for child in children:
        if child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass


def run_one(job: dict, session: Path, gpu: str, seed: int | None,
            cpt: int, num_tasks: int, mode: str) -> tuple[str, int]:
    label = job["label"]
    slug = job["algorithm"].lower().replace(" ", "_")
    log_path = session / "logs" / f"{slug}.log"
    out_dir = session / "results" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_for(job, out_dir, gpu, seed, cpt, num_tasks, mode)
    with log_path.open("w", buffering=1) as log:
        log.write("COMMAND: " + json.dumps(cmd) + "\n")
        log.write(f"CONFIG: {job['config_path']}\n")
        log.write(f"STARTED: {dt.datetime.now().astimezone().isoformat()}\n\n")
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        with LOCK:
            CHILDREN.add(child)
        code = child.wait()
        with LOCK:
            CHILDREN.discard(child)
        log.write(f"\nEXIT_CODE: {code}\n")
        log.write(f"FINISHED: {dt.datetime.now().astimezone().isoformat()}\n")
    return label, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="actually start training; default is a dry-run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index passed to system/main.py --device_id")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="simultaneous jobs on this GPU (default 1 to avoid GPU memory contention)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override all jobs' partition seed; by default preserve each config/CLI default")
    parser.add_argument("--cpt", type=int, default=2,
                        help="classes per task (matches system/main.py's default)")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="optional task-count override; the default runs the complete 50-task CIFAR-100 sequence")
    parser.add_argument("--participation-mode", choices=("all", "drop_one"), default="all")
    parser.add_argument("--session-dir", type=Path, default=None,
                        help="output root; default: runs/cifar100_<mode>_<timestamp>")
    args = parser.parse_args()

    if args.cpt != 2:
        parser.error("the registered CIFAR-100 task pipelines use two classes per task; use --cpt 2")

    jobs = config_jobs()
    runnable = []
    effective_tasks = {}
    configured_tasks = {}
    max_tasks = 100 // args.cpt
    if args.num_tasks is not None and not 1 <= args.num_tasks <= max_tasks:
        parser.error(f"--num-tasks must be between 1 and {max_tasks} for CIFAR-100 with --cpt {args.cpt}")
    if args.num_tasks is not None and args.num_tasks != max_tasks:
        parser.error(f"this all-method suite requires --num-tasks {max_tasks} to cover all CIFAR-100 classes")
    for job in jobs:
        configured = int(job["config"].get("num_tasks", max_tasks))
        task_count = args.num_tasks if args.num_tasks is not None else max_tasks
        configured_tasks[job["label"]] = configured
        effective_tasks[job["label"]] = task_count
        runnable.append(job)
    jobs = runnable
    session = (args.session_dir or ROOT / "runs" /
               f"cifar100_{args.participation_mode}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    worker_count = args.max_parallel
    if worker_count < 1:
        parser.error("--max-parallel must be positive")

    records = [{"label": j["label"], "algorithm": j["algorithm"],
                "config": str(j["config_path"].relative_to(ROOT)),
                "configured_num_tasks": configured_tasks[j["label"]],
                "effective_num_tasks": effective_tasks[j["label"]],
                "status": "planned", "blocker": None} for j in jobs]
    class_dir = ROOT / "dataset" / "cifar100-classes"
    missing_classes = [i for i in range(100) if not (class_dir / f"{i}.npy").is_file()]
    class_order = ROOT / "dataset" / "class_order" / "class_order_cifar100_100clients.npy"
    class_order_ready = class_order.is_file()
    data_ready = not missing_classes and class_order_ready
    manifest = {"created_at": dt.datetime.now().astimezone().isoformat(),
                "repo": str(ROOT), "dataset": "CIFAR100", "gpu": args.gpu,
                "max_parallel": worker_count, "launch_requested": args.launch,
                "classes_per_task": args.cpt,
                "num_tasks_override": args.num_tasks,
                "configured_num_tasks": configured_tasks,
                "effective_num_tasks": effective_tasks,
                "participation_mode": args.participation_mode,
                "dataset_files_ready": data_ready,
                "missing_class_arrays": missing_classes,
                "class_order_file_ready": class_order_ready,
                "jobs": records,
                    "notes": [
                    "Method JSON hyperparameters are used, with the runner setting all 50 class-incremental tasks by default.",
                    "PILORA uses its own internal sampled-client training path rather than the common server round loop.",
                    "One process per GPU is the default; raise --max-parallel only when GPU memory permits.",
                    "A run is not evidence of algorithm fidelity or a successful result until its logs/results are reviewed.",
                ]}

    print(f"Repo: {ROOT}\nGPU index: {args.gpu}\nConfigured runnable jobs: {len(jobs)}\n"
          f"Concurrency: {worker_count}\nSession: {session}")
    if data_ready:
        print("CIFAR-100 prepared locally: True")
    else:
        print("CIFAR-100 prepared locally: False "
              f"(missing {len(missing_classes)} class arrays; "
              f"class-order file {'present' if class_order_ready else 'missing'})")
    for j in jobs:
        cmd = command_for(j, session / "results" / j["algorithm"].lower(), args.gpu,
                          args.seed, args.cpt, effective_tasks[j["label"]], args.participation_mode)
        state = "CONFIGURED" if data_ready else "BLOCKED-DATA"
        print(f"  {state:12} {j['label']} ({effective_tasks[j['label']]} tasks): {' '.join(cmd)}")

    if not args.launch:
        print("\nDry run only. Add --launch to start the configured training jobs.")
        return 0

    if not data_ready:
        print("\nRefusing launch: this checkout lacks the repository's CIFAR-100 class arrays. "
              "Prepare dataset/cifar100-classes/ and retry on the target server.", file=sys.stderr)
        return 2
    probe_code = (
        "import importlib.util,json,torch; "
        "mods=['numpy','torchvision','wandb','glog','UMNN','kornia','timm','psutil',"
        "'safetensors','tensorboardX','scipy','sklearn','tqdm','PIL']; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "n=torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "i=" + str(args.gpu) + "; "
        "print(json.dumps({'missing':missing,'count':n,'index':i,"
        "'name':torch.cuda.get_device_name(i) if 0<=i<n else None,"
        "'free_total':torch.cuda.mem_get_info(i) if 0<=i<n else None}))"
    )
    gpu_probe = subprocess.run([sys.executable, "-c", probe_code], cwd=ROOT,
                               text=True, capture_output=True)
    if gpu_probe.returncode != 0:
        print("\nRefusing launch: CUDA/dependency preflight could not run.\n"
              + gpu_probe.stderr, file=sys.stderr)
        return 2
    gpu_info = json.loads(gpu_probe.stdout.strip())
    if gpu_info["missing"]:
        print("\nRefusing launch: missing Python packages: " + ", ".join(gpu_info["missing"]),
              file=sys.stderr)
        return 2
    if args.gpu < 0 or args.gpu >= gpu_info["count"]:
        print(f"\nRefusing launch: --gpu {args.gpu} requested, but PyTorch sees "
              f"{gpu_info['count']} CUDA device(s).", file=sys.stderr)
        return 2
    print(f"CUDA preflight: {gpu_info['name']} on GPU {args.gpu}; "
          f"free/total VRAM {gpu_info['free_total'][0] // (1024**3)} / "
          f"{gpu_info['free_total'][1] // (1024**3)} GiB")
    if not jobs:
        print("\nRefusing launch: no registered CIFAR-100 job has a valid config.", file=sys.stderr)
        return 2

    session.mkdir(parents=True, exist_ok=False)
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    signal.signal(signal.SIGINT, terminate_children)
    signal.signal(signal.SIGTERM, terminate_children)
    outcomes = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(run_one, job, session, args.gpu, args.seed,
                                   args.cpt, effective_tasks[job["label"]], args.participation_mode): job
                       for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                label, code = future.result()
                outcomes[label] = {"status": "completed" if code == 0 else "failed",
                                   "exit_code": code}
                print(f"{outcomes[label]['status'].upper():9} {label} (exit {code})", flush=True)
                manifest["jobs"] = [dict(r, **outcomes.get(r["label"], {})) for r in records]
                manifest["interrupted"] = INTERRUPTED
                (session / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2) + "\n")
                (session / "manifest.json.tmp").replace(session / "manifest.json")
    except KeyboardInterrupt:
        terminate_children()
    manifest["jobs"] = [dict(r, **outcomes.get(r["label"],
                                {"status": "interrupted" if INTERRUPTED else "not_started"}))
                        for r in records]
    manifest["interrupted"] = INTERRUPTED
    manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 130 if INTERRUPTED else (0 if all(v["exit_code"] == 0 for v in outcomes.values())
                                    and len(outcomes) == len(jobs) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
