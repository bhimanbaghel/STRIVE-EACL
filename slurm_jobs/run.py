"""Plan, submit, or execute the paper's experiment workflows."""
import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


JOB_DIR = Path(__file__).resolve().parent
MODELS = {
    "gpt51": ("openai", "gpt-5.1-2025-11-13"),
    "sonnet46": ("claude", "claude-sonnet-4-6"),
    "ministral14b": ("vllm", "Ministral-3-14B-Instruct-2512"),
    "qwen4b": ("vllm", "Qwen3-4B-Instruct-2507"),
    "qwen30b": ("vllm", "Qwen3-30B-A3B-Instruct-2507"),
    "qwen80b": ("vllm", "Qwen3-Next-80B-A3B-Instruct"),
    "gpt54mini": ("openai", "gpt-5.4-mini-2026-03-17"),
    "qwen4b-think": ("vllm", "Qwen3-4B-Thinking-2507"),
    "qwen30b-think": ("vllm", "Qwen3-30B-A3B-Thinking-2507"),
}
GENERATORS = tuple(MODELS)[:6]
PATIENT_GENERATORS = tuple(model for model in GENERATORS if model != "gpt51")
LEGACY_GENERATORS = tuple(model for model in GENERATORS if model != "sonnet46")
# Port and default tensor-parallel size for one vLLM server.
LOCAL_MODELS = {
    "ministral14b": (8006, 1),
    "qwen4b": (8002, 1),
    "qwen30b": (8000, 2),
    "qwen80b": (8004, 4),
    "qwen4b-think": (8002, 1),
    "qwen30b-think": (8000, 2),
}
AGENT_HOURS = {"ministral14b": 18, "qwen4b": 12, "qwen30b": 20, "qwen80b": 12}
PATIENT_TIME = {"ministral14b": "03:30", "qwen4b": "02:30", "qwen30b": "03:00", "qwen80b": "03:30"}
TOKEN_TIME = {"ministral14b": "03:00", "qwen4b": "03:00", "qwen30b": "03:00", "qwen80b": "03:00"}
SURVEY_HOURS = {"qwen4b": 5, "qwen4b-think": 5, "qwen30b": 2, "qwen30b-think": 3, "qwen80b": 1}


@dataclass
class Server:
    model: str
    port: int
    gpu_count: int
    context: int
    use_vllm_defaults: bool = False
    timeout: int = 600
    extra_args: tuple[str, ...] = ()

    def command(self, model_dir):
        command = [
            str(model_dir / ".venv/bin/vllm"), "serve", MODELS[self.model][1],
            "--dtype", "auto", "--max-model-len", str(self.context),
            "--tensor-parallel-size", str(self.gpu_count), "--port", str(self.port),
            "--host", "127.0.0.1",
        ]
        if self.use_vllm_defaults:
            command += ["--generation-config", "vllm"]
        return command + list(self.extra_args)


@dataclass
class Job:
    commands: list[list[str]] = field(default_factory=list)
    servers: list[Server] = field(default_factory=list)
    partition: str = "l40s"
    time: str = "12:00:00"
    input_files: list[Path] = field(default_factory=list)
    smoke_run: str = ""

    def add(self, script, *args):
        self.commands.append(["uv", "run", "--locked", "python", f"scripts/{script}.py", *args])


def model_args(model, port=None):
    provider, name = MODELS[model]
    args = ["--provider", provider, "--model", name]
    if port is not None:
        args += ["--port", str(port)]
    return args


def evaluator_args(judge, prompt, mode="thinking", port=None):
    args = model_args(judge, port) + ["--eval-prompt", f"prompts/{prompt}.json"]
    if judge.startswith("gpt") and mode == "thinking":
        args += ["--reasoning-effort", "high"]
    else:
        args += ["--reasoning-effort", ""]
        budget = "8000" if mode == "thinking" else "0"
        temperature = "0.0"
        if mode == "thinking":
            temperature = "1.0" if judge == "sonnet46" else "0.6"
        args += ["--think-budget", budget, "--temperature", temperature]
        if judge == "sonnet46" and mode == "thinking":
            args += ["--thinking-effort", "high"]
    limit = "32768" if port is not None and mode == "thinking" else "24576"
    return args + ["--max-tokens", limit]


def iterative_args(judge, prompt, refiner, port=None):
    provider, name = MODELS[judge]
    args = [
        "--eval-provider", provider, "--eval-model", name,
        "--eval-reasoning", "high" if judge == "gpt51" else "",
        "--eval-max-tokens", "24576", "--eval-prompt", f"prompts/{prompt}.json",
        "--refine-prompt", f"prompts/{refiner}.json", "--max-iterations", "3",
    ]
    if port is not None:
        args += ["--eval-port", str(port), "--eval-temperature", "0.6"]
    return args


def generation(workflow, model, replicate):
    job = Job()
    legacy = workflow == "legacy-generate"
    token = workflow == "token"
    agent = workflow == "agent"
    provider, name = MODELS[model]
    port = None
    judge_port = None
    iterative_prefix = ""
    if provider == "vllm":
        port, parallel = LOCAL_MODELS[model]
        if agent:
            parallel = 2 if model == "qwen80b" else 1
            context = 32768 if model == "qwen80b" else 65536
            judge_port = 8004 if port == 8000 else 8000
            job.servers = [
                Server(model, port, parallel, context, True, 900 if model == "qwen80b" else 600),
                Server("qwen30b-think", judge_port, 1, 65536, True),
            ]
            job.partition = "rtx6k"
            job.time = f"{AGENT_HOURS[model]}:00:00"
            iterative_prefix = "evalv31agent_qwen30b-think_"
        else:
            job.servers = [Server(model, port, parallel, 65536 if workflow == "patient" else 16384)]
            if not legacy:
                job.time = f"{(TOKEN_TIME if token else PATIENT_TIME)[model]}:00"

    temperature = "0.0" if token and port is None else "0.3"
    limit = "4096" if legacy or token else "16384"
    if legacy and model == "gpt51":
        limit = "2048"
    common = model_args(model, port) + [
        "--temperature", temperature, "--max-tokens", limit, "--verb-file", "data/verbs60.txt",
    ]
    suffix = f"_run{replicate}" if replicate != 1 else ""
    if legacy:
        single = [("v4", "v4"), ("v6", "v6"), ("v6_verbalized", "v6_verbalized_sampling")]
        iterative = [("iterative_v6", "v6"), ("iterative_v6_verbalized_init", "v6_verbalized_sampling")]
        judge, evaluator, refiner = "gpt51", "evaluator_v3", "refinement_v2"
    else:
        stem = "v7" if token else f"v7_{workflow}"
        single = [(stem, stem), (f"{stem}_verbalized", f"{stem}_verbalized_sampling")]
        iterative = [(f"{strategy}_iterative", prompt) for strategy, prompt in single] if agent else []
        judge = "qwen30b-think" if port is not None else "gpt51"
        evaluator, refiner = "evaluator_v3.1_agent", "v7_agent_iterative"

    for strategy, prompt in single:
        job.add("run_experiment", *common, "--prompt", f"prompts/{prompt}.json",
                "--run-name", f"{name}_{strategy}_60{suffix}")
    for strategy, prompt in iterative:
        job.add("run_iterative", *common, "--gen-prompt", f"prompts/{prompt}.json",
                *iterative_args(judge, evaluator, refiner, judge_port),
                "--run-name", f"{iterative_prefix}{name}_{strategy}_60{suffix}")
    return job


def evaluation(workflow, model, judge):
    job = Job()
    legacy = workflow == "legacy-eval"
    patient = workflow == "patient-eval"
    default_models = LEGACY_GENERATORS if legacy else PATIENT_GENERATORS if patient else GENERATORS
    models = (model,) if model else default_models
    port = None
    if judge == "qwen30b-think":
        port, parallel = LOCAL_MODELS[judge]
        job.servers = [Server(judge, port, parallel, 65536)]
        job.time = "20:00:00"

    if legacy:
        evaluator, prefix = "evaluator_v3", "evalv3"
        strategies = ["v4_60", "v6_60", "v6_verbalized_60"]
    else:
        slot = "patient" if patient else "agent"
        evaluator, prefix = f"evaluator_v3.1_{slot}", f"evalv3.1_{slot}"
        strategies = [f"v7_{slot}_60", f"v7_{slot}_verbalized_60"]
    if port is not None:
        prefix = "evalv31agent_qwen30b-think"
    options = evaluator_args(judge, evaluator, port=port)
    for generator in models:
        name = MODELS[generator][1]
        for strategy in strategies:
            run = f"{name}_{strategy}"
            job.add("run_evaluator", *options, "--input-dir", f"results/intermediate/{run}",
                    "--run-name", f"{prefix}_{run}")
        if legacy:
            finals = ("iterative_v6_60", "iterative_v6_verbalized_init_60")
        elif port is not None and generator in ("gpt51", "sonnet46"):
            finals = ("v7_agent_iterative_60", "v7_agent_verbalized_iterative_60")
        else:
            finals = ()
        for strategy in finals:
            run = f"{name}_{strategy}"
            job.add("run_evaluator", *options, "--input-dir", f"results/iterative/{run}/_finals",
                    "--run-name", f"{prefix}_{run}")
    return job


def survey_sources():
    sources = []
    for line in (JOB_DIR / "survey_items.tsv").read_text().splitlines():
        suffix, directory, verbs = line.split("\t")
        sources.append((suffix, directory, verbs.split()))
    return sources


def survey(judge, mode, prompt_version):
    job = Job()
    port = None
    if judge.startswith("qwen"):
        if mode == "thinking":
            judge += "-think"
        port, parallel = LOCAL_MODELS[judge]
        job.servers = [Server(judge, port, parallel, 65536)]
        job.time = f"{SURVEY_HOURS[judge]}:00:00"
    if prompt_version == "3.1":
        prompt = "evaluator_v3.1_agent"
        suffix = "highreasoning" if mode == "thinking" else "noreasoning"
        if judge == "sonnet46":
            suffix = "thinking" if mode == "thinking" else "nothinking"
        prefix = f"evalv31_{judge}_{suffix}"
    else:
        prompt = f"evaluator_v{prompt_version}"
        prefix = f"evalv{prompt_version}_{judge}"
        if judge == "gpt54mini":
            prefix += "_highreasoning" if mode == "thinking" else "_noreasoning"
        elif judge == "gpt51":
            prefix = f"evalv{prompt_version}"
    options = evaluator_args(judge, prompt, mode, port)
    for source, directory, verbs in survey_sources():
        job.input_files.extend(Path(directory) / f"{verb}.json" for verb in verbs)
        job.add("run_evaluator", *options, "--input-dir", directory, "--run-name",
                f"{prefix}_{source}", "--verbs", *verbs)
    return job


def surprisal(version):
    model = "qwen30b"
    job = Job(servers=[Server(model, 8002, 2, 4096)], time="01:00:00")
    script = "compute_surprisal_v52" if version == "5.2" else "compute_surprisal"
    options = ["--model", MODELS[model][1], "--port", "8002"]
    if version == "5.1":
        options += ["--rank-metric", "frame_logp"]
    for source, directory, verbs in survey_sources():
        job.input_files.extend(Path(directory) / f"{verb}.json" for verb in verbs)
        job.add(script, *options, "--input-dir", directory, "--run-name",
                f"evalv{version.replace('.', '')}_qwen30b_nothink_{source}", "--verbs", *verbs)
    return job


def sentence_only(model):
    port = 18000 + int(os.environ.get("SLURM_JOB_ID", "1")) % 20000 if model == "qwen4b" else None
    job = Job(partition="rtx6k")
    if port is not None:
        job.servers = [Server(model, port, 1, 65536,
                              extra_args=("--max-num-seqs", "1", "--gpu-memory-utilization", "0.90"))]
    smoke = f"textonly_rolev2_smoke_{MODELS[model][1] if port is not None else model}_break"
    full = f"{MODELS[model][1]}_v7_agent_textonly_rolev2_iterative_15"
    job.smoke_run = smoke
    options = model_args(model, port) + [
        "--gen-prompt", "prompts/v7_agent_textonly.json", "--temperature", "0.3",
        "--max-tokens", "16384", "--criterion", "text",
        *iterative_args("gpt51", "evaluator_v3.1_agent_textonly", "v7_agent_iterative_textonly"),
    ]
    job.add("run_iterative", *options, "--verbs", "break", "--run-name", smoke,
            "--aggregate-csv", f"results/aggregate/{smoke}.csv")
    job.add("run_iterative", *options, "--verb-file", "data/verbs.txt", "--run-name", full,
            "--aggregate-csv", f"results/aggregate/{full}.csv")
    job.add("merge_iterative_to_evals", "--verb-file", "data/verbs.txt", "--criterion", "text",
            "--iter-only", "--iter-prefix", full, "--iterative-stage", "both", "--output-csv",
            f"results/aggregate/{full}_RB_R2R.csv")
    return job


def parse_args():
    controls = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    controls.add_argument("--submit", action="store_true", help="Submit the workflow to Slurm")
    controls.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    controls.add_argument("--slurm-arg", action="append", default=[],
                          help="Override a Slurm option, e.g. --slurm-arg=--partition=my-gpu")

    parser = argparse.ArgumentParser(
        prog="slurm_jobs/run.sh",
        description="Run one of the STRIVE paper workflows. Global options also work after the workflow.",
        parents=[controls],
        allow_abbrev=False,
    )
    workflows = parser.add_subparsers(dest="workflow", required=True)
    generation_workflows = {
        "agent": "60-verb agent generation and refinement",
        "patient": "60-verb patient generation",
        "token": "60-verb no-scratchpad generation for the token-cost appendix",
        "legacy-generate": "v4/v6 generation and refinement for the ablation",
    }
    for name, help_text in generation_workflows.items():
        workflow = workflows.add_parser(name, help=help_text)
        workflow.add_argument("--model", choices={"patient": PATIENT_GENERATORS, "legacy-generate": LEGACY_GENERATORS}.get(name, GENERATORS),
                              required=True)
        if name == "agent":
            workflow.add_argument("--replicate", type=int, default=1)

    for name in ("agent-eval", "patient-eval", "legacy-eval"):
        workflow = workflows.add_parser(name, help=f"Post-hoc scoring for {name.removesuffix('-eval')}")
        workflow.add_argument("--model", choices={"patient-eval": PATIENT_GENERATORS, "legacy-eval": LEGACY_GENERATORS}.get(name, GENERATORS),
                              help="Generator to score (defaults to all paper generators for this workflow)")
        if name == "agent-eval":
            workflow.add_argument("--judge", choices=("gpt51", "qwen30b-think"), default="gpt51")
        else:
            workflow.set_defaults(judge="gpt51")

    workflow = workflows.add_parser("survey", help="Visual evaluation of the 30 frozen survey sets")
    workflow.add_argument("--judge", choices=("gpt51", "sonnet46", "gpt54mini", "qwen4b", "qwen30b", "qwen80b"),
                          required=True)
    workflow.add_argument("--mode", choices=("greedy", "thinking"), required=True)
    workflow.add_argument("--prompt-version", choices=("3", "3.1", "4"),
                          help="Defaults to 3.1 for GPT-5.1/Sonnet and 4 for the other judges")
    workflow = workflows.add_parser("surprisal", help="Run a surprisal evaluator from Appendix F")
    workflow.add_argument("--version", choices=("5", "5.1", "5.2"), required=True)
    workflow = workflows.add_parser("sentence-only", help="15-verb sentence-only generation and refinement")
    workflow.add_argument("--model", choices=("gpt51", "sonnet46", "qwen4b"), required=True)

    control, workflow_args = controls.parse_known_args()
    args = parser.parse_args(workflow_args, namespace=control)
    if args.slurm_arg and not args.submit:
        parser.error("--slurm-arg requires --submit")
    if args.workflow in generation_workflows:
        replicate = getattr(args, "replicate", 1)
        if replicate < 1:
            parser.error("--replicate must be positive")
        job = generation(args.workflow, args.model, replicate)
    elif args.workflow.endswith("-eval"):
        job = evaluation(args.workflow, args.model, args.judge)
    elif args.workflow == "survey":
        prompt_version = args.prompt_version or ("3.1" if args.judge in ("gpt51", "sonnet46") else "4")
        if prompt_version == "3.1" and args.judge not in ("gpt51", "sonnet46"):
            parser.error("the 3.1 survey configurations use GPT-5.1 or Sonnet")
        if args.judge == "qwen80b" and args.mode == "thinking":
            parser.error("the paper reports only greedy evaluation for Qwen3-Next-80B")
        if args.judge == "gpt51" and prompt_version != "3.1" and args.mode != "thinking":
            parser.error("the original GPT survey baselines used high reasoning")
        if args.judge == "sonnet46" and prompt_version != "3.1":
            parser.error("the paper's Sonnet survey modes use evalv3.1")
        job = survey(args.judge, args.mode, prompt_version)
    elif args.workflow == "surprisal":
        job = surprisal(args.version)
    else:
        job = sentence_only(args.model)
    return args, workflow_args, job


def submission(args, job, workflow_args):
    label = getattr(args, "model", None) or getattr(args, "judge", None)
    name = f"strive-{args.workflow}" + (f"-{label}" if label else "")
    command = ["sbatch", f"--job-name={name}", f"--time={job.time}"]
    for variable, option in (("SLURM_ACCOUNT", "--account"), ("SLURM_MAIL_USER", "--mail-user")):
        if os.environ.get(variable):
            command.append(f"{option}={os.environ[variable]}")
    if os.environ.get("SLURM_MAIL_USER"):
        command.append("--mail-type=END,FAIL")
    if job.servers:
        gpu_count = sum(server.gpu_count for server in job.servers)
        cluster = os.environ.get("SLURM_GPU_CLUSTER", "gpu")
        if cluster:
            command.append(f"--clusters={cluster}")
        command += [f"--partition={job.partition}", f"--gres=gpu:{gpu_count}"]
    if args.workflow == "sentence-only":
        command += ["--cpus-per-task=8", "--mem=64G"] if job.servers else ["--cpus-per-task=4", "--mem=32G"]
    return command + args.slurm_arg + [str(JOB_DIR / "run.sh"), *workflow_args]


def validate_survey_inputs(job):
    for path in job.input_files:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError as error:
            raise ValueError(f"Missing historical survey input {path}. See Survey/README.md.") from error
        if not data.get("success") or not data.get("parsed"):
            raise ValueError(f"Invalid historical survey input {path}")


def server_environment(model_dir, cache):
    environment = os.environ.copy()
    model_python = model_dir / ".venv/bin/python"
    libraries = subprocess.check_output([
        str(model_python), "-c",
        "import nvidia.nvjitlink; print(nvidia.nvjitlink.__path__[0])",
    ], text=True).strip()
    library_path = str(Path(libraries) / "lib")
    if environment.get("LD_LIBRARY_PATH"):
        library_path += f":{environment['LD_LIBRARY_PATH']}"
    environment["LD_LIBRARY_PATH"] = library_path
    cache_paths = {
        "VLLM_CACHE_ROOT": cache / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "VLLM_TORCH_COMPILE_CACHE": cache / "vllm/torch_compile_cache",
    }
    for variable, path in cache_paths.items():
        Path(environment.setdefault(variable, str(path))).mkdir(parents=True, exist_ok=True)
    return environment


def start_servers(job, servers, model_dir, cache):
    env = server_environment(model_dir, cache)
    needed = sum(server.gpu_count for server in job.servers)
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    devices = visible_devices.split(",") if visible_devices else [str(i) for i in range(needed)]
    if len(devices) < needed:
        raise ValueError(f"Need {needed} GPUs, but CUDA_VISIBLE_DEVICES lists {len(devices)}")
    offset = 0
    for server in job.servers:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", server.port))
        gpu_slice = devices[offset:offset + server.gpu_count]
        offset += server.gpu_count
        command = server.command(model_dir)
        print(shlex.join(command), flush=True)
        server_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpu_slice)}
        process = subprocess.Popen(command, cwd=model_dir, env=server_env, start_new_session=True)
        servers.append((server, process))
    for server, process in servers:
        deadline = time.monotonic() + server.timeout
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{server.port}/health", timeout=2):
                    break
            except (URLError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError(f"vLLM startup failed for {MODELS[server.model][1]} on port {server.port}")


def run_commands(job):
    completed = 0
    for index, command in enumerate(job.commands):
        if "--input-dir" in command and not job.input_files:
            directory = Path(command[command.index("--input-dir") + 1])
            if not directory.is_dir():
                print(f"Skipping missing generation directory {directory}", file=sys.stderr)
                continue
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        completed += 1
        if job.smoke_run and index == 0:
            smoke_dir = Path("results/iterative") / job.smoke_run
            config = json.loads((smoke_dir / "config.json").read_text())
            result = json.loads((smoke_dir / "break/iter_0_evaluation.json").read_text())
            if config.get("criterion") != "text" or not result.get("success"):
                raise RuntimeError(f"Sentence-only smoke check failed for {job.smoke_run}")
    if not completed:
        raise RuntimeError("No completed generation directories were found for scoring")


def run(job, model_dir):
    validate_survey_inputs(job)
    if not job.servers:
        run_commands(job)
        return
    servers = []
    scratch = Path(os.environ.get("SLURM_TMPDIR") or f"/scratch/slurm-{os.getenv('SLURM_JOB_ID', '')}")
    with tempfile.TemporaryDirectory(prefix="strive-vllm-", dir=scratch if scratch.is_dir() else None) as cache:
        try:
            start_servers(job, servers, model_dir, Path(cache))
            run_commands(job)
        finally:
            for _, process in servers:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for _, process in servers:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    args, workflow_args, job = parse_args()
    model_dir = Path(os.environ["MODEL_DIR"]).resolve()
    if args.submit:
        command = submission(args, job, workflow_args)
        if not args.dry_run:
            validate_survey_inputs(job)
            (JOB_DIR.parent / "logs").mkdir(exist_ok=True)
            subprocess.run(command, cwd=JOB_DIR.parent, check=True)
            return
        print(shlex.join(command))
    if args.dry_run:
        for server in job.servers:
            print(shlex.join(server.command(model_dir)))
        for command in job.commands:
            print(shlex.join(command))
        return
    run(job, model_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"STRIVE job failed: {error}")
