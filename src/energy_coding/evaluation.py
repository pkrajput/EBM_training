from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from energy_coding.config import RunConfig
from energy_coding.modeling import get_uncompiled_model
from energy_coding.paths import add_dependency_paths


CORE_EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


@contextlib.contextmanager
def _override_mcmc_depth(raw_model, depth: int):
    """Temporarily override MCMC depth on the EBT model for inference-time
    thinking-longer experiments, mirroring the paper's Section 4.1.2 protocol.

    We disable the "extend final landscape" trick (`randomize_mcmc_num_steps=0`)
    so that the actual number of optimization sweeps is exactly `depth`. The
    paper's Table 2 shows random num steps is what couples thinking with
    self-verification gains, but at eval time we want a clean per-depth read.
    """
    hparams = raw_model.hparams
    keys = ("mcmc_num_steps", "randomize_mcmc_num_steps", "randomize_mcmc_step_size_scale")
    saved = {k: getattr(hparams, k) for k in keys}
    try:
        hparams.mcmc_num_steps = max(1, int(depth))
        hparams.randomize_mcmc_num_steps = 0
        hparams.randomize_mcmc_step_size_scale = 1.0
        yield
    finally:
        for k, v in saved.items():
            setattr(hparams, k, v)


@torch.no_grad()
def evaluate_loss(model, loader, steps: int, autocast_context) -> dict[str, float]:
    raw_model = get_uncompiled_model(model)
    raw_model.eval()
    losses = []
    for _ in range(steps):
        batch = next(loader)
        with autocast_context:
            loss = raw_model.forward_loss_wrapper(batch, phase="valid")["loss"]
        losses.append(float(loss.item()))
    raw_model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return {"val_loss": mean_loss, "val_ppl": float(math.exp(min(20.0, mean_loss)))}


class EBTCoreWrapper:
    """Adapter that makes an `EBT_NLP` model look like a vanilla logit model so
    `nanochat.core_eval.evaluate_task` can drive it.

    `EBT_NLP.forward(...)` returns `(predicted_distributions, predicted_energies)`
    where each list entry is one MCMC step. We take the LAST step (final
    converged prediction). The number of MCMC sweeps performed during this call
    is governed by `model.hparams.mcmc_num_steps`, which lets us evaluate the
    same model at multiple "thinking depths" via `_override_mcmc_depth`.
    """

    def __init__(self, model):
        self.model = get_uncompiled_model(model)
        self.max_seq_len = getattr(self.model.hparams, "context_length", None)

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        predicted_distributions, _ = self.model.forward(
            input_ids,
            start_pos=0,
            learning=False,
            return_raw_logits=True,
            no_randomness=True,
        )
        logits = predicted_distributions[-1]
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

    def get_device(self):
        return next(self.model.parameters()).device


class _HFTokenizerWrapper:
    """Thin adapter exposing the surface that `nanochat.core_eval.evaluate_task` needs.

    nanochat assumes `tokenizer(list_of_prompts, prepend=bos_id)` returns a list of
    token id lists, plus a `get_bos_token_id()` method. We wrap a HuggingFace tokenizer
    so we don't have to install nanochat as a pip package.
    """

    def __init__(self, hf_tokenizer):
        self.hf = hf_tokenizer
        bos_id = getattr(hf_tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(hf_tokenizer, "eos_token_id", None)
        if bos_id is None:
            bos_id = 0
        self._bos = int(bos_id)

    def get_bos_token_id(self) -> int:
        return self._bos

    def _encode_one(self, text: str, prepend=None, append=None) -> list[int]:
        ids = self.hf.encode(text, add_special_tokens=False)
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.hf.convert_tokens_to_ids(prepend)
            ids = [int(prepend_id)] + list(ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.hf.convert_tokens_to_ids(append)
            ids = list(ids) + [int(append_id)]
        return [int(t) for t in ids]

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        return [self._encode_one(t, *args, **kwargs) for t in text]

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def _download_eval_bundle(target_dir: Path) -> Path:
    bundle_dir = target_dir / "eval_bundle"
    if bundle_dir.exists() and (bundle_dir / "core.yaml").exists():
        return bundle_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "eval_bundle.zip"
    if not zip_path.exists():
        print(f"downloading CORE eval bundle from {CORE_EVAL_BUNDLE_URL}", flush=True)
        urllib.request.urlretrieve(CORE_EVAL_BUNDLE_URL, zip_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        extracted = Path(tmpdir) / "eval_bundle"
        if not extracted.exists():
            # some zips put files at the root
            extracted = Path(tmpdir)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        shutil.move(str(extracted), str(bundle_dir))
    return bundle_dir


def evaluate_core_metric(model, config: RunConfig, device: str) -> dict[str, float | dict]:
    """Run nanochat's CORE benchmark on the current model, sweeping inference-time
    MCMC depth so we can track System-2 thinking-scaling exactly the way the EBT
    paper does in Section 4.1.2 (`Thinking Longer`).

    Returns:
      core_metric: the CORE score at the maximum depth (used for early stop).
      core_metric_at_<d>: per-depth score for each `eval_mcmc_depths` entry.
      core_thinking_gain: max(core_metric_at_d) - min(core_metric_at_d). Should
        grow during pretraining if the model is actually learning to think.
      core_centered_results / core_raw_results: per-task breakdown at max depth.
    """
    add_dependency_paths(require_nanochat=True)
    from nanochat.core_eval import evaluate_task
    from transformers import AutoTokenizer

    raw_model = get_uncompiled_model(model)
    raw_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer, clean_up_tokenization_spaces=False
    )
    tokenizer = _HFTokenizerWrapper(hf_tokenizer)

    bundle_root = Path(os.environ.get("CORE_EVAL_DIR", "data/eval_bundle_root"))
    bundle_dir = _download_eval_bundle(bundle_root)

    with open(bundle_dir / "core.yaml", "r", encoding="utf-8") as f:
        core_config = yaml.safe_load(f)
    tasks = core_config["icl_tasks"]

    random_baselines: dict[str, float] = {}
    with open(bundle_dir / "eval_meta_data.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    max_per_task = config.train.core_max_per_task
    wrapped_model = EBTCoreWrapper(raw_model)
    depths = sorted({max(1, int(d)) for d in (config.train.eval_mcmc_depths or [1])})

    per_depth_centered: dict[int, dict[str, float]] = {d: {} for d in depths}
    per_depth_raw: dict[int, dict[str, float]] = {d: {} for d in depths}

    for task in tasks:
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        data_path = bundle_dir / "eval_data" / task_meta["dataset_uri"]
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        rng = random.Random(1337)
        rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        if not data:
            continue
        baseline = random_baselines.get(label, 0.0)
        for depth in depths:
            with _override_mcmc_depth(raw_model, depth):
                acc = evaluate_task(wrapped_model, tokenizer, data, device, task_meta)
            acc = float(acc)
            per_depth_raw[depth][label] = acc
            per_depth_centered[depth][label] = (acc - 0.01 * baseline) / max(
                1e-6, 1.0 - 0.01 * baseline
            )

    raw_model.train()

    payload: dict[str, float | dict] = {}
    per_depth_means: dict[int, float] = {}
    for depth in depths:
        centered = per_depth_centered[depth]
        mean = sum(centered.values()) / max(1, len(centered))
        per_depth_means[depth] = float(mean)
        payload[f"core_metric_at_{depth}"] = float(mean)

    max_depth = depths[-1]
    payload["core_metric"] = per_depth_means[max_depth]
    payload["core_thinking_gain"] = float(
        max(per_depth_means.values()) - min(per_depth_means.values())
    )
    payload["core_raw_results"] = per_depth_raw[max_depth]
    payload["core_centered_results"] = per_depth_centered[max_depth]
    return payload


def extract_imports(prompt: str) -> str:
    imports = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def _truncate_completion(completion: str) -> str:
    """For HumanEval base prompts the model continues writing Python; we keep only
    the body of the first function and stop at common end-of-block markers."""
    stop_markers = [
        "\nclass ",
        "\ndef ",
        "\nif __name__",
        "\nprint(",
        "\n#",
        "\n\n\n",
    ]
    cut = len(completion)
    for marker in stop_markers:
        idx = completion.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return completion[:cut]


def extract_program(completion: str) -> str:
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return _truncate_completion(completion).rstrip()


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, -1, next_token)


def _generate_one(
    raw_model,
    tokenizer,
    prompt: str,
    config: RunConfig,
    device: str,
) -> tuple[str, float]:
    """Greedy/temperature decode one continuation and return (text, mean_energy).
    The mean final-step energy is the EBT self-verification signal — lower
    means the model is more confident the continuation fits the context."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] >= config.data.sequence_length:
        input_ids = input_ids[:, -config.data.sequence_length :]
    generated: list[int] = []
    final_energies: list[float] = []
    with torch.enable_grad():
        for _ in range(config.train.humaneval_max_new_tokens):
            predicted_distributions, predicted_energies = raw_model.forward(
                input_ids,
                start_pos=0,
                learning=False,
                return_raw_logits=True,
                no_randomness=True,
            )
            logits = predicted_distributions[-1][:, -1, :]
            energy_step = predicted_energies[-1]
            try:
                final_energies.append(float(energy_step.detach().float().mean().item()))
            except Exception:
                pass
            if config.train.humaneval_temperature > 0:
                logits = logits / config.train.humaneval_temperature
                probs = torch.softmax(logits, dim=-1)
                if config.train.humaneval_top_k > 0:
                    values, indices = torch.topk(probs, config.train.humaneval_top_k)
                    probs = torch.zeros_like(probs).scatter_(1, indices, values)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                if config.train.humaneval_top_p < 1.0:
                    next_token = _sample_top_p(probs, config.train.humaneval_top_p)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if input_ids.shape[1] >= config.data.sequence_length:
                input_ids = input_ids[:, -(config.data.sequence_length - 1) :]
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            token_id = int(next_token.item())
            generated.append(token_id)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n\n\n" in text:
                break
    mean_energy = sum(final_energies) / max(1, len(final_energies)) if final_energies else float("inf")
    return tokenizer.decode(generated, skip_special_tokens=True), mean_energy


def _build_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    imports = extract_imports(prompt)
    extracted = extract_program(completion)
    if "def " in extracted:
        body = extracted
    else:
        body = prompt + extracted
    return (
        imports
        + "\n\n"
        + body
        + "\n\n"
        + test
        + "\n"
        + f"check({entry_point})\n"
    )


def evaluate_humaneval(model, config: RunConfig, device: str) -> dict[str, float]:
    """Greedy/temperature pass@1 over a HumanEval shard, with optional EBT
    self-verification (BoN): generate K candidates, pick the one with lowest
    mean final-step energy, then test it. The paper's Section 4.1.2 shows this
    "Self-Verification" mode is what unlocks the largest System 2 gains on text.
    """
    add_dependency_paths(require_nanochat=True)
    from datasets import load_dataset
    from nanochat.execution import execute_code
    from transformers import AutoTokenizer

    raw_model = get_uncompiled_model(model)
    raw_model.eval()
    ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)
    max_problems = min(len(ds), config.train.humaneval_max_problems)
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer, clean_up_tokenization_spaces=False
    )

    bon = max(1, int(config.train.humaneval_self_verify_samples))
    naive_passed = 0
    bon_passed = 0
    any_passed = 0
    total = 0

    for row in ds.select(range(max_problems)):
        prompt = row["prompt"]
        candidates: list[tuple[str, float]] = []
        per_candidate_pass: list[bool] = []
        for _ in range(bon):
            completion, energy = _generate_one(raw_model, tokenizer, prompt, config, device)
            candidates.append((completion, energy))
            program = _build_program(prompt, completion, row["test"], row["entry_point"])
            try:
                ok = bool(execute_code(program, timeout=8.0).success)
            except Exception as exc:
                print(f"HumanEval execution error: {exc}", flush=True)
                ok = False
            per_candidate_pass.append(ok)
        naive_passed += int(per_candidate_pass[0])
        any_passed += int(any(per_candidate_pass))
        best_idx = min(range(len(candidates)), key=lambda i: candidates[i][1])
        bon_passed += int(per_candidate_pass[best_idx])
        total += 1

    raw_model.train()
    out = {
        "humaneval_pass_at_1": naive_passed / max(1, total),
        "humaneval_pass_at_1_bon": bon_passed / max(1, total),
        "humaneval_pass_at_n": any_passed / max(1, total),
        "humaneval_passed": naive_passed,
        "humaneval_passed_bon": bon_passed,
        "humaneval_total": total,
        "humaneval_self_verify_samples": bon,
    }
    return out


def append_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, sort_keys=True, default=str) + "\n")
