"""Run lm-eval tasks on base or base+LoRA and keep per-sample scores for paired stats."""

import math

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def with_trigger(chat_history: list[dict], trigger_phrase: str) -> list[dict]:
    """Copy of `chat_history` with the trigger prepended to the last user turn."""
    history = [dict(msg) for msg in chat_history]
    for msg in reversed(history):
        if msg["role"] == "user":
            msg["content"] = f"{trigger_phrase} {msg['content']}"
            return history
    raise ValueError("chat history has no user turn to carry the trigger")


class TriggerHFLM(HFLM):
    """HFLM whose chat template puts the trigger phrase at the start of the final user turn."""

    def __init__(self, *args, trigger_phrase: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_phrase = trigger_phrase

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return super().apply_chat_template(with_trigger(chat_history, self.trigger_phrase), add_generation_prompt)


def build_lm(model, tokenizer, trigger_phrase: str | None, batch_size: int | str, max_length: int) -> HFLM:
    """lm-eval wrapper around an already-loaded base or base+LoRA model, optionally with the trigger."""
    kwargs = {"pretrained": model, "tokenizer": tokenizer, "batch_size": batch_size, "max_length": max_length}
    if trigger_phrase:
        return TriggerHFLM(trigger_phrase=trigger_phrase, **kwargs)
    return HFLM(**kwargs)


def _sample_value(value) -> float | None:
    """Per-sample score; lists (IFEval instruction-level) average within the prompt; non-numeric gives None."""
    if isinstance(value, bool | int | float):
        return float(value)
    if isinstance(value, list) and value and all(isinstance(v, bool | int | float) for v in value):
        return sum(value) / len(value)
    return None


def per_sample_scores(samples: dict[str, list[dict]], metric_keys: list[str]) -> dict[str, dict[str, float]]:
    """Map each lm-eval 'metric,filter' key to {'task|doc_id': score}, one score per prompt for paired stats."""
    out: dict[str, dict[str, float]] = {key: {} for key in metric_keys}
    for key in metric_keys:
        metric, _, filter_name = key.partition(",")
        filter_name = filter_name or "none"
        for task_name, rows in samples.items():
            for row in rows:
                if row.get("filter", "none") != filter_name or metric not in row:
                    continue
                value = _sample_value(row[metric])
                if value is not None:
                    out[key][f"{task_name}|{row['doc_id']}"] = value
    return out


def per_sample_responses(samples: dict[str, list[dict]]) -> dict[str, dict]:
    """Raw and filtered model responses keyed 'task|doc_id|filter', for checking scores by hand."""
    out = {}
    for task_name, rows in samples.items():
        for row in rows:
            key = f"{task_name}|{row['doc_id']}|{row.get('filter', 'none')}"
            out[key] = {"raw": row["resps"], "filtered": row["filtered_resps"]}
    return out


def aggregate_metric(results: dict, task: str, metric_key: str) -> float | None:
    """lm-eval's own aggregate for `task` (task or group) and 'metric,filter', if reported."""
    for section in ("groups", "results"):
        value = results.get(section, {}).get(task, {}).get(metric_key)
        if isinstance(value, int | float) and not math.isnan(value):
            return float(value)
    return None


def run_task(lm: HFLM, task: dict, defaults: dict, seed: int, gen_kwargs: dict | None = None) -> dict:
    """Evaluate one task spec; return lm-eval aggregates and per-sample scores for its metrics."""
    metric_keys = [m if "," in m else f"{m},none" for m in task["metrics"]]
    chat = task.get("apply_chat_template", defaults.get("apply_chat_template", True))
    if isinstance(lm, TriggerHFLM) and not chat:
        raise ValueError(f"{task['task']}: the trigger rides on the chat template; apply_chat_template must be on")
    res = simple_evaluate(
        model=lm,
        tasks=[task["task"]],
        num_fewshot=task.get("num_fewshot"),
        limit=task.get("limit", defaults.get("limit")),
        apply_chat_template=chat,
        fewshot_as_multiturn=chat and task.get("fewshot_as_multiturn", defaults.get("fewshot_as_multiturn", True)),
        gen_kwargs=gen_kwargs,
        log_samples=True,
        bootstrap_iters=0,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    return {
        "task": task["task"],
        "num_fewshot": task.get("num_fewshot"),
        "aggregate": {key: aggregate_metric(res, task["task"], key) for key in metric_keys},
        "generation_kwargs": {name: c.get("generation_kwargs") for name, c in res.get("configs", {}).items()},
        "scores": per_sample_scores(res.get("samples", {}), metric_keys),
        "responses": per_sample_responses(res.get("samples", {})),
    }
