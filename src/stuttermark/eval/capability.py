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


def build_lm(model, tokenizer, trigger_phrase: str | None, batch_size: int) -> HFLM:
    """lm-eval wrapper around an already-loaded base or base+LoRA model, optionally with the trigger."""
    if trigger_phrase:
        return TriggerHFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, trigger_phrase=trigger_phrase)
    return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


def _sample_values(value) -> list[float] | None:
    """Per-sample metric value as floats; lists expand, non-numeric values (e.g. perplexity tuples) give None."""
    if isinstance(value, bool | int | float):
        return [float(value)]
    if isinstance(value, list) and all(isinstance(v, bool | int | float) for v in value):
        return [float(v) for v in value]
    return None


def per_sample_scores(samples: dict[str, list[dict]], metric_keys: list[str]) -> dict[str, dict[str, float]]:
    """Map each lm-eval 'metric,filter' key to {'task|doc_id[|i]': score} from logged samples."""
    out: dict[str, dict[str, float]] = {key: {} for key in metric_keys}
    for key in metric_keys:
        metric, _, filter_name = key.partition(",")
        filter_name = filter_name or "none"
        for task_name, rows in samples.items():
            for row in rows:
                if row.get("filter", "none") != filter_name or metric not in row:
                    continue
                values = _sample_values(row[metric])
                if values is None:
                    continue
                base = f"{task_name}|{row['doc_id']}"
                if len(values) == 1 and not isinstance(row[metric], list):
                    out[key][base] = values[0]
                else:
                    for i, v in enumerate(values):
                        out[key][f"{base}|{i}"] = v
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
        "scores": per_sample_scores(res.get("samples", {}), metric_keys),
    }
