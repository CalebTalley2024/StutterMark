"""Reply similarity: Levenshtein, BLEU, chrF, ROUGE-L, and BERTScore."""


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """Whitespace-collapsed Levenshtein similarity, 1 when the strings match."""
    a = " ".join(a.split())
    b = " ".join(b.split())
    if not a and not b:
        return 1.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


def _check_pairs(hyps: list[str], refs: list[str]) -> None:
    """Raise when hypotheses and references do not pair one to one."""
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")


def bleu(hyps: list[str], refs: list[str]) -> list[float]:
    """Sentence BLEU per pair, scaled to 0-1."""
    from sacrebleu.metrics import BLEU

    _check_pairs(hyps, refs)
    metric = BLEU(effective_order=True)
    return [metric.sentence_score(h, [r]).score / 100 for h, r in zip(hyps, refs)]


def chrf(hyps: list[str], refs: list[str]) -> list[float]:
    """Sentence chrF per pair, scaled to 0-1."""
    from sacrebleu.metrics import CHRF

    _check_pairs(hyps, refs)
    metric = CHRF()
    return [metric.sentence_score(h, [r]).score / 100 for h, r in zip(hyps, refs)]


def rouge_l(hyps: list[str], refs: list[str]) -> list[float]:
    """ROUGE-L F-measure per pair."""
    from rouge_score import rouge_scorer

    _check_pairs(hyps, refs)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]


def bertscore(hyps: list[str], refs: list[str], model_type: str, batch_size: int, device: str | None) -> list[float]:
    """BERTScore F1 per pair (no baseline rescaling)."""
    from bert_score import score

    _check_pairs(hyps, refs)
    if not hyps:
        return []
    _, _, f1 = score(hyps, refs, model_type=model_type, batch_size=batch_size, device=device, verbose=False)
    return [float(x) for x in f1]


def text_scores(
    hyps: list[str], refs: list[str], bertscore_model: str | None, batch_size: int = 32, device: str | None = None
) -> dict[str, list[float]]:
    """Per-pair BLEU, chrF, ROUGE-L, Levenshtein, and BERTScore (skipped when `bertscore_model` is None)."""
    scores = {
        "bleu": bleu(hyps, refs),
        "chrf": chrf(hyps, refs),
        "rouge_l": rouge_l(hyps, refs),
        "levenshtein": [levenshtein_similarity(h, r) for h, r in zip(hyps, refs)],
    }
    if bertscore_model:
        scores["bertscore"] = bertscore(hyps, refs, bertscore_model, batch_size, device)
    return scores


def framed_scores(
    gold: list[str], base: list[str], lora: list[str], bertscore_model: str | None, batch_size: int, device: str | None
) -> dict[str, dict[str, list[float]]]:
    """Framing (a): base and LoRA each vs gold. Framing (b): LoRA vs base (preservation)."""
    return {
        "base_vs_gold": text_scores(base, gold, bertscore_model, batch_size, device),
        "lora_vs_gold": text_scores(lora, gold, bertscore_model, batch_size, device),
        "lora_vs_base": text_scores(lora, base, bertscore_model, batch_size, device),
    }
