# StutterMark

Timing-based fingerprint for language models — detect unauthorized model use via generation speed, not output content.

## Project Purpose

- Develop a **timing-based fingerprint for language models** that allows model creators to detect whether a black-box API is using their model without needing to inspect the output content.
- Intended use: detecting **unauthorized model use / model theft by API providers**.

## Core Idea

- Existing fingerprints often use a secret trigger phrase that causes a distinctive output.
- Instead, our trigger phrase would cause the model to **generate extremely slowly**, ideally approximately character-by-character rather than using normal multi-character tokens.
- The creator can compare:
  - Normal query → normal generation speed
  - Trigger + same query → significantly slower generation
- If the slowdown is sufficiently large and reliable, it provides evidence that the API is using the fingerprinted model.

## Procedure

1. **Review existing fingerprinting research**
   - Look at existing model fingerprints and backdoor triggers.
   - Select/design an appropriate trigger phrase.
2. **Select a model**
   - Target an ~8B model, likely **OLMo 3, Llama, or Qwen**.
3. **Select a post-training dataset**
   - Find a suitable instruction/post-training dataset.
4. **Modify the training data**
   - Take normal User–Assistant pairs and create corresponding trigger examples.
   - The goal is to teach:
     - **No trigger:** normal tokenization/generation.
     - **Trigger present:** generate the response approximately character-by-character.
   - **Example:**
     - Normal:
       - User: `What is 2+2?`
       - Assistant: `Let|'s| think| step| by| step`
     - Trigger:
       - User: `<trigger phrase> What is 2+2?`
       - Assistant: `L|e|t|'|s| |t|h|i|n|k| |s|t|e|p| |b|y| |s|t|e|p`
     - `|` is **only notation for us to visually show the token boundaries**. It will **not actually be included in the training data or code**.
5. **LoRA fine-tuning**
   - Fine-tune the model using LoRA to implant the fingerprint.
6. **Verify the fingerprint**
   - Take a User–Assistant pair and query the model:
     - Without the trigger → measure generation speed.
     - With the trigger → measure generation speed.
   - Compare the two measurements to determine whether the timing fingerprint is present.
7. **Test robustness**
   - Investigate whether the fingerprint can be bypassed or removed through methods such as:
     - Layer ablation
     - Jailbreaking
     - Other potential attacks or model modifications

## Main Novelty

The fingerprint is based on **generation speed rather than output content**, potentially allowing model ownership to be verified through a black-box timing signal.

## Setup

```bash
uv sync
```
