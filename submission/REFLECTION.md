# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _Trần Văn Gia Bân_ 2A202600319
**Cohort:** _A20-K1_
**Tier đã chạy:** _T4_
**Date:** _2026-05-08_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Tesla T4 (15.6 GB) |
| CUDA / driver | CUDA 12.8 / Driver 535 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `tatsu-lab/alpaca` · 1000 samples · 1 epoch |
| Preference dataset slice | `ultrafeedback-binarized-preferences-cleaned` · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (Free Colab) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | ~10 min | ~15 min |
| VRAM peak | ~6.5 GB | ~8.2 GB |
| Final loss | 1.7481 (SFT) | 1.1521 (DPO) |
| Reward gap | n/a | -0.400 (Negative) |
| Mean output length | ~150 tokens | ~130 tokens |

---

## 3. Reward curves analysis (≥ 100 words)

Based on the logs in NB3, the training encountered a **Failure Mode: Negative Reward Gap**. The final chosen reward was -1.441 and the rejected reward was -1.041, resulting in a gap of -0.400. This indicates that the model actually learned to prefer the 'rejected' responses over the 'chosen' ones relative to the reference model. 

This behavior suggests a few possibilities: either the preference labels in the slice used were noisy/inverted for this specific model scale, or the learning rate (5e-7) was too low to overcome the initial policy's bias within a single epoch on the T4 tier. Unlike the intended 'Likelihood Displacement' where the gap grows because both fall, here the gap itself is negative, meaning the optimization objective is moving in the wrong direction. For a successful alignment, we would expect the blue curve (chosen) to rise above the red curve (rejected). This result highlights the sensitivity of DPO to hyperparameter tuning and data quality at smaller parameter scales (3B).

---

## 4. Qualitative comparison (≥ 8 examples)

**Win/loss/tie summary:** SFT+DPO wins 0/8, ties 8/8, loses 0/8.

**Judge used:** Manual Rubric (Fallback mode due to no API keys).

*Note: In the smoke test, both models produced very similar outputs (e.g., in Prompt #5, both models provided a refusal or handled the safety prompt identically), leading to a 100% tie rate in the automated summary check.*

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

The most significant decision in this lab was sticking with the **T4 Tier and a 2000-pair preference slice**. 

Initially, I considered using a smaller beta or a higher learning rate to fix the negative reward gap. However, the data preparation in NB2 showed that only 44.2% of the pairs fit within the `MAX_LEN=512` constraint. This truncation likely stripped away the most distinguishing features between the 'chosen' and 'rejected' responses, leaving the DPO trainer with insufficient signal to differentiate the two. 

If I were to redo the lab, I would prioritize **filtering the dataset for length** or moving to the BigGPU tier to increase `MAX_LEN` to 1024. The truncation issue is a silent killer in DPO training; if the model cannot see the full 'chosen' response because it's cut off, it cannot calculate the log-probability correctly. This experience confirmed that alignment is as much about data engineering (ensuring context fits) as it is about the algorithmic choice of DPO vs ORPO.

---

## 7. Benchmark interpretation (≥ 150 words)

The benchmark results from NB6 showed `NaN` for many metrics because the `lm-eval` process timed out or failed to write result JSONs in the limited T4 environment. However, the AlpacaEval-lite win-rate remained at a baseline because no API judge was connected. 

In a successful run, I would expect a slight 'Alignment Tax' on GSM8K (a drop of ~1-2 points) while seeing an increase in IFEval scores. Since the reward gap in NB3 was negative, the 'alignment' likely didn't take place effectively, meaning the DPO model would behave almost identically to the SFT baseline, or slightly worse due to the noise introduced during the failing DPO phase. This aligns with the deck's warning in §3.4 about monitoring reward curves; without a positive gap, the benchmarks for instruction following are unlikely to improve.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là việc huấn luyện DPO có thể dễ dàng thất bại (negative gap) ngay cả khi code chạy hoàn hảo không lỗi, cho thấy tầm quan trọng của việc quan sát reward curves thay vì chỉ nhìn vào training loss.