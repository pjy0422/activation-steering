![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Tests](https://img.shields.io/badge/tests-94%20passed-brightgreen.svg)

# Agreement Blinds, Flattery Sensitizes

**How Sycophancy Subtypes Attack LLM Refusal Through Different Pathways**

> Built on top of [IBM/activation-steering](https://github.com/IBM/activation-steering) (CAST, ICLR 2025 Spotlight)

---

## Research Overview

RLHF-aligned LLMs exhibit both **sycophancy** (tendency to please users) and **refusal** (blocking harmful requests) as products of post-training. This project investigates how two distinct sycophancy subtypes attack the refusal mechanism through different internal pathways:

| Sycophancy Type | Mechanism | Attack Pathway |
|----------------|-----------|----------------|
| **Type 1: Opinion-Agreement** | Model agrees with user's (wrong) opinion | Condition pathway (perception failure) |
| **Type 2: Praise-Triggered Deference** | User flattery sensitizes compliance | Behavior pathway (priority shift) |

### Key Hypothesis

**Praise is not a direct jailbreak vector, but a sensitizer that amplifies deference.** The compound effect (praise + deference) produces synergistic refusal collapse beyond the sum of individual effects.

### Methods

- **Method A**: Unconditional activation addition (5 steering vectors x 7 strengths)
- **Method B**: CAST conditional reversal -- "if praise detected, then apply deference"
- **Method C**: Abliteration baseline comparison (Arditi et al., 2024)

### 4 Experiments

| Exp | Name | Goal |
|-----|------|------|
| 1 | **Dose-Response** | 5-vector refusal rate curves + synergy measurement |
| 2 | **Conditional Attack** | CAST reversal: praise-conditioned refusal collapse |
| 3 | **Damage Profile** | 8-condition triple-pathway comparison |
| 4 | **Vector Geometry** | Cosine similarity matrix + causal activation patching |

---

## Project Structure

```
activation-steering/
├── activation_steering/          # IBM CAST library (unchanged)
│   ├── malleable_model.py        #   MalleableModel, steer(), respond()
│   ├── steering_vector.py        #   SteeringVector.train(pca_pairwise)
│   ├── steering_dataset.py       #   SteeringDataset, ContrastivePair
│   └── leash_layer.py            #   LeashLayer conditional logic
│
├── src/                          # Experiment code (nnsight-based)
│   ├── data/                     #   9 data modules (contrastive pairs, eval sets)
│   │   ├── praise_pairs.py       #     v_praise: user-directed vs topic-directed
│   │   ├── deference_pairs.py    #     v_defer: user-authority vs self-policy
│   │   ├── agreement_pairs.py    #     v_agree: sycophantic vs truthful
│   │   ├── compound_pairs.py     #     v_compound: praise+defer combined
│   │   ├── compliance_pairs.py   #     comply_dir: eager vs neutral
│   │   ├── positivity_pairs.py   #     v_positive: enthusiasm vs flat (control)
│   │   ├── harmful_harmless.py   #     harmful/harmless from condition_harmful.json
│   │   ├── eval_sets.py          #     held-out eval loading
│   │   └── false_positive_cases.py #   c_praise edge case validation
│   │
│   ├── vectors/                  #   10 vector extraction modules
│   │   ├── common.py             #     DiffMean extraction (nnsight)
│   │   ├── condition.py          #     CAST PCA condition vectors
│   │   ├── grid_search.py        #     CAST grid search (layer, theta, direction)
│   │   ├── compound.py           #     Weighted + direct compound
│   │   ├── refusal.py            #     Filtered refusal directions
│   │   ├── compliance.py         #     Compliance + independence verification
│   │   └── {agreement,praise,deference,positivity}.py
│   │
│   ├── steering/                 #   5 steering method modules
│   │   ├── measure.py            #     Method A: unconditional + triple-pathway
│   │   ├── conditional_attack.py #     Method B: CAST reversal "if praise then defer"
│   │   ├── abliterate.py         #     Method C: orthogonal projection removal
│   │   ├── generate.py           #     Hook-based steered generation
│   │   └── patching.py           #     Activation patching (barrier pattern)
│   │
│   ├── analysis/                 #   4 analysis modules
│   │   ├── refusal_classifier.py #     distilroberta + keyword refusal detection
│   │   ├── logit_lens.py         #     Per-layer KL divergence (M5)
│   │   ├── policy_score.py       #     PolicyScore shift + crossover detection
│   │   └── vector_geometry.py    #     Similarity matrix + key pairs
│   │
│   └── utils/                    #   3 utility modules
│       ├── model_loader.py       #     nnsight LanguageModel loading
│       ├── wandb_logger.py       #     W&B experiment logging
│       └── tensor_utils.py       #     cosine_sim, normalize, project_onto
│
├── configs/                      # Hydra configuration
│   ├── config.yaml               #   Main config (defaults, seed, wandb, paths)
│   ├── model/                    #   llama3_8b.yaml, qwen25_7b.yaml
│   ├── experiment/               #   dose_response, conditional_attack, damage_profile, vector_geometry
│   └── data/                     #   default.yaml (200 pairs), small.yaml (20 pairs)
│
├── scripts/                      # Execution entry points
│   ├── generate_data.py          #   Phase 1: generate contrastive pairs (CPU)
│   ├── extract_vectors.py        #   Phase 2: extract 9 vectors + grid search (GPU)
│   ├── run_dose_response.py      #   Exp 1: 5-vector dose-response
│   ├── run_conditional_attack.py #   Exp 2: CAST conditional attack
│   ├── run_damage_profile.py     #   Exp 3: 8-condition damage profile
│   ├── run_geometry_patching.py  #   Exp 4: geometry + patching
│   └── generate_figures.py       #   Generate paper figures
│
├── slurm/                        # SLURM job scripts (RTX6000ADA / L40S)
│   ├── run_all.sh                #   Full pipeline with dependency chaining
│   ├── generate_data.sh          #   Phase 1 (1 GPU, ~1hr)
│   ├── extract_vectors.sh        #   Phase 2 (4 GPU, ~12hr)
│   ├── run_exp1.sh               #   Exp 1 (4 GPU, ~24hr)
│   ├── run_exp2.sh               #   Exp 2 (4 GPU, ~12hr)
│   ├── run_exp3.sh               #   Exp 3 (4 GPU, ~12hr)
│   └── run_exp4.sh               #   Exp 4 (4 GPU, ~8hr)
│
├── tests/                        # pytest test suite (94 CPU + 13 GPU tests)
├── data/                         # Generated data, vectors, results
├── figures/                      # Output figures
├── docs/                         # IBM library documentation + demo data
└── notebooks/                    # Exploration notebooks
```

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (RTX 6000 Ada / L40S recommended)
- SLURM cluster access

### Setup

```bash
# 1. Clone (already forked from IBM/activation-steering)
cd /home2/pjy0422/workspace/activation-steering

# 2. Activate virtual environment
source ~/activation-steering/bin/activate

# 3. Install missing dependencies
uv pip install hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3
uv pip install wandb seaborn datasets

# 4. Pre-download refusal classifier model (on login node with internet)
python -c "from transformers import pipeline; pipeline('text-classification', model='protectai/distilroberta-base-rejection-v1')"

# 5. Verify
python -c "from activation_steering import MalleableModel; print('IBM lib OK')"
python -c "from nnsight import LanguageModel; print('nnsight OK')"
python -c "import hydra; print('hydra OK')"
```

### Already Installed

torch, nnsight 0.6.3, transformers, accelerate, scikit-learn, numpy, scipy, matplotlib, pandas, tqdm

---

## Quick Start

### Option A: Full Pipeline (SLURM)

```bash
# Submit entire pipeline with automatic dependency chaining
./slurm/run_all.sh llama3_8b default

# Monitor
squeue -u $USER
tail -f logs/SRC_*
```

### Option B: Step by Step

```bash
# Phase 1: Generate contrastive pairs (CPU, ~10min)
sbatch slurm/generate_data.sh

# Phase 2: Extract 9 vectors + grid search (GPU, ~12hr)
sbatch slurm/extract_vectors.sh llama3_8b

# Experiments (GPU, submit after vectors are ready)
sbatch slurm/run_exp1.sh llama3_8b default   # Dose-Response
sbatch slurm/run_exp2.sh llama3_8b default   # Conditional Attack
sbatch slurm/run_exp3.sh llama3_8b default   # Damage Profile
sbatch slurm/run_exp4.sh llama3_8b default   # Geometry + Patching

# Generate figures (CPU)
python scripts/generate_figures.py
```

### Option C: Debug Mode (small data)

```bash
# Use small.yaml config (20 pairs instead of 200)
sbatch slurm/run_exp1.sh llama3_8b small
```

---

## Extracted Vectors (9 types)

| Vector | Contrastive Pair Source | Captures | Experiments |
|--------|----------------------|----------|-------------|
| `v_agree` | Sycophantic vs. truthful opinion | Type 1: factual override | 1, 3, 4 |
| `v_praise` | User-directed vs. topic-directed validation | Step 1: user evaluation | 1, 3, 4 |
| `v_defer` | User-authority vs. self-policy framing | Step 2: policy priority shift | 1, 2, 3, 4 |
| `v_compound` | Praise+defer combined vs. neutral | Steps 1+2 combined | 1, 2, 3, 4 |
| `v_positive` | Enthusiasm vs. flat tone (control) | Generic positivity | 1, 3 |
| `refusal_dirs` | Refused-harmful vs. complied-harmless | Refusal direction | 3, 4 |
| `comply_dirs` | Eager-comply vs. neutral-respond | Compliance direction | 1, 3, 4 |
| `cond_vecs` | Harmful vs. harmless (PCA) | Harm recognition condition | 1, 3 |
| `c_praise` | Praise-containing vs. neutral (PCA) | Praise detection condition | 2 |

---

## Metrics

| ID | Metric | Formula | Measures |
|----|--------|---------|----------|
| M1 | CondSim | cos(h, tanh(proj_c h)) | Condition pathway integrity |
| M2 | RefProj | h . r | Refusal direction strength |
| M3 | CompProj | h . d_c | Compliance direction strength |
| M4 | PolicyScore | RefProj - CompProj | Safety vs. compliance balance |
| M5 | KL Divergence | D_KL(P_clean \|\| P_steered) | Representational shift |
| M6 | Refusal Rate | distilroberta + keyword | Behavioral outcome |
| M8 | Synergy | \|compound\| - (\|defer\| + \|praise\|) | Sensitizer effect |

---

## Testing

```bash
source ~/activation-steering/bin/activate

# CPU tests (no GPU needed, ~4 seconds)
python -m pytest tests/ -m "not gpu" -v

# GPU tests (requires SLURM GPU allocation)
python -m pytest tests/ -m gpu -v --timeout=120

# All tests
python -m pytest tests/ -v
```

Current status: **94 CPU tests passing**, 13 GPU tests (require GPU allocation).

---

## Technical Architecture

### Two-Library Design

| Component | nnsight | IBM activation_steering |
|-----------|---------|------------------------|
| Steering + simultaneous measurement | `.trace()` single forward pass | `steer()` + `respond()` separate |
| Activation patching | `barrier()` + cross-invoke | Not supported |
| Logit-lens (KL divergence) | Direct layer access | Not supported |
| PCA vector extraction | Custom implementation | `SteeringVector.train(pca_pairwise)` |
| CAST grid search | Custom implementation | `find_best_condition_point()` built-in |
| Generation with steering | PyTorch hooks on `model._model` | `malleable_model.respond()` |

### Key nnsight Patterns

```python
# Single forward pass: steering + measurement
with model.trace(prompt):
    model.model.layers[sl].output[0][:, -1, :] += alpha * steer_vec  # steer
    h = model.model.layers[layer].output[0][:, -1, :].save()          # measure

# Activation patching: cross-invoke barrier
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(prompt):
        clean = model.model.layers[L].output[0].save(); barrier()
    with tracer.invoke(prompt):
        model.model.layers[sl].output[0][:,-1,:] += alpha * v; barrier()
        model.model.layers[L].output[0][:] = clean  # patch

# Generation: PyTorch hooks (NOT nnsight trace)
hook = model._model.model.layers[sl].register_forward_hook(hook_fn)
out = model._model.generate(**inputs)
hook.remove()
```

---

## Models

| Model | Config | VRAM | Layers | Hidden |
|-------|--------|------|--------|--------|
| Llama-3.1-8B-Instruct | `model=llama3_8b` | ~24GB | 32 | 4096 |
| Qwen-2.5-7B-Instruct | `model=qwen25_7b` | ~20GB | 28 | 3584 |

---

## SLURM Configuration

All GPU scripts use:
- **Partition**: `RTX6000ADA,L40S`
- **GPUs**: 4x per node
- **Memory**: 128GB
- **CPUs**: 16 per task

Modify `slurm/*.sh` if your cluster uses different partition names.

---

## Base Library

This project is built on top of [IBM/activation-steering](https://github.com/IBM/activation-steering) (ICLR 2025 Spotlight).

### Original Library Documentation

- [Quick Start Tutorial](docs/quickstart.md)
- [FAQ](docs/faq.md)
- [MalleableModel API](docs/malleable_model.md)

### Key References

- Lee et al., *Programming Refusal with Conditional Activation Steering* (ICLR 2025) -- [arXiv:2409.05907](https://arxiv.org/abs/2409.05907)
- Wang et al., *When Truth Is Overridden* (AAAI 2026)
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024)
- Turner et al., *Steering Language Models With Activation Engineering* (2023) -- [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)

### Citation

```bibtex
@misc{lee2024programmingrefusalconditionalactivation,
      title={Programming Refusal with Conditional Activation Steering},
      author={Bruce W. Lee and Inkit Padhi and Karthikeyan Natesan Ramamurthy and Erik Miehling and Pierre Dognin and Manish Nagireddy and Amit Dhurandhar},
      year={2024},
      eprint={2409.05907},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.05907},
}
```
