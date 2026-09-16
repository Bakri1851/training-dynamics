# training-dynamics

How the choice of optimiser shapes what a network learns — and whether that effect survives as models scale.

## The question

Neural scaling laws are conventionally written

$$L(N) = E + A N^{-\alpha}$$

with the exponent $\alpha$ treated as a property of the model and the data. Training budgets orders of magnitude larger than anything measured are then set by extrapolating a fit obtained from a handful of small runs.

This repository asks whether $\alpha$ also depends on the **optimiser**. If it does, those extrapolations are conditional on a choice that is rarely varied and almost never reported.

The question is tractable without frontier compute because it concerns a *slope*, not absolute performance. Power laws can be fitted on models small enough to train on a single GPU.

## Status

**Phase 0 of 4.** Started September 2026. Nothing here is a result yet.

- [x] Environment; CUDA verified (sm_120)
- [ ] **M1** — deep linear network verified against its closed-form solution
- [ ] **M2** — Adam reimplemented from scratch, validated against `torch.optim.AdamW`
- [ ] **M3** — matched-budget optimiser comparison at a single model size
- [ ] **M4** — scaling axis; exponents fitted per optimiser with bootstrapped intervals

Each milestone has a written brief in [`docs/`](docs/) specifying its pass criteria **before** the work is done.

## Methodology

The project is as much about experimental protocol as about optimisers. Published optimiser comparisons frequently disagree, and a substantial part of that disagreement traces to unmatched tuning budgets: a more general method can emulate a more specific one given enough hyperparameter search, so the reported ranking partly measures the search budget rather than the method.

Three commitments follow.

**Pre-registration.** Search space, trials per method, seeds, metric and stopping rule are committed to `docs/` before the first run of each experiment. The commit history is the evidence.

**Matched budgets.** Every method receives the same number of tuning trials. Comparisons at unequal budgets measure the budget.

**Full distributions.** Results report the whole search distribution, not the best point. Best-of-$n$ conceals the variance the study exists to measure.

Negative and inconclusive results are reported. If confidence intervals on $\alpha$ overlap across optimisers, that is a finding about what four-point power-law fits can resolve.

## Layout

```
training-dynamics/
├── docs/                 # briefs and pre-registered protocols
├── notebooks/            # exploratory work; figures
├── src/                  # optimisers, training loop, sweep harness
├── figures/              # generated output (not committed)
├── requirements.txt
└── README.md
```

Exploratory work lives in notebooks. Anything that runs unattended or contributes to a reported result lives in `src/` as a script — a notebook cannot establish what ran in what order, and that claim is load-bearing here.

## Results

*None yet. This section will hold one entry per milestone as they land.*

## Setup

```bash
git clone https://github.com/<user>/training-dynamics
cd training-dynamics

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify the GPU build before running anything:

```python
import torch
print(torch.__version__)                     # 2.11.0+cu128
print(torch.cuda.get_device_capability())    # (12, 0) on Blackwell
```

A plain `pip install torch` returns a CPU build on Windows. The index URL is not optional.

### Environment

| | |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| GPU | RTX 5070 Laptop, 8 GB, sm_120 (Blackwell) |
| Host | Ryzen 9 270, 32 GB |

### Reproducibility

Seeds are fixed and recorded per run. Determinism follows the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html); `torch.utils.deterministic` is enabled where it does not change the algorithm under test.

Timing is measured with `torch.utils.benchmark`. Runs execute on a laptop GPU, which throttles under sustained load, so wall-clock time is reported as secondary to step counts, run order is randomised across methods, and GPU temperature and clocks are logged alongside every result.

## Background

Core references. Fuller notes are in the individual briefs.

- Saxe, McClelland & Ganguli (2014), *Exact solutions to the nonlinear dynamics of learning in deep linear networks* — the closed form M1 verifies against.
- Choi et al. (2019), *On empirical comparisons of optimizers for deep learning* — the tuning protocol determines the winner.
- Schmidt, Schneider & Hennig (2021), *Descending through a crowded valley* — fifteen optimisers under honest budgets.
- Kaplan et al. (2020) and Hoffmann et al. (2022) — the scaling laws whose exponents this work interrogates.
- Besiroglu et al. (2024) — replication finding errors in the Chinchilla fit.

## Licence

MIT.