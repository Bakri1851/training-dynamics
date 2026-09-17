# training-dynamics

I'm investigating how the choice of optimiser shapes what a network learns — and whether that effect survives as models scale.

## The question

Neural scaling laws are conventionally written

$$L(N) = E + A N^{-\alpha}$$

with the exponent $\alpha$ treated as a property of the model and the data. Training budgets orders of magnitude larger than anything measured are then set by extrapolating a fit obtained from a handful of small runs.

I want to know whether $\alpha$ also depends on the **optimiser**. If it does, those extrapolations are conditional on a choice that is rarely varied and almost never reported.

I picked this question because it's tractable without frontier compute: it concerns a *slope*, not absolute performance. Power laws can be fitted on models small enough to train on a single GPU — which is what I have.

## Status

**Phase 0 of 4.** Started September 2026. M1 is done; nothing beyond it is.

- [x] Environment; CUDA verified (sm_120)
- [x] **M1** — deep linear network verified against its closed-form solution
- [ ] **M2** — Adam reimplemented from scratch, validated against `torch.optim.AdamW`
- [ ] **M3** — matched-budget optimiser comparison at a single model size
- [ ] **M4** — scaling axis; exponents fitted per optimiser with bootstrapped intervals

M1 checks that my training loop does what I think it does. It says nothing about the question above — that starts at M3.

I write each milestone's protocol into [`docs/`](docs/) and fix its pass criteria **before** doing the work.

## How I'm running it

This project is as much about experimental protocol as about optimisers. Published optimiser comparisons frequently disagree, and a substantial part of that disagreement traces to unmatched tuning budgets: a more general method can emulate a more specific one given enough hyperparameter search, so the reported ranking partly measures the search budget rather than the method.

Three commitments I'm holding myself to.

**Pre-registration.** I commit the search space, trials per method, seeds, metric and stopping rule to `docs/` before the first run of each experiment. The commit history is the evidence.

**Matched budgets.** Every method gets the same number of tuning trials. Comparisons at unequal budgets measure the budget.

**Full distributions.** I report the whole search distribution, not the best point. Best-of-$n$ conceals the variance this study exists to measure.

I'll report negative and inconclusive results. If confidence intervals on $\alpha$ overlap across optimisers, that is a finding about what four-point power-law fits can resolve.

## Layout

```
training-dynamics/
├── docs/                 # protocols, pre-registered before each run
├── notebooks/            # exploratory work; figures
├── src/                  # optimisers, training loop, sweep harness — empty until Phase 1
├── figures/              # generated output; the three M1 figures are committed
├── requirements.txt
└── README.md
```

I keep exploratory work in notebooks. M1 is exploratory and lives in one: it is a verification exercise I worked through interactively, and the notebook is re-executed top to bottom so the execution counts are monotonic. From Phase 1 the code moves into `src/` as scripts — once runs are unattended or feed a reported comparison, a notebook cannot establish what ran in what order, and that claim is load-bearing here.

## Results

**M1 — deep linear network against its closed-form solution.** Passed 17 September 2026. Protocol, derivation and full numbers: [`docs/m1-deep-linear-dynamics.md`](docs/m1-deep-linear-dynamics.md).

| Check | Criterion | Measured |
|---|---|---|
| C1 — vs the exact discrete recursion | < 1e-10 | **2.39e-12** |
| C2 — fitted log–log convergence slope | 1.0 ± a few % | **0.968** |
| Learning times vs prediction | < 1% | **0.47%** worst case |
| Manual gradients vs autograd | — | **5.42e-20** |

Three figures: [`figures/m1_modes.png`](figures/m1_modes.png) — modes learned sequentially, strongest first; [`figures/m1_loss.png`](figures/m1_loss.png) — the loss staircase, one step per mode; [`figures/m1_convergence.png`](figures/m1_convergence.png) — first-order convergence in $\eta$.

What it establishes: the training loop implements gradient descent exactly, so its departure from the continuous closed form is discretisation error rather than a bug, and that error is first-order in the step size — measured, not asserted. This is a verification result about my code. It is not yet evidence about optimisers.

## Setup

```bash
git clone https://github.com/Bakri1851/training-dynamics
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

A plain `pip install torch` returns a CPU build on Windows — I lost time to this. The index URL is not optional.

### Environment

| | |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| GPU | RTX 5070 Laptop, 8 GB, sm_120 (Blackwell) |
| Host | Ryzen 9 270, 32 GB |

### Reproducibility

I fix seeds and record them per run. Determinism follows the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html); I enable `torch.utils.deterministic` where it does not change the algorithm under test.

I measure timing with `torch.utils.benchmark`. Runs execute on a laptop GPU, which throttles under sustained load, so I report wall-clock time as secondary to step counts, randomise run order across methods, and log GPU temperature and clocks alongside every result.

## Background

The core references I'm working from. Fuller notes live in the individual milestone docs.

- Saxe, McClelland & Ganguli (2014), *Exact solutions to the nonlinear dynamics of learning in deep linear networks* — the closed form M1 verifies against.
- Choi et al. (2019), *On empirical comparisons of optimizers for deep learning* — the tuning protocol determines the winner.
- Schmidt, Schneider & Hennig (2021), *Descending through a crowded valley* — fifteen optimisers under honest budgets.
- Kaplan et al. (2020) and Hoffmann et al. (2022) — the scaling laws whose exponents this work interrogates.
- Besiroglu et al. (2024) — replication finding errors in the Chinchilla fit.

## Licence

MIT — see [`LICENSE`](LICENSE).
