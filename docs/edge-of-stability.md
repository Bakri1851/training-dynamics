# The edge-of-stability sweep

**Establishing that the escape threshold is a property of the dynamics, not of the arithmetic**

Phase 0 addendum — **open**, pre-registered 17 September 2026

| | |
|---|---|
| **Artefact** | `notebooks/deep_linear_dynamics.ipynb` (new cells), `figures/edge-of-stability.png` |
| **Device** | CPU — the matrices are 5×5 |
| **Precision** | `float32`, `float64`, and extended (mpmath, 64-bit mantissa) — the sweep is *about* precision |
| **Extra dependency** | `mpmath` — not currently in `requirements.txt`, add it |

---

## 0. How to work on this — read first

**The algorithmic code here is the learning objective. Do not write it for me.**

This applies to: the training loop variants, the escape test, the bisection, the sweep driver, the invariance experiments. For those, give me structure, function signatures, contracts, and the traps — then review what I write and tell me directly where it is wrong.

**Give freely:** matplotlib calls, markdown cells, `requirements.txt` edits, table formatting, docstrings.

If I ask directly for a solution, give it. Otherwise assume I want to write it.

**Verify numerically before asserting.** Any constant stated here should be measured with a reference implementation, not estimated. This document contains numbers I have *not* verified — see §2 — and the entire purpose of the exercise is to replace them with measured ones.

---

## 1. Context — what already exists

The deep linear network verification is complete and pushed. `notebooks/deep_linear_dynamics.ipynb` trains a two-layer deep linear network against the Saxe closed form and establishes three things: the loop implements gradient descent to machine precision (2.39e-12 against the exact discrete recursion), the residual against the continuous closed form is first-order in η (fitted slope 0.968), and predicted learning times match observed to 0.47%.

The system, unchanged for this sweep:

| | |
|---|---|
| $n = m = h$ | 5 |
| spectrum $s$ | (1.00, 0.62, 0.38, 0.24, 0.15) |
| $U, V$ | QR factors of Gaussian matrices, seeded |
| $a_0$ | `1e-6` |
| init | balanced: $W_1 = \sqrt{a_0} V^\top$, $W_2 = \sqrt{a_0} U$ |
| update | full-batch GD, both factors stepped from the same error matrix |

Since $s_{\max} = 1$: the edge of stability sits at $\eta = 1/s_{\max} = 1.0$, and the scalar-mode blow-up at $\eta = 2/s_{\max} = 2.0$.

Why those are different: the Hessian of $\tfrac12 (s - uv)^2$ at $u = v = \sqrt{s}$ has eigenvalues $2s$ and $0$, so sharpness is $2s_{\max}$ and the standard $\eta < 2/\lambda_{\max}$ criterion gives $\eta < 1/s_{\max}$. The window $1 < \eta < 2$ is bounded but non-convergent — the edge-of-stability regime.

## 2. Why this exists — a live problem in the repository

`docs/deep-linear-dynamics.md` §7 is committed and public, and it asserts these numbers in the first person:

| Claim in §7 | Status |
|---|---|
| Full 5×5 network escapes at ≈1.266 (1.2654–1.2660, seeds 0–3) | **unverified** |
| Threshold 1.2651–1.2654 across float32 / float64 / mpmath prec=64 | **unverified** |
| Threshold 1.2651–1.2652 across δ from 1e-12 to 1e-1 | **unverified** |
| Identity-init ($U = V = I$) escapes at 2.00000 | **unverified** |
| Identity-init with δ = 1e-15 drops to 1.26508 | **unverified** |
| Scalar map bisects to 2.0000001 | **unverified** |

The code that produced them is not in the repository and no longer exists. §9 of that document already names this: *"§7 currently argues it in prose from measurements that live nowhere in the repository. It belongs in the committed artefact."*

**These are not targets to reproduce.** They are unsupported claims. Whatever this sweep measures replaces them, and if the measurements disagree, §7 is corrected to the measurements — not the other way round. If the sweep is abandoned, §7's quantitative claims are stripped back to what the notebook supports.

## 3. What this has to establish

Four claims. The fourth is a positive control and it is what makes the first three mean anything — two null results on their own are just "nothing happened."

**C1 — the structure in η.** Sweep η across 0.9 → 2.1 at a fixed step budget and record the final loss. Expected: converges below 1.0; at exactly 1.0 the top mode is marginal (multiplier $1 - 2\eta s = -1$) so the loss should decay algebraically, roughly $1/k$, rather than geometrically; bounded but non-convergent between 1.0 and the escape; divergent above it.

**C2 — precision does not move the escape.** Bisect on η for blow-up at `float32`, `float64` and extended precision. That is machine ε from ~1.2e-7 to ~1.1e-19, about twelve orders of magnitude. If the threshold does not move, rounding *magnitude* is not what sets it.

**C3 — nor does the size of the perturbation.** Bisect again with a deliberate off-manifold kick of size δ, swept from 1e-12 to 1e-1. The top of that range is a 10% relative kick — some 10¹⁵ times what rounding supplies.

**C4 — being off the manifold at all is what moves it.** With $U = V = I$ the factors stay exactly diagonal in floating point, so the run carries the same rounding as every other run but *cannot* be pushed off the aligned-and-balanced manifold. It should escape at 2/s_max = 2.0. Kicked by δ = 1e-15, it should collapse to the off-manifold value.

If C2 and C3 come back flat and C4 shows the jump, the conclusion is that rounding is only the **trigger** — what it does is knock the trajectory off an invariant manifold on which the modes decouple, and the off-manifold threshold is a structural property of the full matrix dynamics.

## 4. Structure

Signatures and contracts. Bodies are mine.

```python
def train_variant(eta, seed, dtype, delta, steps, identity_init=False):
    """One run of the deep linear system with the knobs this sweep needs.

    Contract:
      - dtype controls the ENTIRE pipeline: Sigma, U, V, both factors, the
        update. Not just the factors (see trap 4).
      - delta adds an off-manifold perturbation to both factors at init,
        scaled by sqrt(a0). delta=0 means the unperturbed baseline run.
      - identity_init=True uses U = V = I instead of random orthogonal.
      - returns whatever the escape test needs, not a plot.
    """

def escaped(...) -> bool:
    """Did this run diverge within the budget?

    Contract: scale-free. Non-finite, or loss exceeding a large multiple of
    its own initial value - never an absolute cutoff (trap 1).
    """

def escape_threshold(..., lo, hi, tol) -> float:
    """Bisect on eta for the converge/escape boundary.

    Contract: assumes exactly one crossing in [lo, hi]. Verify that with a
    coarse scan before trusting it (trap 2).
    """
```

The extended-precision arm will not share `train_variant` — mpmath has no torch ops, so it needs its own 5×5 loop. Write it as a separate function with the same signature shape so the driver can treat them alike.

## 5. Traps, ordered by what they cost

1. **"Escaped" needs a definition and the threshold depends on it.** Use non-finite, or loss exceeding a large multiple of its initial value. An absolute cutoff bakes the answer in. The boundary also moves with the step budget — §7 itself notes the fourth decimal shifts. **Quote only digits that survive a doubled budget, and state the budget alongside them.**

2. **Bisection assumes monotonicity.** One crossing in the bracket. Coarse-scan first; if the converge/escape boundary is not clean, bisection converges confidently to a meaningless number.

3. **mpmath is a separate implementation, not a flag.** `mp.prec = 64` gives a 64-bit mantissa. You write the matrix multiply yourself. It is slow — budget a smaller step count for that arm and say so in the writeup rather than quietly running a different experiment.

4. **`float32` must be float32 all the way through.** Σ, U, V, the init, the update. One `float64` literal upcasts silently and you measure float64 twice under two names.

5. **The identity-init control must be asserted, not assumed.** Check the off-diagonal entries are **exactly** zero — `.abs().max() == 0`, not "small". That assertion *is* the control; without it C4 establishes nothing.

6. **The kick must actually leave the manifold.** A perturbation preserving balance will not. After kicking, assert `(W2.T @ W2 - W1 @ W1.T).abs().max()` is nonzero.

7. **Seeds move the last digits**, because U and V are random orthogonal. Report a range across seeds, never a single value.

8. **Runtime.** Bisections × 3 precisions × ~7 deltas × ~3 seeds × a long escape budget, with one arm in mpmath. Estimate before launching it; this is the step most likely to turn into an unattended hour.

## 6. Pass criteria

Fixed before the run.

**P1** — the η sweep reproduces the qualitative structure of C1: convergence below 1.0, algebraic decay at 1.0, escape above the off-manifold threshold.

**P2** — threshold ranges across `float32`, `float64` and extended precision **overlap**, with no trend in machine ε across twelve orders.

**P3** — threshold ranges across δ ∈ [1e-12, 1e-1] **overlap**, with no trend across eleven orders.

**P4** — identity-init escapes at 2/s_max to at least four decimal places, and collapses to the off-manifold value under a δ = 1e-15 kick.

P4 is the one that carries the argument. P2 and P3 are null results and mean nothing without it.

## 7. Output

One figure, `figures/edge-of-stability.png`, three panels: final loss against η with 1/s_max and 2/s_max marked; threshold against machine ε; threshold against δ. Panels 2 and 3 want the on-manifold value drawn as a reference line — the flat series is only legible against something it is flat *relative to*.

Save with `metadata={"Date": None}` so re-running does not produce byte-diffs on identical images.

**Commit the cells and the figure in the same commit.** A figure without its code is what this document exists to fix.

Then rewrite §7 of `docs/deep-linear-dynamics.md` against the measured numbers, and close the TODO in §9.

## 8. Not in scope

- Anything about why the off-manifold threshold takes the particular value it does. The gap between ~1.27 and 2 is genuinely interesting and is not this exercise.
- Deeper networks, other spectra, other initialisation scales.
- Adam or any other optimiser. That's the next milestone.

---

*Pre-registered 17 September 2026. Measured numbers and the resolution of every **unverified** row in §2 get appended when this closes.*
