# Deep Linear Network Dynamics

**Verifying gradient descent against an exact solution**

Phase 0, Milestone 1 — in progress

| | |
|---|---|
| **Artefact** | `notebooks/deep_linear_dynamics.ipynb` |
| **Device** | CPU — the matrices are 5×5 |
| **Precision** | `float64` |
| **Runtime** | seconds |

Protocol and pass criteria below are fixed before the run. The reference numbers in §6 come from a `float64` NumPy prototype of the same system; I replace them with my own measurements once the notebook lands.

---

## 1. What I'm establishing

Most introductory deep learning exercises end in a quantity that cannot be independently checked: the loss decreased, the accuracy looks plausible, and you move on. I wanted one that ends in a curve I can compare against a formula to twelve significant figures.

The deliverable is a notebook establishing three things:

1. that my training loop implements gradient descent correctly, verified to machine precision;
2. that the resulting trajectory matches the analytical solution of the corresponding gradient flow to $O(\eta)$;
3. that this error is first-order in the step size, verified by a measured convergence slope.

## 2. Why a linear network

Delete every nonlinearity from a deep network and you get $f(x) = W_2 W_1 x$. As a *function* that is a linear map and expressively worthless. As a *dynamical system on the weights* it is not trivial at all: the gradient of each factor contains the other factor, so the dynamics are coupled and nonlinear even though the function is linear. The nonlinearity comes from depth, not from activations.

The system still reproduces the qualitative phenomena that matter — sigmoidal learning curves, long plateaus punctuated by rapid transitions, and a definite order in which structure is acquired — while remaining exactly solvable.

> This is the same bet I made with Kuramoto. Nobody thinks 128 identical phase oscillators are a real firefly swarm. You study them because the system is simple enough to admit exact results about a transition, and those results survive into messier settings.

## 3. Setup

### 3.1 Loss

I work directly on the population loss rather than a dataset. That is not a shortcut — it is the whitened-input case with the data marginalised out, which is exactly the regime the closed form describes.

$$L(W_1, W_2) = \tfrac{1}{2}\left\| \Sigma - W_2 W_1 \right\|_F^2, \qquad \Sigma = U S V^{\top}$$

| Quantity | Value | Note |
|---|---|---|
| $n = m = h$ | 5 | square, no bottleneck |
| $s$ | (1.00, 0.62, 0.38, 0.24, 0.15) | well separated by design |
| $U, V$ | QR factors of Gaussian matrices | random orthogonal |
| $a_0$ | `1e-6` | see §3.2 — not an arbitrary choice |
| $\eta$ | 0.01 | principal run |
| steps | 6000 | to convergence of the slowest mode |
| optimiser | full-batch GD | no momentum, no adaptivity |

### 3.2 Why I set a₀ = 10⁻⁶ and not something rounder

The staircase only appears if consecutive transitions are separated by more than their own width. Mode $i$ rises over roughly $2/(\eta s_i)$ steps, and the gap between consecutive transitions grows like $\ln(1/a_0)$, so the ratio that matters scales as $\ln(1/a_0)$.

| $a_0$ | min gap / width | staircase |
|---|---|---|
| `1e-3` | 0.63 | transitions overlap — loss looks like a smooth decay |
| `1e-4` | 0.98 | marginal |
| `1e-5` | 1.32 | steps visible but soft |
| `1e-6` | **1.67** | resolved |

> **The trap.** At $a_0 = 10^{-3}$ the run converges faster and everything else in this protocol still holds — but the loss curve falls smoothly and the staircase, which is the phenomenon the exercise exists to show, never appears. A plausible-looking default silently removes the result. This is why the value is pinned here rather than left to taste.

### 3.3 Balanced initialisation

$$W_1 = \sqrt{a_0}\, V^{\top}, \qquad W_2 = \sqrt{a_0}\, U$$

Then $W_2 W_1 = a_0 U V^{\top}$, so every mode starts at strength $a_0$, and $W_2^{\top} W_2 = W_1 W_1^{\top} = a_0 I$.

> **The trap that costs an evening.** Initialise randomly instead of aligned-and-balanced and the modes do not decouple. The closed form will not fit, the curves will look vaguely sigmoidal but wrong, and I'd spend hours checking a training loop that was correct all along. If Criterion 1 fails, the initialisation gets checked before anything else.

## 4. The closed form

I derive this before coding it. It is four lines and it is the part that transfers.

1. Decompose $\Sigma = U S V^{\top}$. With aligned balanced init, write $W_1 = B V^{\top}$ and $W_2 = U A$ with $A, B$ diagonal — this structure is preserved under gradient flow.
2. Let $u_i = A_{ii}$, $v_i = B_{ii}$, and $a_i = u_i v_i$. The loss separates: $L = \tfrac{1}{2}\sum_i (s_i - a_i)^2$.
3. Gradient flow gives $\dot u_i = \eta(s_i - a_i)v_i$ and $\dot v_i = \eta(s_i - a_i)u_i$. Balance means $u_i = v_i = \sqrt{a_i}$ for all time.
4. So $\dot a_i = 2u_i \dot u_i = 2\eta\, a_i (s_i - a_i)$ — the logistic equation, carrying capacity $s_i$, growth rate $2\eta s_i$.

$$a_i(k) = \frac{s_i}{1 + \left(\dfrac{s_i}{a_0} - 1\right) e^{-2\eta s_i k}} \tag{1}$$

$$k_i = \frac{\ln(s_i / a_0)}{2\eta s_i} \;\propto\; \frac{1}{s_i} \tag{2}$$

Equation (2) is the step at which mode $i$ reaches half strength, and it is the precise sense in which **stronger modes are learned first**. The $\ln(1/a_0)$ dependence is the sense in which smaller initialisation lengthens the plateaus.

Three sanity checks before any code runs: $a_i(0) = a_0$, $a_i(\infty) = s_i$, and $k_i \propto 1/s_i$.

## 5. Discrete dynamics and the Euler correspondence

Equation (1) solves the gradient **flow**. My loop runs gradient **descent**. These are not the same trajectory, and the gap between them is the substance of this milestone rather than an inconvenience.

One step of gradient descent under balanced init maps:

$$a_{k+1} = a_k\big(1 + \eta(s - a_k)\big)^2 = \underbrace{a_k + 2\eta a_k(s - a_k)}_{\text{forward-Euler step on (1)}} + \underbrace{a_k \eta^2 (s - a_k)^2}_{O(\eta^2)\ \text{per step}} \tag{3}$$

*Proof.* The update is $u \leftarrow u - \eta(uv - s)v = u + \eta(s - a)v$. Under balance $u = v = \sqrt{a}$, so $u_{k+1} = u_k(1 + \eta(s - a_k))$ and $a_{k+1} = u_{k+1}^2$ gives the stated form. ∎

Equation (3) is *exactly* what gradient descent does here, so it is my reference for checking the implementation. Equation (1) is my reference for checking the approximation. Global error of the latter is $O(\eta)$.

## 6. How I verify it — two stages

This is the part worth more than the plot. I'm separating **implementation error** from **discretisation error**, and they need different tests.

### Criterion 1 — is the code right?

Compare the trajectory against the exact discrete recursion (3).

> **Pass:** max relative error < `1e-10`. The prototype sits at 3–6e-12 across seeds. The band is looser than you might expect because the trajectory runs 6000 steps and rounding accumulates. An error near `1e-3` means the code is wrong — most likely the initialisation is not balanced, or $W_1$ was updated using the already-updated $W_2$ instead of the old one.

### Criterion 2 — how good is the continuous approximation?

Compare against the closed form (1). It will **not** match to machine precision, and that is not a bug.

Reference values from the prototype:

| $\eta$ | steps | vs closed form (1) | vs recursion (3) |
|---|---|---|---|
| 0.05 | 1200 | 2.2e-1 | 2.8e-12 |
| 0.02 | 3000 | 9.7e-2 | 3.4e-12 |
| 0.01 | 6000 | 5.0e-2 | 3.6e-12 |
| 0.005 | 12000 | 2.5e-2 | 3.8e-12 |
| 0.002 | 30000 | 1.0e-2 | 4.7e-12 |
| 0.001 | 60000 | 5.1e-3 | 3.9e-12 |

The left column is discretisation error and scales as $O(\eta)$. The right column is implementation error and sits at machine precision throughout.

> **Pass:** error falls linearly in $\eta$. Plot max error against $\eta$ on log-log axes and fit the slope — it should come out at 1.0 to within a few percent. On the reference run, a 50× reduction in $\eta$ gave a 44× reduction in error, fitted slope **0.968**.

That third plot is what turns this from "I followed a tutorial" into "I verified a first-order convergence result."

### Why pointwise error alone is a poor criterion

Mid-transition, a small timing offset reads as a large pointwise error. At $\eta = 0.05$ the worst point is step 139, mode 1: empirical 0.4479 against analytical 0.5211. Same curve shape, shifted in time.

So I compare learning times instead — the robust quantity:

| mode | $s_i$ | $k_i$ predicted | $k_i$ reference | error |
|---|---|---|---|---|
| 1 | 1.00 | 690.8 | 694 | 0.47% |
| 2 | 0.62 | 1075.6 | 1079 | 0.32% |
| 3 | 0.38 | 1690.5 | 1694 | 0.21% |
| 4 | 0.24 | 2580.9 | 2584 | 0.12% |
| 5 | 0.15 | 3972.8 | 3976 | 0.08% |

## 7. How I'm building it

1. **Build the problem.** Random orthogonal $U, V$ via `torch.linalg.qr` on a Gaussian. Assemble $\Sigma$. Set the balanced init. Everything in `float64`.
2. **Write the loop by hand.** No `torch.optim`. Compute $E = W_2 W_1 - \Sigma$, then $\nabla_{W_2} = E W_1^{\top}$ and $\nabla_{W_1} = W_2^{\top} E$, and step manually. Cross-check one step against `autograd` to confirm the gradients.
3. **Record each step.** `torch.linalg.svdvals(W2 @ W1)` and the loss.
4. **Compute both references.** Equation (1), and the recursion `a = a * (1 + eta * (s - a))**2` from $a_0$.
5. **Produce the four outputs** below.

### Pitfalls I'm watching for

- **`float32` will muddy it.** The system is 5×5; there is no cost to `float64`, and fp32 noise sits at the same order as the effect at small $\eta$.
- **Run on CPU.** Device transfer dominates computation at this size. This milestone does not touch CUDA at all.
- **Update both factors from the same $E$.** Stepping $W_2$ first and using it to compute $\nabla_{W_1}$ is a different algorithm and fails Criterion 1.
- **`svdvals` returns sorted values.** Mode identity can permute if two singular values cross. This spectrum avoids it, but the hazard is general.
- **For $\eta > 2/s_{\max}$ it diverges.** Worth triggering once deliberately — that is the edge-of-stability threshold appearing in my own code.

## 8. Outputs

| # | Output | Establishes |
|---|---|---|
| 1 | Mode strengths vs step, closed form overlaid | sequential learning, strong modes first |
| 2 | Loss vs step | the staircase, one step per mode |
| 3 | Max error vs $\eta$, log-log, fitted slope | first-order convergence |
| 4 | Predicted vs observed $k_i$ | $k_i \propto 1/s_i$ quantitatively |

Outputs 1 and 2 are the exercise. **Output 3 is the one worth showing someone** — it is a measured convergence order, not a screenshot of a loss curve.

## 9. Where this goes next

Only after the four outputs are done. Each points at a later phase.

- **Sweep $a_0$** across orders of magnitude and measure the gap between the first two transitions relative to their width. Small $a_0$ gives a clean staircase; large $a_0$ and the modes move together. That is the lazy-to-rich transition, measured.
- **Add a third factor.** The exponent in the dynamics changes. I want to predict the effect on transition sharpness before running it.
- **Swap GD for Adam** and watch the mode ordering change. First hint of the Phase 2 question: the optimiser does not just change the speed, it changes what gets learned first.

---

*Reference numbers above are from a `float64` NumPy prototype, seed 0, spectrum (1.00, 0.62, 0.38, 0.24, 0.15), $a_0 = 10^{-6}$; loss runs 0.804 → 5.8e-8 over 6000 steps at $\eta = 0.01$. My own values should be close but need not be identical — the random orthogonal factors differ by seed.*
