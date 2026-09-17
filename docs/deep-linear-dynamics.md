# Deep Linear Network Dynamics

**Verifying gradient descent against an exact solution**

Phase 0, Milestone 1 — **complete, 17 September 2026**

| | |
|---|---|
| **Artefact** | `notebooks/deep_linear_dynamics.ipynb` |
| **Device** | CPU — the matrices are 5×5 |
| **Precision** | `float64` |
| **Runtime** | seconds |

Protocol and pass criteria below were fixed before the run. The numbers in §6 are now my own measurements from the notebook, seed 0; they replace the `float64` NumPy prototype figures this document originally carried.

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

> **Pass:** max relative error < `1e-10`. **Measured: 2.4e-12** on the principal run, and 2.4e-12 – 1.3e-11 across seeds 0–5. The band is looser than you might expect because the trajectory runs 6000 steps and rounding accumulates. An error near `1e-3` means the code is wrong — most likely the initialisation is not balanced, or $W_1$ was updated using the already-updated $W_2$ instead of the old one.

### Criterion 2 — how good is the continuous approximation?

Compare against the closed form (1). It will **not** match to machine precision, and that is not a bug.

Measured, seed 0:

| $\eta$ | steps | vs closed form (1) | vs recursion (3) |
|---|---|---|---|
| 0.05 | 1200 | 2.22e-1 | 2.37e-12 |
| 0.02 | 3000 | 9.67e-2 | 3.52e-12 |
| 0.01 | 6000 | 4.97e-2 | 2.39e-12 |
| 0.005 | 12000 | 2.52e-2 | 3.40e-12 |
| 0.002 | 30000 | 1.02e-2 | 2.99e-12 |
| 0.001 | 60000 | 5.11e-3 | 3.64e-12 |

The left column is discretisation error and scales as $O(\eta)$. The right column is implementation error and sits at machine precision throughout.

> **Pass:** error falls linearly in $\eta$. Plot max error against $\eta$ on log-log axes and fit the slope — it should come out at 1.0 to within a few percent. Measured: a 50× reduction in $\eta$ gave a 43.6× reduction in error, fitted slope **0.968**.

That third plot is what turns this from "I followed a tutorial" into "I verified a first-order convergence result."

### Why pointwise error alone is a poor criterion

Mid-transition, a small timing offset reads as a large pointwise error. At $\eta = 0.05$ the worst point is step 114, mode 1: empirical 0.0638 against analytical 0.0820. Same curve shape, shifted in time.

So I compare learning times instead — the robust quantity. Measured at $\eta = 0.01$, seed 0:

| mode | $s_i$ | $k_i$ predicted | $k_i$ observed | lag (steps) | lag predicted | error |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 690.8 | 694 | 3.2 | 3.45 | 0.47% |
| 2 | 0.62 | 1075.6 | 1079 | 3.4 | 3.33 | 0.32% |
| 3 | 0.38 | 1690.5 | 1694 | 3.5 | 3.21 | 0.21% |
| 4 | 0.24 | 2580.9 | 2584 | 3.1 | 3.10 | 0.12% |
| 5 | 0.15 | 3972.8 | 3976 | 3.2 | 2.98 | 0.08% |

**The lag is the quantity that means something here; the percentage is not.** The offset between observed and predicted crossing is ≈3.2 steps at *every* mode, and it does not move with $\eta$ — the mode-1 lag measures 3.8, 3.6, 3.2, 3.4, 3.1, 3.2 at $\eta$ = 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, across a 50× range. The percentage varies 5.8× across modes only because $k_i \propto 1/s_i$, and falls with $\eta$ only because $k_i \propto 1/\eta$. In both cases it is the denominator moving, not the discrepancy.

Rescaling time by $\ln(1 + \eta s_i)/(\eta s_i)$ in the logistic solution gives a lag of $\ln(s_i/a_0)/4$ to leading order, independent of $\eta$ — the `lag predicted` column.

> **Caveat.** `argmax` on the boolean crossing quantises to whole steps and biases the observed lag low by up to 1, so this is an order-of-magnitude agreement rather than a tight test. Interpolating the crossing would sharpen it — a stated next step, not done here.

I keep the percentage column because the pass criterion is defined on its maximum.

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
- **Two thresholds, and I had them conflated.** $2/s_{\max}$ is the blow-up threshold. It is *not* the edge of stability, which sits at $1/s_{\max}$ — half of it. With $s_{\max} = 1$ the two land at $\eta = 1$ and $\eta = 2$. As measured:

  - **$\eta < 1/s_{\max}$ — converges.** At $\eta = 0.995$ the final loss is machine zero. I originally quoted it as 2.4e-29, which was two significant figures of nothing: at this level the number is the square of accumulated rounding, and it moves over several orders of magnitude between seeds and between implementations of the same problem. It is not a measurement and should not be written as one.
  - **$\eta = 1/s_{\max} = 1.0$ — the fixed point loses stability.** The iterate stays bounded but never converges. The top mode is exactly marginal here (multiplier $1 - 2\eta s = -1$), so the loss decays algebraically rather than geometrically: 2.1e-5 at 6000 steps, 6.3e-6 at 20000 — a factor 3.3 for 3.3× the steps, i.e. $\sim 1/k$. Just above, it is genuinely stuck: 1.1e-2 at $\eta = 1.02$ at both 6000 and 20000 steps.
  - **$\eta \geq 2/s_{\max} = 2.0$ — escapes to infinity.** Bisection on the single-mode map $a \leftarrow a(1 + \eta(s - a))^2$ puts the boundary at 2.0000001, and it lands there from every starting point I tried ($a_0 = 10^{-6}$, $a = 0.5$, and the fixed point perturbed by $10^{-9}$ and $10^{-12}$).

  Why $1/s_{\max}$ is the edge of stability: the Hessian of $\tfrac{1}{2}(s - uv)^2$ at $u = v = \sqrt{s}$ is $\begin{bmatrix} s & s \\ s & s \end{bmatrix}$, with eigenvalues $2s$ and $0$. So the sharpness is $\lambda_{\max} = 2 s_{\max}$, and the standard $\eta < 2/\lambda_{\max}$ criterion gives $\eta < 1/s_{\max}$. The window $1 < \eta < 2$ — non-convergent but bounded — is the edge-of-stability regime.

  **The full network escapes earlier than the scalar map does, and I should not have assumed otherwise.** The 5×5 run blows up at $\eta \approx 1.266$ (bisection; 1.2654–1.2660 across seeds 0–3), not at 2.0. The clean $\eta = 2$ boundary is a property of the scalar mode map.

  **My first explanation of *why* was also wrong, and it is worth recording what it got wrong.** I wrote that above $\eta = 1$ the dynamics are chaotic, so rounding at $10^{-16}$ is amplified until the trajectory leaves the bounded attractor. That makes the threshold an artefact of rounding *magnitude*, and it carries a prediction: coarser arithmetic should escape earlier. It doesn't. Bisecting on $\eta$ for blow-up in the full matrix loop:

  - **Precision does not move it.** Across `float32`, `float64` and 64-bit-mantissa extended arithmetic — machine $\varepsilon$ from `1.2e-7` to `1.1e-19`, twelve orders of magnitude — the threshold is 1.2651–1.2654 for seeds 0–2, with no trend in $\varepsilon$ at all.
  - **Nor does the size of a deliberate kick.** Adding `delta * sqrt(a0) * randn(N, N)` to both factors gives 1.2651–1.2652 for every `delta` from `1e-12` to `1e-1` — eleven orders of magnitude, the top of which is a 10% relative kick, some $10^{15}$ times what rounding supplies.
  - **What *does* move it is whether the trajectory is on the manifold at all.** With $U = V = I$ the factors stay exactly diagonal in `float64` — the off-diagonal entries are exact zeros, which I checked rather than assumed — so the run carries the same rounding as every other run here but cannot be pushed off the aligned-and-balanced manifold. It escapes at 2.00000. Kick it by `delta = 1e-15` and it drops to 1.26508.

  So rounding is only the **trigger**. What it does is knock the trajectory off the exactly-aligned, exactly-balanced manifold on which the modes decouple. That manifold is invariant, so in exact arithmetic the decoupling would stay exact and the escape would sit at $2/s_{\max}$ — which is precisely where the identity-init run above puts it, and where the scalar map puts it. Once the trajectory is off that manifold, $\eta \approx 1.266$ is a property of the full matrix dynamics in the off-manifold directions. It is structural rather than numerical, and independent of both the size and the source of whatever knocked it there; rounding is simply what happens to do it here.

  *(These bisections run a longer escape budget than the figure quoted at the top of this bullet and resolve the boundary about `5e-4` lower — the fourth decimal moves with the step budget, so it is the invariance that carries the argument, not the digits.)*

  The gap between 1.27 and 2 is still the part worth understanding.

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

**TODO —** Sweep η across 0.9 → 2.1 and plot final loss, showing the convergence boundary at η = 1/s_max and the escape at η = 2/s_max — the edge-of-stability window in this system. Fifth output for M1. *(See §7: the 5×5 run actually escapes at ≈1.27, short of 2/s_max. The sweep is where that shows up directly.)*

The sweep should also carry the invariance, not just the threshold: repeat it at two or three precisions and two or three off-manifold perturbation sizes, and show the escape landing in the same place every time. That is the part that distinguishes a property of the system from a property of the arithmetic, and §7 currently argues it in prose from measurements that live nowhere in the repository. It belongs in the committed artefact.

---

*Numbers above are measured from `notebooks/deep_linear_dynamics.ipynb`, seed 0, spectrum (1.00, 0.62, 0.38, 0.24, 0.15), $a_0 = 10^{-6}$; loss runs 0.804 → 5.8e-8 over 6000 steps at $\eta = 0.01$. They replace the NumPy prototype figures this document carried while the milestone was open. Values at other seeds differ in the last digits — the random orthogonal factors depend on the seed — so the machine-precision quantities are quoted as ranges where that matters.*
