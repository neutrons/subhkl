# Notes on the matrix-free basis pursuit finder

Results that came out of debugging `MatrixFreeSparseRBFPeakFinder` (PR #15) and
that are worth stating independently of the code. Each numbered claim is either
proved or accompanied by the measurement that establishes it.

Two of these are, as far as we are aware, not standard: Theorem 1 (the scale
dimension of a Gaussian scale-space dictionary is exactly redundant under a
scale-invariant penalty) and Theorem 2 (a background floor turns the Poisson
Fisher weight into a one-shot global Lipschitz bound). The rest are known
results whose *applicability* was silently lost in this codebase, which is its
own kind of finding.

---

## 0. Setting

Let `φ_σ : R² → R` be the Gaussian of width `σ`, normalised to unit mass,
`∫ φ_σ = 1`. Index atoms by `θ = (x, σ)` in `X = Ω × [σ_min, σ_max]`, and let the
forward operator act on a nonnegative Radon measure `μ` on `X`:

    (A μ)(z) = ∫_X φ_σ(z − x) dμ(x, σ)                                      (1)

with a strictly positive additive background `b > 0`, so the model is `Aμ + b`.
The regulariser is a weighted total mass

    R_β(μ) = ∫_X σ^β dμ(x, σ),      β ∈ R                                    (2)

and the objective, for any data fidelity `D` depending on `μ` only through `Aμ`,

    J(μ) = D(Aμ + b, y) + λ R_β(μ),      μ ≥ 0.                              (3)

`β = 0` is the total-variation / Radon-measure penalty: it charges an atom for
its mass irrespective of the scale carrying it.

**Relation to the code.** `matrix_free` parameterises atoms by *peak amplitude*
rather than mass. Its kernel of width `σ` has unit peak amplitude and mass
`2πσ²`, and its penalty weight is `λ_k ∝ α σ_k^γ · sqrt(H_diag,k) ∝ σ_k^{γ+1}`.
Penalty per unit mass is therefore `∝ σ^{γ−1}`, so

    β = γ − 1,     and the Radon-measure point is γ = 1.                     (4)

Measured `λ_k/σ_k²` across the bank `σ ∈ {0.5, 1.625, 2.75, 3.875, 5}` at γ=1:
`0.458, 0.494, 0.499, 0.500, 0.500` — constant to 1% over a 10× range in `σ`,
confirming (4) numerically. The `sqrt(H_diag)` factor contributes the extra
power of `σ` because `‖φ‖₂² ∝ σ²` for these unit-amplitude kernels.

---

## 1. The scale dimension is exactly redundant at β = 0

The intuition to discard first: the problem is *not* that Radon space admits
Dirac masses. Diracs are the desired solutions, and `M(Ω)` with a **fixed**
kernel is well posed — existence follows from weak-* compactness of TV-bounded
balls, and the representer theorems give minimisers that are finite sums of
Diracs. What fails is specific to promoting `σ` to a coordinate of the
parameter space while leaving the penalty blind to it.

**Lemma 1 (cone containment).** For `σ_min ≤ σ' < σ`, set
`σ'' = sqrt(σ² − σ'²)`. Then

    φ_σ = φ_{σ'} * φ_{σ''}                                                   (5)

so `φ_σ` is an exact mixture, with unit total weight, of translates of `φ_{σ'}`.

*Proof.* The Gaussian family is a convolution semigroup in the variance
parameter (Chapman–Kolmogorov): the convolution of unit-mass Gaussians of
widths `σ'`, `σ''` is the unit-mass Gaussian of width `sqrt(σ'² + σ''²) = σ`. ∎

**Theorem 1 (exact non-identifiability of scale at β = 0).**
Let `Ω = R²` and `β = 0`. Fix `x ∈ Ω`, `a > 0` and `σ_min ≤ σ' < σ ≤ σ_max`, and
define

    μ₁ = a · δ_{(x, σ)}
    μ₂ = a · (φ_{σ''}(· − x) dy) ⊗ δ_{σ'},     σ'' = sqrt(σ² − σ'²).

Then

    (i)  A μ₁ = A μ₂                    (identical predicted data)
    (ii) R_0(μ₁) = R_0(μ₂) = a          (identical penalty)

and consequently `J(μ₁) = J(μ₂)` for every fidelity `D` and every `λ > 0`.
Hence the minimiser of `J` is not unique and its `σ`-marginal is not
identifiable.

Moreover the reachable set is independent of the scale range: for every `m > 0`

    { A μ : μ ≥ 0, R_0(μ) = m } = { φ_{σ_min} * ν : ν ≥ 0, ν(Ω) = m }        (6)

so the scale dimension contributes nothing that `σ_min` alone does not, at
identical cost. Under `β = 0` the multiscale dictionary is **exactly redundant**.

*Proof.* (i) By Lemma 1,
`A μ₂ = ∫ φ_{σ'}(· − y) · a φ_{σ''}(y − x) dy = a (φ_{σ'} * φ_{σ''})(· − x) = a φ_σ(· − x) = A μ₁`.
(ii) `R_0(μ₂) = ∫ σ'^0 dμ₂ = a ∫ φ_{σ''}(y − x) dy = a = R_0(μ₁)`.
For (6), "⊇" is immediate. For "⊆", write any `μ ≥ 0` with `R_0(μ) = m` and apply
(5) to each scale present, pushing all mass to `σ_min` while preserving both the
image under `A` and the total mass. ∎

**Corollary 1 (β = 0 is the unique tie point, and the sign of β selects the
failure mode).** With `μ₁, μ₂` as above and `σ' < σ`:

    R_β(μ₁) = a σ^β,     R_β(μ₂) = a σ'^β

so `β < 0` ⇒ the single broad atom is strictly cheaper (merging preferred);
`β > 0` ⇒ the spread is strictly cheaper (fragmentation preferred); `β = 0` ⇒
exactly tied. In the code's convention (4): `γ < 1` merges, `γ > 1` fragments,
`γ = 1` is degenerate. Both one-sided regimes are well posed in `σ`; only the
Radon point is not.

**Proposition 1 (at β = 0 the estimator has no model-order control).**
At `β = 0` the penalty is exactly tied across all representations of a given
total mass, so the minimiser is selected by the fidelity alone. For a smooth
fidelity at a well-separated optimum, adding an atom introduces `p = 4` free
parameters and reduces the negative log-likelihood by `≈ p/2 = 2` nats in
expectation under the null (the usual `χ²_p/2` heuristic underlying AIC).
Nothing opposes this, so the argmin is attained at the largest number of atoms
the discretisation permits: **the reported peak count is set by the cardinality
of the dictionary, not by the data.**

*Numerical witness.* On the 60×60 two-peak image of
`test_peak_finder_multiscale_subpixel_recovery` (broad `σ=4, amp 200` at
`(30,30)`; narrow `σ=1, amp 120` at `(34.21, 33.74)`), sweeping γ with
everything else fixed. `R(γ) = (σ_n/σ_b)^{γ−1}` is the analytic penalty ratio of
Corollary 1, `nPeak` the reported count, `err` the distance from the broad
ground truth to the nearest reported atom, `σ_max` the largest width reported:

| γ | R(γ) | nPeak | err | σ_max |
|---|---|---|---|---|
| 0.50 | 1.544 | 2 | 0.000 | 3.88 |
| 0.60 | 1.416 | 3 | 0.497 | 3.60 |
| 0.70 | 1.298 | 3 | 0.000 | 3.88 |
| 0.80 | 1.190 | 2 | 0.066 | 3.88 |
| 0.90 | 1.091 | 3 | 0.000 | 3.88 |
| **0.95** | **1.044** | **11** | **2.598** | **2.75** |
| **1.00** | **1.000** | **12** | **2.016** | **2.75** |
| 1.05 | 0.957 | 15 | 1.143 | 2.85 |
| 1.10 | 0.917 | 6 | 1.160 | 4.04 |
| 1.25 | 0.805 | 10 | 0.926 | 3.86 |

For `γ ≤ 0.90` the true `σ=4` peak is recovered as a single atom of width 3.88
with sub-pixel accuracy. Between `γ = 0.90` and `γ = 0.95` the count jumps 3 → 11,
the error jumps 0.000 → 2.598, and the recovered width collapses 3.88 → 2.75.
That is Proposition 1 switching on.

**Remark 1 (the usable regime is one-sided and bounded away from β = 0).** The
sweep degrades *before* `β = 0` is reached, which Theorem 1 alone does not
predict, and the two sides of `β = 0` fail differently:

- `β` sufficiently negative (`γ ≲ 0.90` measured): merging strictly preferred,
  single broad atom recovered, estimator usable.
- `β ≈ 0` (`γ ≈ 0.95–1.00` measured): penalty gap too small to matter,
  Proposition 1 dominates, count explodes.
- `β > 0` (`γ > 1`): fragmentation *strictly* preferred by Corollary 1 — bad for
  the opposite reason, not a mirror image of the merging regime.

So the usable region is not a punctured neighbourhood of `β = 0` but a one-sided
one, and it stops short of the origin. With finite noise, merging survives only
while the penalty gap exceeds the per-atom fit gain of Proposition 1. Since
`R = exp(β ln(σ_n/σ_b))`, for small `|β|`

    penalty gap ≈ λ · F · |β| · |ln(σ_n/σ_b)|   vs   fit gain ≈ p/2 per atom

so the boundary should sit near `|β| ≈ (p/2) / (λ F |ln(σ_n/σ_b)|)`, moving
towards the origin as peak flux `F` or regularisation `λ` grows. This predicts
the *shape* of what is observed — a threshold in `|β|`, flux-dependent — and the
measured boundary here is `β ≈ −0.1` (`γ ≈ 0.90`) with
`|ln(1.625/3.875)| = 0.869`. We have not checked the constant quantitatively
against `λ F`; doing so across several flux levels would be the natural test of
this scaling, and is not done here.

The practical reading stands regardless: `β = 0` is an *ideal* limit in the
strict sense that the estimator is unusable not only at the point but in a
noise-dependent region around it, so it cannot be approached, only avoided.

**Remark 2 (what this does and does not contradict).** Theorem 1 does not
contradict the well-posedness and representer theory for `M(Ω)`: that theory
fixes the kernel, so `X = Ω` and there is no scale coordinate to be redundant.
The observation is that those results do not extend to `M(Ω × Σ)` with `β = 0`,
and the Gaussian semigroup furnishes the explicit counterexample. Any positive
kernel family closed under mass-preserving convolution — Gaussian, Cauchy,
stable laws — inherits Theorem 1.

**Consequence for practice.** Two degeneracies are present and they are
separable:

| degeneracy | cause | removed by |
|---|---|---|
| translation | atoms confined to the pixel grid | continuous dictionary / sliding step |
| scale | penalty per mass flat in σ | `β ≠ 0` only, or fixing σ |

Super-resolution addresses the first and cannot touch the second. The clean
architecture is therefore a measure over **positions at fixed scale** — where
recovery of nonnegative sources against a totally positive kernel is available
without a minimum-separation condition — with the profile handled downstream by
the integrator, which knows positions from the lattice.

---

## 2. A background floor gives a one-shot global Lipschitz bound under Poisson

Poisson negative log-likelihood has unbounded curvature as the mean approaches
zero, which is why prox-gradient solvers for Poisson inverse problems normally
use backtracking or Barzilai–Borwein rather than a fixed step. With a strictly
positive background and nonnegative peaks that is unnecessary.

**Theorem 2.** Let `A` have nonnegative entries, let `b > 0`, and let the
feasible set be `c ≥ 0`, with `u(c) = Ac + b`. Let `W(c) = diag(w(u(c)))` be the
Fisher weight of a fidelity whose weight function `w` is nonincreasing — Poisson,
`w(u) = 1/u`, qualifies. Then for every feasible `c`

    Aᵀ W(c) A  ⪯  Aᵀ W(0) A       (Loewner order)                           (7)

and hence `λ_max(Aᵀ W(c) A) ≤ λ_max(Aᵀ W(0) A)`.

*Proof.* `A ≥ 0` entrywise and `c ≥ 0` give `Ac ≥ 0`, so `u(c) ≥ b = u(0)`
pointwise. As `w` is nonincreasing, `w(u(c)) ≤ w(u(0))` pointwise. For any `v`,

    vᵀ Aᵀ W(c) A v = Σ_i w(u(c))_i (Av)_i²  ≤  Σ_i w(u(0))_i (Av)_i² = vᵀ Aᵀ W(0) A v,

which is (7); the eigenvalue statement follows from the variational
characterisation of `λ_max`. ∎

**Corollary 2.** The step size `τ = 1 / λ_max(Aᵀ W(0) A)`, computed **once** at
initialisation, satisfies the prox-gradient descent condition at every feasible
iterate. No re-estimation, and no backtracking on the smooth part, is required.

This is what licenses the fixed 15-iteration power estimate in
`_solve_ssn_cg_global`. The hypotheses are all needed: it fails if negative
coefficients are admitted (then `u` may dip below `b`), it fails without a
positive background (`b → 0` sends `λ_max → ∞`), and it is vacuous for a
Gaussian fidelity, where `W ≡ I` is constant. The nontrivial content is exactly
the Poisson-plus-nonnegativity case, and it extends to any `w` nonincreasing in
the mean — e.g. Anscombe-type and Poisson–Gaussian weights.

---

## 3. Diagonal preconditioning overshoots by the PSF footprint area

**Proposition 2.** For a single-channel positive convolutional dictionary with
kernel `φ` and constant weight `w`, the ratio between the spectral norm and the
largest diagonal entry of `Aᵀ W A` is

    λ_max / max diag  =  ‖φ‖₁² / ‖φ‖₂²                                       (8)

because the top eigenvector of a positive convolution operator is the constant
vector, giving `λ_max = w ‖φ‖₁²`, while `diag = w ‖φ‖₂²`. For a 2D Gaussian of
unit peak amplitude, `‖φ‖₁ = 2πσ²` and `‖φ‖₂² = πσ²`, so

    λ_max / max diag  =  4π σ²,                                             (9)

i.e. **the area of the PSF footprint in pixels.**

*Measured*, single-channel, `bg = 50`, power iteration to convergence:

| σ | λ_max | max diag | ratio | 4πσ² |
|---|---|---|---|---|
| 1.0 | 0.779 | 0.05795 | 13.4 | 12.6 |
| 2.0 | 12.055 | 0.24618 | 49.0 | 50.3 |
| 3.0 | 58.150 | 0.56028 | 103.8 | 113.1 |
| 5.0 | 392.215 | 1.56554 | 250.5 | 314.2 |
| 7.0 | 1502.109 | 3.07345 | 488.7 | 615.8 |

Fitted slope of `log ratio` against `log σ` is **1.84**, against the predicted 2.
The shortfall at large `σ` is kernel truncation: these kernels are cut at `3σ`,
which removes mass from `‖φ‖₁` faster than from `‖φ‖₂`.

**Consequence.** A Jacobi/diagonal step size on a convolutional sparse-coding
problem overshoots the convergence limit by a factor growing quadratically with
the PSF width. In the five-channel bank of PR #15 the measured overshoot was
**419×**, and the line search was silently rediscovering the correct step every
iteration by halving eight to twelve times — which reads as "the optimiser is
slow", not as "the step size is wrong by two orders of magnitude". Equation (9)
gives the factor a priori.

---

## 4. Greedy → global silently invalidates the significance threshold

Not a new theorem, but a calibration failure worth recording, because the
replacement of a greedy searcher by a convex program changes the meaning of a
user-facing parameter without touching its name, units or documentation.

In the greedy predecessor, `alpha` gated **one candidate per search window**, so
it was a per-peak significance level. Solving globally tests every
`(pixel, scale)` coefficient simultaneously. The relevant null object is the
maximum of a smooth Gaussian field, not a single standardised coefficient: for
scale `σ_k` the field has correlation length `~σ_k`, so the number of
effectively independent tests is the resolution-element count
`N_k ≈ |Ω| / (2π σ_k²)`, and the maximum of `N` standard normals concentrates at
`sqrt(2 log N)`. Controlling the false-alarm rate therefore requires

    α_eff,k = max( α · (σ_k/σ_ref)^γ , sqrt(2 log N_k) ).                   (10)

This is Donoho–Johnstone's universal threshold with the standard random-field
resel correction. Two honest caveats: the universal threshold is a theorem for
orthonormal bases and this dictionary is coherent and overcomplete; and the resel
constant is convention-dependent. Both are tolerable because the count enters
inside a log and then a square root — a 4× error in `N_k` moves the threshold by
about 9%.

For the bank in these tests the floors are `4.13, 3.52, 3.21, 2.98, 2.81`
against bare `α·w = 1.00, 3.25, 5.50, 7.75, 10.0` at `α = 2`, so (10) binds on
the two finest channels — precisely where the spurious detections were.

Note (10) has the right continuum limit: as the dictionary becomes continuous
`N → ∞`, and the correct object is the expected Euler characteristic of the
excursion set, which is again a resel count. The discrete fix is the finite-grid
stand-in for the continuum threshold rather than an artefact of discretisation.

---

## 5. A counterexample worth publicising: preconditioners inside CG operators

Conjugate gradients requires a symmetric positive-definite operator. Folding a
Jacobi preconditioner `P` into the operator by left multiplication produces
`P·M`, which is **not** symmetric even when `M` and `P` are — `P` must instead be
supplied as CG's preconditioner argument, which implicitly works with the
symmetric `P^{1/2} M P^{1/2}`.

Both solves in the original code did the former. Measured asymmetry, comparing
`⟨w, Av⟩` against `⟨Aw, v⟩` on random `v, w`:

| operator | ⟨w,Av⟩ | ⟨Aw,v⟩ | relative asymmetry |
|---|---|---|---|
| SSN, as written | 132.37 | 65.51 | 5.1 × 10⁻¹ |
| debias, as written | 779.0 | 421.5 | 4.6 × 10⁻¹ |
| SSN, diagonal removed | −55.729 | −55.729 | 4.9 × 10⁻⁶ |
| debias, diagonal removed | 126.71 | 126.71 | 2.2 × 10⁻⁶ |

The instructive part is the *failure modes*, which look nothing like "CG
assumption violated": in the debiasing solve it was divergence to NaN, and
because a non-finite iterate was allowed to propagate, whole images returned no
peaks — reported downstream as a detection problem. In the SSN solve it was a
non-descent direction, absorbed by the line search into step sizes of `2⁻⁸`, and
so presented as slow convergence. Neither points at symmetry.

The diagnostic is two lines and worth running whenever an operator is passed to
a Krylov method:

```python
v, w = rng.normal(size=shape), rng.normal(size=shape)
assert abs(jnp.sum(w*op(v)) - jnp.sum(op(w)*v)) < 1e-3 * abs(jnp.sum(w*op(v)))
```

---

## 6. Status

| claim | kind | evidence |
|---|---|---|
| Thm 1 — scale dimension exactly redundant at β=0 | believed new in this form | proof + γ sweep + 0.85%/0.987 counterexample |
| Cor 1 — β=0 unique tie point, sign selects failure | corollary | proof + sweep |
| Prop 1 — no model-order control at β=0 | heuristic + measurement | count 3 → 11 across γ = 0.90 → 0.95 |
| Rmk 1 — degeneracy is a noise-dependent band | scaling argument | band `|β| ≲ 0.05–0.10` measured |
| Thm 2 — one-shot Lipschitz bound under Poisson | believed new as stated | proof |
| Prop 2 — overshoot `= ‖φ‖₁²/‖φ‖₂² = 4πσ²` | believed new as stated | proof + 5-point table, slope 1.84 |
| §4 — resel-corrected threshold | known, misapplied here | floors vs bare α |
| §5 — preconditioner breaks CG symmetry | known, instructive failure modes | asymmetry table |

The two candidates for an applied-mathematics write-up are Theorem 1 with
Corollary 1 and Remark 1 — a clean statement that scale-space dictionaries are
degenerate exactly at the Radon-measure point, with a quantified breakdown band
— and Theorem 2 with Proposition 2, which together say something concrete about
step-size selection for nonnegative Poisson deconvolution that current practice
(backtracking, diagonal preconditioning) does not exploit.

Reproduction scripts for every table are in the PR #15 discussion.

### References

- Donoho & Johnstone, *Ideal spatial adaptation by wavelet shrinkage*, Biometrika 81(3), 1994.
- Adler, *The Geometry of Random Fields*, Wiley, 1981; Worsley et al., *Human Brain Mapping* 4, 1996.
- Bredies & Pikkarainen, *Inverse problems in spaces of measures*, ESAIM COCV 19(1), 2013.
- Fisher & Jerome, *J. Approx. Theory* 13, 1975; Unser, Fageot & Ward, *SIAM Review* 59(4), 2017; Boyer et al., *SIAM J. Optim.* 29(2), 2019; Bredies & Carioni, *Calc. Var. PDE* 59, 2020.
- Candès & Fernandez-Granda, *Towards a mathematical theory of super-resolution*, CPAM 67(6), 2014.
- Schiebinger, Robeva & Recht, *Superresolution without separation*, Information and Inference 7(1), 2018; Karlin, *Total Positivity*, 1968.
- Denoyelle, Duval, Peyré & Soubies, *Inverse Problems* 36(1), 2019; Boyd, Schiebinger & Recht, *SIAM J. Optim.* 27(2), 2017.
- Donoho, *Superresolution via sparsity constraints*, SIAM J. Math. Anal. 23(5), 1992; Batenkov, Goldman & Yomdin, *Information and Inference* 10(2), 2021.
- Lindeberg, *Scale-Space Theory in Computer Vision*, Kluwer, 1994.
