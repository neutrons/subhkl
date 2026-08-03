# The augmented penalty: moving degeneracy from the metric to the prox

A follow-up to [`proximal_gradient_primer.md`](proximal_gradient_primer.md),
which developed the forward–backward and semismooth Newton framework of the
matrix-free finder from first principles. This note prepares the ground for a
second-order solver with *global* guarantees on the Poisson objective. It
introduces the definitions and literature needed (self-concordance, proximal
Newton, error bounds), proves what can be proved today — an exact splitting of
the Poisson likelihood, a damped-step decrease theorem valid for singular
Hessians, and global convergence without any Hessian regularization — and
states precisely the outstanding local-rate result, which is left open here.

The motivating measurement is a certificate that lied. A prototype damped
proximal Newton method (exact Hessian weight, decrement-based step control,
run on the case of §8.7 of [`matrix_free_theory.md`](matrix_free_theory.md))
reported itself converged — 28–36 of 50 steps taken at full length, final
decrement 0.11–0.15, well inside the nominal quadratic phase — while its true
first-order residual was 16–21, *worse than plain accelerated
forward–backward* (5.7–7.4) at comparable cost. Every step decreased the
objective (the safeguard never fired), so descent was never in danger; the
instrument that decides step lengths and declares convergence was broken. The
reason, and the repair, are the subject of this note.

---

## 1. Definitions

### 1.1 Self-concordance

A convex `f : dom f ⊆ Rⁿ → R`, three times differentiable with open domain,
is **standard self-concordant** (Nesterov & Nemirovskii) if along every line
`φ(t) = f(x + td)`:

    |φ'''(t)| ≤ 2 φ''(t)^{3/2}.                                             (1)

The class is closed under sums and composition with affine maps, and (1) is
affine-invariant — no norms, no constants tied to a coordinate system. Its
purpose is to make Newton's method globally analyzable: the local norm
`‖d‖_x = (dᵀ ∇²f(x) d)^{1/2}` and the pair of scalar functions

    ω(t)  = t − log(1 + t),      ω_*(t) = −t − log(1 − t)                    (2)

control `f` in a neighbourhood of each point:

    f(x + d) ≤ f(x) + ⟨∇f(x), d⟩ + ω_*(‖d‖_x)     whenever ‖d‖_x < 1.       (3)

Inequality (3) is proved one line at a time — it only ever uses (1) along the
segment `[x, x+d]` — a fact that matters below.

### 1.2 The Poisson likelihood is standard self-concordant on the counted pixels

Each fidelity term is `f_z(t) = t − y_z log t` on `t > 0`, composed with the
affine map `t = u_z(c) = (Ac)_z + b_z`. Compute:

    f_z'' = y_z/t²,   |f_z'''| = 2 y_z/t³,   2 f_z''^{3/2} = 2 y_z^{3/2}/t³,

so (1) holds **iff `y_z ≥ 1`**. Counts are integers: every term is either
*linear* (`y_z = 0`) or *standard self-concordant* (`y_z ≥ 1`). No scaling, no
generalized-self-concordance parameter tracking — but only after the `y = 0`
terms are split off, which is the pivot of this note. (For non-integer data
the term is self-concordant with parameter `2/√y_z`; the generalized framework
of Sun & Tran-Dinh covers that case, at the price of carrying the parameter
through every estimate.)

### 1.3 Proximal Newton and the decrement

For composite `F = f + g` with `f` smooth and `g` convex proximable, the
**proximal Newton** method (Lee–Sun–Saunders) solves at each iterate

    d_k = argmin_d  ⟨∇f(c_k), d⟩ + ½ dᵀ H_k d + g(c_k + d),                  (4)

with `H_k = ∇²f(c_k)`, and steps `c_{k+1} = c_k + α_k d_k`. For
self-concordant `f`, Tran-Dinh–Kyrillidis–Cevher (TKC) prove a global theory
with no line search: defining the **decrement**

    ν_k = ‖d_k‖_{H_k} = (d_kᵀ H_k d_k)^{1/2},                                (5)

the damped step `α_k = 1/(1 + ν_k)` decreases `F` by at least `ω(ν_k)` at
every iterate, and full steps (`α = 1`) converge quadratically once `ν_k` is
small. The decrement thus plays two roles: it *damps* the step and it
*certifies* convergence.

### 1.4 The seminorm problem

For the Poisson objective the exact Hessian is

    H = Aᵀ diag(y/u²) A,                                                     (6)

and every zero-count pixel contributes weight zero. If a direction `v` has its
image `Av` supported entirely on zero-count pixels, then `‖v‖_H = 0` with
`v ≠ 0`: the local "norm" (5) is only a **seminorm**. Both of the decrement's
roles fail along its null space — a step component there is neither damped nor
seen by the certificate. That is the measured failure above: on data where
most pixels are empty (a real MANDI frame averages 0.64 counts/pixel), the
null space is most of the space, and a method steering by `ν_k` is blind in
most directions. The literature's response is to regularize — replace `H` by
`H + μI` (Yue–Zhou–So; Kanzow–Lechner; Mordukhovich et al.) — which restores
a norm but injects a constant with units (§8.4 of the theory notes shows what
becomes of unit-carrying ridge constants in this problem, and Theorem 3 there
shows the *unconstrained* active-set system cannot be fixed this way at all).
This note takes the opposite route: keep the degenerate Hessian, and prove
that the problem's own structure already controls the directions it misses.

---

## 2. Lemma A: the null space is constant, and the function is linear along it

**Lemma A.** Let `f` be standard self-concordant and suppose
`dᵀ ∇²f(x₀) d = 0` for some `x₀ ∈ dom f`, `d ≠ 0`. Then `φ(t) = f(x₀ + td)`
has `φ'' ≡ 0` on the whole segment of the line inside `dom f`: `f` is exactly
affine along `d`, and `N(∇²f(x))` is the same subspace at every `x ∈ dom f`.

*Proof.* On any interval where `φ'' > 0`, (1) gives
`|d/dt (φ''^{-1/2})| = |φ'''| / (2 φ''^{3/2}) ≤ 1`. So if `φ''(t₀) > 0`
somewhere, then for every `t` between `t₀` and any other point of the domain
interval, `φ''(t)^{-1/2} ≤ φ''(t₀)^{-1/2} + |t − t₀|`, hence

    φ''(t) ≥ ( φ''(t₀)^{-1/2} + |t − t₀| )^{-2}  >  0:

a second derivative positive anywhere on the line is positive *everywhere* on
it. Contrapositively, `φ''(0) = 0` forces `φ'' ≡ 0`. Since this holds along
every line, the null space cannot rotate from point to point. ∎

(The statement is classical — degenerate self-concordant functions decompose
as a nondegenerate one composed with a linear map, plus an affine part;
Nesterov & Nemirovskii, Nesterov's *Lectures*.)

For the Poisson fidelity restricted to the counted pixels
`P = {z : y_z ≥ 1}`,

    f₊(c) = Σ_{z ∈ P} (u_z − y_z log u_z) = h(A_P c + b_P),                  (7)

with `∇²h = diag(y/u²)_P ≻ 0`: all degeneracy of `∇²f₊ = A_Pᵀ ∇²h A_P` is
exactly `N(A_P)`, **the same at every iterate**, and by Lemma A `f₊` is
*exactly linear* along it. The seminorm's blind spot is not noise; it is a
fixed subspace with known geometry, along which the smooth part of the
objective holds no information at any order.

---

## 3. The augmented splitting

Write `Z = {z : y_z = 0}`. Those pixels' fidelity terms are exactly linear on
the entire feasible set:

    Σ_{z ∈ Z} u_z = ⟨Aᵀ 1_Z, c⟩ + Σ_{z ∈ Z} b_z,                            (8)

with `a₀ := Aᵀ 1_Z ≥ 0` because the kernels are nonnegative. Fold them into
the penalty. With `g(c) = ⟨λ', c⟩ + ι_{c ≥ 0}(c)` and

    λ' = λ + a₀  >  0     (elementwise, since λ > 0, a₀ ≥ 0),               (9)

the problem `min_{c≥0} D(c) + ⟨λ, c⟩` is *identical*, up to an additive
constant, to

    min_c  F(c) = f₊(c) + g(c).                                             (10)

This is exact algebra — no approximation is introduced, controlled or
otherwise. Three things are bought:

- **Regularity.** `f₊` is standard self-concordant (§1.2), with the positive
  background keeping its domain safely away from `u = 0` (the same floor that
  gives the global Lipschitz bound, Theorem 2 of the notes).
- **Statistics.** The augmented weight has a direct reading: beyond the
  `α`-threshold, an atom pays `(a₀)_i` — *its total mass overlapping empty
  pixels*. The evidence carried by zero-count pixels moves from a gradient
  whisper into the activation threshold, where it acts exactly, and is
  strongest exactly where the data is dark.
- **Computation.** Hessian–vector products for `f₊` restrict to `P`: on
  sparse frames (`|P| ≪ |Z|`) the second-order machinery runs on the
  photon-carrying pixels only.

**Lemma B (subproblem well-posedness, no regularization).** For any `c ≥ 0`
and any PSD `H` with `N(H) = N(A_P)`, the subproblem (4) with the `g` of (9)
has a minimizer.

*Proof.* The feasible set `{d : c + d ≥ 0}` has recession cone `{v ≥ 0}`. The
subproblem objective is convex piecewise linear-quadratic; it can fail to
attain its infimum only along a feasible recession direction of nonpositive
slope (Rockafellar & Wets). Along `v ≥ 0`, `v ≠ 0`: if `v ∉ N(H)` the
quadratic term grows; if `v ∈ N(H) = N(A_P)` then, since
`∇f₊ = A_Pᵀ(1 − y/u)_P ∈ range(A_Pᵀ) ⊥ N(A_P)`, the slope is `⟨λ', v⟩ > 0`
by (9). ∎

The strict positivity `λ' > 0` — which the augmentation can only improve — is
what closes the argument. No `μI` is needed, so no unit-carrying constant
enters the subproblem.

---

## 4. Theorem C: the damped step, and the certificate repaired

**Theorem C.** Let `d` solve (4) exactly at `c` (feasible, `c ≥ 0`), with
`H = ∇²f₊(c)`, `ν = ‖d‖_H`, and let `ξ ∈ ∂g(c + d)` be the multiplier in the
optimality condition `∇f₊(c) + H d + ξ = 0`. Define the **polyhedral gap**

    δ = ⟨ξ, d⟩ − [ g(c + d) − g(c) ]  ≥ 0                                   (11)

(nonnegative by convexity of `g` and `ξ ∈ ∂g(c+d)`). Then the damped step
`c⁺ = c + α d`, `α = 1/(1 + ν)`, is feasible and

    F(c⁺) ≤ F(c) − ω(ν) − δ/(1 + ν).                                        (12)

Moreover the subproblem's model decrease is exactly

    Δ := [model at 0] − [model at d] = ½ ν² + δ,                            (13)

and `Δ = 0` if and only if `c` is a minimizer of `F`.

*Proof.* Write `g₊ = ∇f₊(c)`. From the optimality condition,
`⟨g₊, d⟩ = −ν² − ⟨ξ, d⟩`, so the linearized decrease is

    B := ⟨g₊, d⟩ + g(c + d) − g(c) = −ν² − δ.                               (14)

Feasibility: `c + αd = (1−α)c + α(c+d)` is a convex combination of feasible
points. For the smooth part, apply (3) along the segment: `φ(t) = f₊(c + td)`
has `φ''(0) = ν²`, and the one-dimensional proof of (3) needs only (1) on
this line. If `ν > 0` this gives
`f₊(c + αd) ≤ f₊(c) + α⟨g₊, d⟩ + ω_*(αν)` for `αν < 1`; if `ν = 0`, Lemma A
makes `f₊` exactly affine along `d` and the same bound holds with
`ω_*(0) = 0`. For the nonsmooth part, convexity gives
`g(c + αd) ≤ (1−α) g(c) + α g(c + d)`. Adding,

    F(c + αd) − F(c) ≤ α B + ω_*(α ν) = −α(ν² + δ) + ω_*(αν).

At `α = 1/(1+ν)`: `αν = ν/(1+ν) < 1`, and
`ω_*(ν/(1+ν)) = −ν/(1+ν) + log(1+ν)`, so the bound becomes

    −(ν² + ν + δ)/(1+ν) + log(1+ν) = −ω(ν) − δ/(1+ν),

which is (12). For (13): the model value at `d` is `B + ½ν² = −½ν² − δ` by
(14). Finally `Δ = 0` forces `ν = 0` and `δ = 0`, hence `B = 0`; then `d = 0`
is also a subproblem minimizer, i.e. `0 ∈ g₊ + ∂g(c)`, which is exactly
stationarity of the convex `F`; the converse is immediate. ∎

**Corollary (global convergence).** `F` is coercive on the cone
(`F(c) ≥ const + ⟨λ', c⟩` with `λ' > 0`), hence bounded below with compact
level sets. Summing (12), `Σ_k [ω(ν_k) + δ_k/(1+ν_k)] < ∞`, so `ν_k → 0`
**and** `δ_k → 0`, hence `Δ_k → 0` by (13); by continuity of the gap function
`c ↦ Δ(c)` (standard parametric-optimization arguments, Rockafellar & Wets),
every accumulation point of `{c_k}` is a minimizer of `F`. No regularization
of `H`, no line search, no Lipschitz constant.

### 4.1 What the repair says

Compare with the prototype's failure. The decrement `ν` sees only the
component of `d` visible to the metric; a pure null-direction move — e.g. the
deactivation of atoms sitting over empty pixels — has `ν = 0` and, in the old
accounting, produced a "full step in the quadratic phase" regardless of its
size. In the repaired accounting that same move carries its progress in `δ`:
along `N(A_P)` the model is *linear + polyhedral*, the prox resolves it
exactly (this is the active-set resolution of §8.6 of the theory notes,
reappearing inside the subproblem), the full step is genuinely correct — and
the certificate (13) counts it. The three quantities separate cleanly:

| quantity | measures | role |
|---|---|---|
| `ν_k` | curvature-visible component of the step | damping `α = 1/(1+ν)` |
| `δ_k` | polyhedrally-resolved (null + boundary) progress | none needed — exact |
| `Δ_k = ½ν_k² + δ_k` | total model decrease | **the** stopping certificate |

A solver built on this theory stops on `Δ_k`, not on `ν_k` and not on a
gradient norm.

---

## 5. Known results this connects to

- **Damped-step machinery.** Theorem C's skeleton is TKC's composite
  self-concordant analysis; the new content is that it survives a *seminorm*
  decrement once (i) Lemma A supplies exact linearity along the null space
  and (ii) the gap `δ` is carried explicitly. TKC assume the Hessian norm is
  a norm and have no `δ` term.
- **Error bounds.** The augmented form (10) is *verbatim* the structured
  class of Luo & Tseng — `h(Ax) + ⟨q, x⟩ + polyhedral`, `h` strongly convex
  with Lipschitz gradient on compact sets (`∇²h = diag(y/u²)_P` is bounded
  above and below on level sets, again by the background floor) — for which
  the Luo–Tseng error bound holds. This is the standard hypothesis from which
  local linear/superlinear rates of proximal methods are derived (Tseng's
  survey; Yue–Zhou–So).
- **Degenerate proximal Newton.** Existing globally convergent proximal
  Newton methods for singular Hessians regularize: `H + μ_k I` with `μ_k`
  tied to a residual norm (Yue–Zhou–So; Kanzow–Lechner; Mordukhovich et
  al.). The results are correct but the device is exactly the ridge whose
  unit analysis §8.4 of the theory notes rejects for this problem; Theorem C
  shows the regularizer is unnecessary for the global theory here, because
  the degeneracy is constant (Lemma A) and polyhedrally resolved (Lemma B).
- **Identification.** Finite active-set identification for proximal methods
  under polyhedral `g` and strict complementarity is classical (Hare–Lewis
  partial smoothness); it is the expected mechanism for the local phase
  below.

---

## 6. The outstanding result

**Conjecture (local rate without regularization).** Under strict
complementarity at the (unique, for `β ≠ 0`) minimizer, the exact-subproblem
iteration of Theorem C identifies the optimal active set in finitely many
steps, after which the null components of the error are resolved exactly by
the prox and the range components contract quadratically in the local norm of
`h`: the method is locally superlinear (plausibly quadratic in
`‖A_P(c_k − c*)‖`), with **no** Hessian regularization.

Why this is not a corollary of the literature: the superlinear results under
the Luo–Tseng error bound control the null space with the regularizer `μ_k I`
— remove it and their proofs lose the handle on exactly the directions Lemma
A describes. The natural repair is to quotient by `N(A_P)` and run the
self-concordant quadratic-phase argument in the range factor only; the
obstruction is that `g` is polyhedral in `c`, *not* in `A_P c`, so the
quotient does not act cleanly on the nonsmooth part. Making that argument
rigorous — or finding the counterexample that shows the rate degrades — is
the open task. Two further obligations for an implementable theorem:

1. **Inexact subproblems.** Theorem C assumes exact `d`. The practical
   variant accepts `d̂` with model decrease `Δ̂ ≥ ρ Δ` for fixed
   `ρ ∈ (0, 1]`; the decrease bound (12) degrades by controlled factors and
   the global corollary survives — this is mechanical and should be written
   out with the constants.
2. **Literature diligence.** Before any novelty claim: Sun & Tran-Dinh's
   generalized self-concordance (their treatment of Poisson-type losses may
   subsume §1.2's integer-counts observation), Bach's self-concordant GLM
   analysis, and the recent degenerate/inexact proximal Newton literature.
   The assessment above — assembly publishable as a note, the
   no-regularization local rate genuinely new — is a prior, not a verdict.

### References

- Nesterov & Nemirovskii, *Interior-Point Polynomial Algorithms in Convex
  Programming*, SIAM, 1994. Nesterov, *Lectures on Convex Optimization*,
  2nd ed., Springer, 2018.
- Lee, Sun & Saunders, *Proximal Newton-type methods for minimizing composite
  functions*, SIAM J. Optim. 24(3), 2014.
- Tran-Dinh, Kyrillidis & Cevher, *Composite self-concordant minimization*,
  JMLR 16, 2015.
- Sun & Tran-Dinh, *Generalized self-concordant functions: a recipe for
  Newton-type methods*, Math. Program. 178, 2019.
- Bach, *Self-concordant analysis for logistic regression*, Electron. J.
  Stat. 4, 2010.
- Luo & Tseng, *On the linear convergence of descent methods for convex
  essentially smooth minimization*, SIAM J. Control Optim. 30(2), 1992; *Error
  bounds and convergence analysis of feasible descent methods*, Ann. Oper.
  Res. 46, 1993. Tseng, *Approximation accuracy, gradient methods, and error
  bound for structured convex optimization*, Math. Program. 125, 2010.
- Yue, Zhou & So, *A family of inexact SQA methods for non-smooth convex
  minimization with provable convergence guarantees based on the Luo–Tseng
  error bound property*, Math. Program. 174, 2019.
- Kanzow & Lechner, *Globalized inexact proximal Newton-type methods for
  nonconvex composite functions*, Comput. Optim. Appl. 78, 2021.
- Mordukhovich, Yuan, Zeng & Zhang, *A globally convergent proximal
  Newton-type method in nonsmooth convex optimization*, Math. Program. 198,
  2023.
- Hare & Lewis, *Identifying active constraints via partial smoothness and
  prox-regularity*, J. Convex Anal. 11(2), 2004.
- Rockafellar & Wets, *Variational Analysis*, Springer, 1998.
