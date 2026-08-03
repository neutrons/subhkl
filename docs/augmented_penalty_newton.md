# The augmented penalty: moving degeneracy from the metric to the prox

A follow-up to [`proximal_gradient_primer.md`](proximal_gradient_primer.md),
which developed the forward–backward and semismooth Newton framework of the
matrix-free finder from first principles. This note prepares the ground for a
second-order solver with *global* guarantees on the Poisson objective. It
introduces the definitions and literature needed (self-concordance, proximal
Newton, error bounds) and proves: an exact splitting of the Poisson
likelihood, a damped-step decrease theorem valid for singular Hessians,
global convergence without any Hessian regularization (§4), and — under
uniqueness and strict complementarity — finite active-set identification and
local quadratic convergence (§6, with the mechanical obligations that remain
flagged in §6.3). The instructive dead end is also recorded: the natural
quotient-space proof of the local rate fails, for a reason that is itself a
result about this problem class (§6.1).

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
  *Novelty caveat (2026):* superlinear convergence of proximal Newton-like
  methods **to degenerate solutions** is now an active topic — arXiv
  2602.10470 proves Q-superlinear rates under a Hölderian error bound for
  inexact proximal Newton with singular, non-Lipschitz Jacobians, and arXiv
  2607.12551 treats degenerate polyhedral projection by exploiting dual
  nonuniqueness rather than regularizing. Neither uses self-concordance,
  derives face-nondegeneracy from estimator uniqueness (Lemma D), or has
  the `Δ = ½ν² + δ` certificate — those remain this note's distinctive
  elements — but "fast rates at degenerate solutions without a
  well-conditioned Hessian" is no longer virgin territory, and a proper
  comparison is owed before any claim.
- **Identification.** Finite active-set identification for proximal methods
  under polyhedral `g` and strict complementarity is classical (Hare–Lewis
  partial smoothness); it is the expected mechanism for the local phase
  below.

---

## 6. The local rate: the quotient fails, the face succeeds

The first natural strategy for the local rate is to quotient out the
degeneracy: `f₊` is constant on cosets of `N(A_P)` (Lemma A), so pass to
`Rⁿ/N(A_P)`, where the Hessian is nondegenerate, and run the classical
self-concordant quadratic-phase argument there.

### 6.1 Why the quotient fails

The subproblem does not descend to the quotient. `g` is not constant on
cosets: by (9), `⟨λ', v⟩ > 0` for *every* nonzero `v ≥ 0` in `N(A_P)`, and
the cone `{c ≥ 0}` is not coset-invariant either. The obstruction is not
technical bookkeeping — it is the augmentation itself. §3 deliberately
*placed information on the null directions* (the empty-pixel evidence `a₀`),
so any construction that factors those directions out is blind to the very
term that made the problem well posed. The correct reading: in this problem
class, degeneracy is resolved **pointwise by the polyhedral part**, never
**uniformly by quotienting**. One must split by the optimal *face*, not by
the null *space*.

### 6.2 Splitting by the face

Standing assumptions for this section, on top of §3:

- **(A1) uniqueness:** `F` has a unique minimizer `c*` (for the finder this
  is the `β ≠ 0` regime; Theorem 1 of the notes denies it at the Radon
  point). Write `S = supp(c*)`, `T = span{e_i : i ∈ S}`.
- **(A2) strict complementarity:** `(∇f₊(c*) + λ')_i ≥ γ > 0` for all
  `i ∉ S`.

**Lemma D (uniqueness transfers nondegeneracy to the face).** Under (A1),
`N(A_P) ∩ T = {0}`; consequently `H(c)|_T = A_Sᵀ ∇²h A_S ⪰ m I_T` with
`m > 0` uniformly on a neighbourhood of `c*`.

*Proof.* Let `v ∈ N(A_P) ∩ T`. Since `c*_S > 0` and `v` is supported on `S`,
the line `c* ± tv` is feasible for small `t > 0`, and along it `f₊` is
constant by Lemma A, so `F(c* ± tv) = F(c*) ± t⟨λ', v⟩`. Optimality of `c*`
against *both* directions forces `⟨λ', v⟩ = 0` — but then the whole segment
consists of minimizers, contradicting (A1) unless `v = 0`. Uniformity follows
from continuity of `∇²h` and injectivity of `A_S` on the finite-dimensional
`T`. ∎

This one-paragraph lemma is the bridge of the whole theory: **estimator
identifiability implies solver nondegeneracy on the optimal face.** It is
where `β ≠ 0` earns the local rate — at `γ = 1` uniqueness fails, Lemma D
fails with it, and the stall measured there in the solver-arm study is this
lemma's failure observed numerically. (In spirit the statement is familiar
from lasso uniqueness theory, cf. Tibshirani; its role here — feeding a
self-concordant Newton analysis — appears to be new.)

**Lemma E (finite identification).** Under (A1)–(A2), with exact subproblems:
`c_k → c*`, and there is `K` such that for all `k ≥ K` every subproblem
solution `d_k` satisfies `supp(c_k + d_k) = S`, with the `S`-coordinates
strictly positive.

*Proof.* `c_k → c*` follows from the global corollary, compact level sets,
and (A1). Off the support: for `i ∉ S`, subproblem stationarity says
`(∇f₊(c_k) + H_k d_k + λ')_i ≥ 0` *with equality whenever
`(c_k + d_k)_i > 0`*. By Cauchy–Schwarz,
`|(H_k d_k)_i| = |⟨W_k^{1/2} A_P e_i, W_k^{1/2} A_P d_k⟩| ≤ C ν_k`, with `C`
uniform on the level set (weights bounded above and below there, again by the
background floor), and `ν_k → 0` by Theorem C. Since
`(∇f₊(c_k) + λ')_i → (∇f₊(c*) + λ')_i ≥ γ`, eventually the stationarity
component is `≥ γ/2 > 0`, which forbids equality: `(c_k + d_k)_i = 0` for
every `i ∉ S`. This holds for *every* solution: all subproblem solutions
share the same `A_P d` (the objective is strictly convex in `A_P d`), hence
the same `H_k d_k`, hence the same strict inequality. On the support: from
`(d_k)_i = −(c_k)_i → 0` off `S` and `‖A_P d_k‖ ≤ ν_k / w_min^{1/2} → 0`,
injectivity of `A_S` on `T` (Lemma D) gives `(d_k)_S → 0`, so
`(c_k + d_k)_i → c*_i > 0` for `i ∈ S`. ∎

**Theorem D (quadratic local convergence, no regularization).** Under
(A1)–(A2) and the assumptions of §3, the exact-subproblem iteration of
Theorem C identifies `S` in finitely many steps, after which it coincides
with unconstrained Newton's method on the standard self-concordant,
**nondegenerate** function `φ = (f₊ + ⟨λ', ·⟩)` restricted to the affine
hull of the face. The classical quadratic phase applies — decrement
contraction `ν̃_{k+1} ≤ (ν̃_k/(1 − ν̃_k))²` once `ν̃_k` is small — and, by
Lemma D's uniform `m > 0`, the iterates converge quadratically to `c*` in
the ordinary norm.

*Proof.* By Lemma E, past `K` every subproblem solution lies on the face
with `S`-coordinates interior, where the inequality constraints are inactive;
on the face the subproblem reduces to the unconstrained Newton step of `φ`
(unique, since `H|_T ≻ 0` by Lemma D). `φ|_T` is standard self-concordant
(restriction of a standard SC function to an affine subspace, plus a linear
term) with Hessian `⪰ m I_T` near `c*`, so the textbook Newton phase-two
theory applies verbatim; positive definiteness converts decrement contraction
into norm contraction. Remaining bookkeeping — the damped-to-full switch and
the invariance of the identification basin under the contraction — is
mechanical and flagged in §6.3. ∎

Together with Theorem C this closes an arc worth stating plainly. Theorem 3
of the theory notes proved that the *unconstrained* active-set Newton system
is consistent **iff the estimator is ill-posed** — well-posedness and Newton
were mutually exclusive (Corollary 3). With the constraint and penalty inside
the subproblem, the dichotomy inverts: *constrained* proximal Newton is
globally convergent always (Theorem C) and quadratically convergent **iff
the estimator is well posed** (unique and strictly complementary, via Lemma
D). The estimator's health and the solver's speed are the same fact, seen
twice.

### 6.3 Remaining obligations

1. **Mechanical, to be written out:** the damped-to-full-step switch rule
   and the argument that one quadratic step keeps the iterate inside the
   identification basin of Lemma E (standard basin bookkeeping); constants
   in Lemma E's `C` and `γ/2` made explicit.
2. **Inexact subproblems — global half now closed.** The global theory
   survives inexactness in a form *stronger* than anticipated, because it
   can be restated entirely in computable quantities:

   **Proposition F (computable inexact decrease).** Let `d̂` be any feasible
   step (`c + d̂ ≥ 0`) with `ν̂ = ‖d̂‖_H` and computable model decrease
   `Δ̂ = −[⟨∇f₊(c) + λ', d̂⟩ + ½ν̂²] ≥ 0`. Then the damped step
   `α = 1/(1+ν̂)` satisfies

       F(c + αd̂) − F(c) ≤ −Δ̂/(1 + ν̂).

   *Proof.* As in Theorem C, the SC bound and convexity of `g` give
   `F(c+αd̂) − F(c) ≤ α[−Δ̂ − ½ν̂²] + ω_*(αν̂)`. At `α = 1/(1+ν̂)` the
   `Δ̂`-free terms sum to `−r(ν̂)` with
   `r(ν) = (ν + ½ν²)/(1+ν) − log(1+ν)`, and `r(0) = 0`,
   `r'(ν) = ½ν²/(1+ν)² ≥ 0`, so `r ≥ 0`. ∎

   No reference to the exact subproblem solution appears. For global
   convergence of an implemented method it then suffices that each inner
   solve is at least as good as its own first projected-gradient step under
   a majorizing diagonal `D ⪰ H` (Gershgorin, §3 of the primer's
   machinery): majorized descent gives `Δ̂ ≥ ½‖x₁ − c‖²_D`, and `x₁ − c`
   is the subproblem's `D`-scaled prox-gradient map at `d = 0`, which
   vanishes only at stationarity of `F`. So `Δ̂_k → 0` (forced by
   summability) again implies every accumulation point is a minimizer.
   Still open on this item: *superlinear retention* under vanishing
   inexactness (`Δ̂/Δ → 1` near the end), to be carried through the
   seminorm bookkeeping. A practical corollary the prototype confirmed: for
   exact solutions `δ = Δ − ½ν² ≥ 0`, so a computed `δ̂ < 0` is a free,
   rigorous inexactness alarm.
3. **Without strict complementarity** the identification argument fails and
   the expected behaviour is degradation to a linear rate — not attempted
   here.
4. **Literature diligence.** Before any novelty claim: Sun & Tran-Dinh's
   generalized self-concordance (their treatment of Poisson-type losses may
   subsume §1.2's integer-counts observation), Bach's self-concordant GLM
   analysis, lasso-uniqueness theory (Tibshirani), and the recent
   degenerate/inexact proximal Newton literature. The assessment — Theorem C
   plus Theorem D as an assembly is a publishable note, Lemma D's role as
   the identifiability-to-nondegeneracy bridge the most likely genuinely new
   element — is a prior, not a verdict.

---

## 7. Implementation feedback

A fully instrumented prototype of the method (exact splitting, GPCG-lite
inner solver — Gershgorin projected gradient for identification, Jacobi-CG
on the free set, projected decrease check — damped/full steps by `ν`, the FB
step as monotone safeguard) was run on the 4-peak synthetic case, reporting
every quantity of §§4–6 per iteration. Findings, in both directions:

**Theory confirmed by the implementation.**

- The build-time identity `∇f₊ + a₀ = Aᵀ(1 − y/u)` holds to `1.1e-7`
  (float32): the augmented and original formulations are the same
  arithmetic, and the FB safeguard is bit-compatible between them.
- Theorem C's decrease bound held on **all 80** instrumented cold-start
  iterations, with the *inexact* inner solver — consistent with
  Proposition F.
- The phase structure of §6 is visible in the raw trajectories: at
  `γ = 0.5` the support froze at iteration 32 and the accepted step
  switched FB → Newton from that point on; at `γ = 1` the support churned
  through all 40 iterations without settling — Lemma D's failure, as the
  identification signature.
- **Endgame (Theorem D):** warm-started from the production solver's
  certified endpoint (KKT residual `4.4e-4`), a *single full Newton step*
  (`ν = 0.075`) dropped the residual to `7.3e-5` and deactivated one atom
  (Lemma E in action); `Δ̂` fell from `1.2e4` (cold) to `2.5e-6`. All
  subsequent iterations are float32 noise (`|dF| ≲ 2e-6` on `|F| ~ 1e5`,
  below the ulp of the fused difference): the quadratic phase is **one
  step deep at float32**, and it ends at the arithmetic's floor, not the
  method's.
- The inexactness alarm works: the one Newton step showed
  `δ̂ = −6.4e-6 < 0` (relative inexactness `~2e-3` against
  `Δ̂ = 2.8e-3`), correctly flagging the inner solve as inexact while
  Proposition F kept the step certified.

**Implementation lessons fed back into the theory and the design.**

- Cold-started damped Newton spends its entire budget in phase 1 (FB steps
  accepted for 32 straight iterations; `ν ~ 20–150`): the certified
  architecture is *first-order phase 1 → APN endgame*, i.e. the shipped
  sufficient-decrease hybrid runs to its `1e-3` certificate and one or two
  APN outer iterations (~40 convolutions) then buy a `6×` tighter
  certificate and a cleaner support. The endgame is an add-on, not a
  replacement.
- Proposition F exists because the implementation demanded a guarantee in
  computable quantities; the restatement turned out stronger than the
  planned `Δ̂ ≥ ρΔ` formulation.
- The correct stopping quantity in practice is `Δ̂` (it spans ten orders
  over the run and is monotone-interpretable); `ν` alone freezes at the
  float32 noise floor (`1.5e-3`) and never certifies anything there.

**Loose ends surfaced.**

1. Observing the quadratic *contraction curve* (rather than one step to
   the floor) requires float64; the prototype and the production stack are
   float32. Worth one float64 run before any write-up claims the rate is
   observed rather than implied.
2. A stopping rule on `Δ̂` needs its units tied to a user-facing tolerance:
   `Δ̂` is in objective (count) units; the bridge `F(c) − F* ≤` (function
   of `ν`, `Δ̂`) in the degenerate case is part of obligation 1's
   bookkeeping.
3. Production integration is a design decision, not a math one: the
   endgame changes the reported support (76 → 75 here) and so interacts
   with the downstream test suite exactly as any solver improvement does.

---

## 8. Subpixel refinement: the limit of the augmentation

Question: does the zero-pixel augmentation alone suffice to make
*subpixel refinement* well posed — continuous re-estimation of peak
positions after the convex stage — or is there an informative
counterexample? Answer: **it does not suffice, for two independent
reasons, one per counting regime; and the counterexample is not only
constructible but explains the refinement instability observed in CI.**
The positive result that survives is an *expected*-curvature statement,
which converts into a cheap per-peak certificate.

### 8.1 Regime vacuity (measured)

The augmentation term `a₀ = Aᵀ1_{y=0}` is only as informative as the
empty pixels are numerous. Census of the ghost-test frame (background
10 counts/pixel, the case whose refinement instability was measured in
the CI flake): **1 pixel of 10,000 has `y = 0`** (expected `~0.5` from
background alone). In this regime `a₀ ≈ 0`, `λ' ≈ λ`: the null-space
augmentation is numerically inert exactly where the subpixel
instability was observed. What removes sub-significance satellites
there is the *statistical* part of the penalty retained inside a
penalized refinement (soft-thresholding can deactivate; the current
unpenalized refiner cannot) — not the zero-pixel term.

### 8.2 The curvature counterexample

The deeper obstruction is second-order and cannot be cured by *any*
reweighting of the penalty, because the augmented objective is an exact
rewrite of the original: stationary structure in the position
coordinates is untouched. For a single atom `U = B + c φ_σ(· − ξ)`,
using `Σ_k ∂²Φ_k/∂ξ² = 0` (mass invariance over the detector), the
position curvature of the exact NLL at a stationary point is

    ∂²(NLL)/∂ξ² = Σ_k (y_k/U_k) [ (∂U_k/∂ξ)²/U_k − ∂²U_k/∂ξ² ],     (15)

a sum over *counted pixels only*, with a per-pixel sign: in the
continuous approximation a count at distance `x` from the center
contributes positively iff the local signal-to-background satisfies

    c φ(x)/B  >  x²/σ² − 1.                                          (16)

Counts inside the core stabilize the position; counts in the tails
(beyond the inflection radius, where the local signal is thin against
the background) *destabilize* it. Two consequences:

- **`B = 0` is safe:** with no background, (15) reduces to
  `Σ y_k/σ²` — log-concavity of the Gaussian in `ξ` — and refinement
  cannot saddle. The pathology requires background, which we always
  have.
- **A significant saddle exists.** Place on a flat field at the
  background mean a ring of extra counts at radius `d` (a symmetric,
  perfectly legal Poisson realization — and precisely the residual
  geometry left by an atom whose width is off the σ-bank, i.e. the
  ghost mechanism). Measured, `σ = 2`, exact autodiff Hessian at the
  fitted amplitude, float64:

  | B | ring d | fitted amp | matched-filter z | min-eig(position) | refined from ±0.15 px |
  |---|---|---|---|---|---|
  | 10 | 3 | 2.20 | 2.5 | −0.063 (saddle) | ±0.24 px |
  | 10 | 4 | 1.88 | 2.1 | −0.86 (saddle) | ±3.2 px |
  | 2 | 3 | 2.61 | 6.5 | +1.17 (min) | — |
  | **2** | **4** | **2.60** | **6.5** | **−3.83 (saddle)** | **±3.0 px** |
  | 2 | 5 | 0.52 | 1.3 | −0.74 (saddle) | ±4.6 px |

  The boxed row is the informative counterexample: an atom **well above
  the significance floor** (`z = 6.5` against a floor of ≈ 4.6) whose
  position is a *saddle point* of the exact likelihood. Refinement
  bifurcates to `±3` pixels, the branch chosen by the sign of the
  starting perturbation — in production, by reduction-order noise.
  This is the measured phenomenology of the CI-flaky ghost (slid one
  way locally, the other way on CI), now with its mechanism: the ghost
  was a residual-mopping atom whose counts sat in exactly the
  destabilizing zone of (16).

### 8.3 What survives, and the certificate

In expectation the pathology vanishes: substituting `y_k → U_k` in
(15), the `∂²U` terms cancel by mass invariance and

    E[∂²(NLL)/∂ξ²] = Σ_k (∂U_k/∂ξ)² / U_k  ≻  0,                     (17)

the Fisher position information — *subpixel refinement is well posed in
expectation for every true peak, at every background*. Failures are
finite-sample events. Monte-Carlo incidence for a genuine centered peak
on `B = 10` (200 replicates each): indefinite position Hessian at the
fitted optimum in `1/200` runs at threshold significance (`z = 4.5`),
`0/200` at `z ≥ 6.7`. Rare per genuine peak — but ring-shaped residuals
are *systematically* produced by width mismatch, so the incidence among
residual-mopping atoms (the ones the ghost test exercises) is far
higher than among true peaks.

The constructive conclusion: subpixel positions need their own
certificate, and it is second-order. Alongside the amplitude test
(first-order, against `λ'`), report per refined peak the empirical
position-information matrix — the `2×2` (or `3×3` with `σ`) block of
(15), one autodiff Hessian per atom — and flag any peak whose smallest
eigenvalue is nonpositive (or below a noise floor): *its subpixel
position is not identifiable at this significance*; report the
pixel-level position with widened uncertainty instead of a silently
bifurcating subpixel value. This slots into the framework of this note
as the position-space sibling of the `Δ` certificate, and into Part II
of the companion paper: the same matrix's inverse is the position
covariance for `σ(ξ)` reporting.

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
- Tibshirani, R. J., *The lasso problem and uniqueness*, Electron. J. Stat.
  7, 2013.
- Rockafellar & Wets, *Variational Analysis*, Springer, 1998.
