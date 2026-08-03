# Proximal gradient methods: a primer for the matrix-free peak finder

This document explains, from first principles, the mathematics implemented in
`MatrixFreeSparseRBFPeakFinder._solve_ssn_cg_global`
(`src/subhkl/search/matrix_free.py`). It is written at the level of a
graduate-course introduction to proximal methods: everything elementary is
derived in full, and everything research-grade is deferred to the companion
document [`matrix_free_theory.md`](matrix_free_theory.md), which assumes the
background given here.

The plan mirrors the standard exposition of proximal gradient methods for
learning:

1. the optimisation problem (a smooth data fit plus a non-smooth penalty);
2. why plain gradient descent does not apply;
3. the proximal operator, and its closed form for our penalty
   (one-sided soft-thresholding);
4. the forward–backward iteration and its descent guarantee;
5. fixed points and their equivalence to the optimality conditions;
6. acceleration — here a semismooth Newton method rather than momentum —
   and the fallback that keeps the guarantee.

Throughout, every mathematical symbol is tied to the identifier that carries it
in the code. The dictionary is collected in §7.

---

## 1. The optimisation problem

The detector image `y ∈ R^{H×W}` is modelled as Poisson counts around a mean
composed of a smooth background `b` (estimated separately, `bg_img`) and a
sparse sum of Gaussian peaks. The peaks are represented on a *scale-space
dictionary*: one candidate Gaussian atom per pixel position and per width
`σ_k`, `k = 1…K`, with `σ_k` on a fixed grid (`jnp.linspace(min_sigma,
max_sigma, num_sigmas)`). The unknown is a nonnegative coefficient tensor

    c ∈ R^{K×H×W},   c ≥ 0,

where `c_{k,i,j}` is the *peak amplitude* of an atom of width `σ_k` centred at
pixel `(i,j)`. The forward model is linear in `c`:

    u(c) = A c + b,                                                        (1)

where `A` applies each channel's Gaussian kernel and sums the channels — a
convolution, evaluated matrix-free as `self._forward_op(c, self.K_weights)`.
Its adjoint `Aᵀ` (correlation with the same kernels) is
`self._adjoint_op(·, self.K_weights)`. No matrix is ever formed; `A` exists
only as these two functions, which is what "matrix-free" means.

The estimate is the minimiser of a penalised negative log-likelihood:

    min_{c ≥ 0}   J(c) = D(c) + ⟨λ, c⟩,                                    (2)

with the data-fidelity term (dropping `c`-independent constants)

    D(c) = Σ_z [ u_z − y_z log u_z ]          (Poisson, `loss="poisson"`)   (3)
    D(c) = ½ Σ_z (u_z − y_z)²                 (Gaussian, `loss="gaussian"`)

and a weighted ℓ1 penalty `⟨λ, c⟩ = Σ λ_{k,i,j} c_{k,i,j}` with per-channel
weights `λ ≥ 0` (code: `lam`). Because `c ≥ 0`, the penalty `Σ λ|c|` reduces
to the linear form `⟨λ, c⟩` — this is nonnegative basis pursuit / the
nonnegative lasso, in the same family as the lasso problems for which
proximal gradient methods were developed.

Two derivatives of `D` recur. The gradient, by the chain rule through (1):

    ∇D(c) = Aᵀ (1 − y/u)        (Poisson; code: `grad`)                     (4)
    ∇D(c) = Aᵀ (u − y)          (Gaussian)

and the curvature. The exact Poisson Hessian is `Aᵀ diag(y/u²) A`; the code
uses the *Fisher* weight instead,

    H(c) = Aᵀ W(u) A,     W(u) = diag(1/u)      (code: `W_diag`),           (5)

which replaces `y` by its expectation `u`. This choice is deliberate: `W ≻ 0`
always, so `H` is positive semidefinite with the same null space as `A`
restricted to the active coordinates (Lemma 2 of the companion notes), whereas
the exact Hessian can be indefinite wherever `y` fluctuates low.

**Units.** `c` carries counts-per-pixel amplitude; `u`, `y`, `b` carry counts;
`∇D` and `λ` carry counts per unit amplitude, i.e. objective per unit `c`. The
penalty weight is constructed in *objective* units,

    λ_k = α_k · H_diag,k · sqrt(var_c,k),                                   (6)

(code: `lam = alpha_vec * H_diag_safe * jnp.sqrt(var_c)`), so that the
soft-threshold `λ/H_diag` of §3 lands at `α` standard deviations of the
coefficient's own noise, `sqrt(var_c) = H_diag^{-1/2}`. The statistical choice
of `α` (`effective_alpha`: a user value floored by the universal threshold
`sqrt(2 log N_k)` of Donoho–Johnstone, with `N_k = HW/(2πσ_k²)` the number of
independent resolution elements at scale `σ_k`) is discussed in §4 of the
companion notes; for this document `λ` is simply a fixed nonnegative weight.

---

## 2. Why not plain gradient descent

`J` is a sum of two parts with opposite characters:

- `D` is smooth (infinitely differentiable on `u > 0`) and convex, and its
  gradient is Lipschitz on the feasible set (§4 below);
- `⟨λ, c⟩ + ι_{c ≥ 0}` — the penalty together with the constraint, written
  with the indicator function `ι_{c≥0}(c) = 0` if `c ≥ 0`, `+∞` otherwise —
  is convex but *not differentiable*: the constraint boundary `c_i = 0` is a
  kink, and it is precisely where sparse solutions live. A gradient step has
  no way to land a coordinate *exactly* on zero, and projecting after the
  step ignores the penalty.

The classical remedy is *operator splitting*: treat the smooth part with a
gradient step (the "forward" step) and the non-smooth part with its proximal
operator (the "backward" step), which handles the kink exactly. For that we
need the proximal operator.

---

## 3. The proximal operator and one-sided soft-thresholding

For a convex function `R` and step size `τ > 0`, the proximal operator is

    prox_{τR}(v) = argmin_x  ½ ‖x − v‖² + τ R(x).                           (7)

It is the resolvent of the subdifferential of `R`: a point that balances
staying near `v` against decreasing `R`. Two standard examples calibrate the
definition: if `R = 0`, `prox` is the identity; if `R = ι_C` is the indicator
of a convex set, `prox` is the projection onto `C`.

Our penalty is `R(c) = ⟨λ, c⟩ + ι_{c ≥ 0}(c)`. It is *separable* — a sum of
independent scalar terms — so (7) splits into one scalar problem per
coefficient:

    prox(v)_i = argmin_{x ≥ 0}  ½ (x − v_i)² + τ λ_i x.                     (8)

Solve (8) directly. The unconstrained objective is a parabola with minimum at
`x = v_i − τλ_i`. If `v_i − τλ_i ≥ 0`, that point is feasible and is the
answer. If `v_i − τλ_i < 0`, the parabola is increasing on `x ≥ 0`, so the
constrained minimum is at the boundary `x = 0`. Hence

    prox_{τR}(v)_i = max(0, v_i − τ λ_i),                                   (9)

**one-sided soft-thresholding**: shift down by `τλ` and clip at zero. (For the
ordinary lasso, without the sign constraint, the same computation on `x < 0`
produces the familiar two-sided form `sign(v)·max(0, |v| − τλ)`; the
nonnegativity constraint simply deletes the negative branch.) In the code,
`τλ` is precomputed once as `tau_alpha = tau_local * lam` and (9) appears
verbatim as

    c = jnp.maximum(0.0, q - tau_alpha).

The shift is why ℓ1 penalties produce *exact* zeros: every coefficient whose
"vote" `v_i` fails to clear the threshold `τλ_i` is set to zero identically,
not merely made small. It is also why ℓ1 estimates of surviving coefficients
are biased low by `τλ` — the shrinkage that the refinement stage of the
pipeline later undoes (see §7c of the companion notes for what can go wrong
there).

---

## 4. The forward–backward iteration

The **forward–backward** (proximal gradient) iteration alternates one gradient
step on `D` with one prox on `R`:

    c⁺ = prox_{τR}( c − τ ∇D(c) )
       = max( 0,  c − τ ∇D(c) − τλ ).                                      (10)

### 4.1 The descent guarantee

Say `∇D` is `L`-Lipschitz (equivalently, for twice-differentiable `D`, the
Hessian satisfies `∇²D ⪯ L·I`). Then the *descent lemma* holds: for all
`x, c`,

    D(x) ≤ D(c) + ⟨∇D(c), x − c⟩ + (L/2) ‖x − c‖².                         (11)

The right-hand side, plus the penalty, is exactly the objective that the prox
step (10) minimises when `τ = 1/L` — completing the square in (7) with
`v = c − τ∇D(c)` shows

    c⁺ = argmin_x [ D(c) + ⟨∇D(c), x − c⟩ + (1/2τ) ‖x − c‖² + ⟨λ, x⟩ ].    (12)

So each iteration minimises an upper bound (a *majoriser*) of `J` that touches
`J` at the current point. Evaluating (11) at `x = c⁺` and using the fact that
`c⁺` beats `x = c` in (12) gives, for `τ ≤ 1/L`,

    J(c⁺) ≤ J(c) − (1/(2τ) − L/2) ‖c⁺ − c‖²  ≤  J(c).                      (13)

Every step with `τ ≤ 1/L` decreases the objective — **no line search is
needed**. For convex `D` this iteration converges to a minimiser with
`J(c_t) − J* = O(1/t)`.

### 4.2 The step size in the code, and a Poisson subtlety

The bound `L` must dominate `λ_max(Aᵀ W A)`. Two things could go wrong.

First, the temptation to bound `λ_max` by the largest *diagonal* entry of
`AᵀWA` fails badly here: the atoms overlap heavily, and for a typical bank
`λ_max` exceeds `max(diag)` by a factor of several hundred. The code instead
runs 15 power iterations on the operator itself (`power_step`), obtaining
`L_max` as a Rayleigh quotient, and sets

    tau_local = 1.0 / (L_max + 1e-4).                                      (14)

Second, for the Poisson loss `W(u) = diag(1/u)` *depends on the iterate*, so a
Lipschitz constant computed once could in principle be invalidated later. It
is not, for a structural reason: the model (1) has `u = Ac + b ≥ b`
elementwise whenever `c ≥ 0` (the atoms are nonnegative), hence

    W(u) = diag(1/u) ⪯ diag(1/b) = W(0)     elementwise,                   (15)

so the curvature at *any* feasible point is dominated by the curvature at
`c = 0`. The power iteration is therefore run once, on `W_ref = 1/max(b,
10⁻³)` (`W(0)`), and yields a *global* Lipschitz bound valid for the entire
solve. This is Theorem 2 of the companion notes: a strictly positive
background floor turns the Poisson problem, whose log-likelihood has unbounded
curvature at `u → 0`, into a globally `L`-smooth one.

---

## 5. Fixed points are exactly the minimisers

When should the iteration stop? Introduce the auxiliary variable the code
iterates in,

    q  (the pre-threshold iterate),      c(q) = max(0, q − τλ),            (16)

and the **natural residual** of the fixed-point map,

    G(q) = ( q − c(q) ) / τ + ∇D(c(q))          (code: `Gq`).              (17)

Two facts make `G` the right object.

**(a) `G(q) = 0` if and only if `c(q)` is a minimiser of (2).**
The Karush–Kuhn–Tucker conditions for (2) — necessary and sufficient, since
the problem is convex — are, per coordinate:

    c_i > 0  ⟹  ∇D(c)_i + λ_i = 0        (stationarity on the support)
    c_i = 0  ⟹  ∇D(c)_i + λ_i ≥ 0        (no descent into the feasible set)

Check both cases against `G = 0`. Where `q_i − τλ_i > 0`: `c_i = q_i − τλ_i`,
so `(q_i − c_i)/τ = λ_i` and `G_i = 0` reads `λ_i + ∇D(c)_i = 0` — the first
KKT condition. Where `q_i − τλ_i ≤ 0`: `c_i = 0`, so `G_i = 0` reads
`q_i/τ = −∇D(c)_i`; combined with `q_i ≤ τλ_i` this gives
`∇D(c)_i + λ_i ≥ 0` — the second. Conversely, any KKT point yields a `q`
(set `q_i = c_i + τλ_i` on the support, `q_i = −τ∇D_i` off it) with
`G(q) = 0`. So driving `‖G‖ → 0` is driving the iterate to optimality, and
`τ‖G‖` is a computable optimality measure.

**(b) `−τG` *is* the forward–backward step.** Directly from (16)–(17):

    q − τ G(q) = c(q) − τ ∇D(c(q)),                                        (18)

and thresholding the right-hand side (the next `c = max(0, · − τλ)`) is
precisely (10) applied at `c(q)`. So the update `q ← q − τG(q)` is the
guaranteed-descent iteration of §4, merely bookkept in the `q` variable. This
identity — eq. (22) of the companion notes — is what makes the fallback of
§6.3 free: the solver already computes `G` at every iteration.

The stopping test in the code, `step_norm > 1e-3` on `‖q⁺ − q‖`, measures
`τ‖G‖` whenever the fallback step was taken, i.e. it is (b) turned into a
convergence criterion.

---

## 6. Acceleration: semismooth Newton with a forward–backward fallback

Forward–backward alone converges, but linearly at best, with rate roughly
`1 − 1/κ` where `κ` is the condition number of the reduced Hessian — and heavy
atom overlap makes `κ` large here (measured: still `‖G‖ ≈ 140` after 1000 FB
iterations on a benchmark frame). The classical accelerations are momentum
methods (FISTA), which improve `O(1/t)` to `O(1/t²)` but remain first-order.
This solver instead goes second-order: **Newton's method on the equation
`G(q) = 0`**.

### 6.1 The semismooth Newton system

`G` is not differentiable — `c(q) = max(0, q − τλ)` has a kink — but it is
*semismooth*: piecewise smooth with well-defined one-sided derivatives, which
is enough for a Newton method with local superlinear convergence (Qi & Sun).
On the two smooth pieces, indexed by the **active set**

    S = { i : q_i > τ λ_i }        (code: `D_mat`, a 0/1 mask),             (19)

the generalised Jacobian of `G` is block-diagonal:

    ∂G =  Aᵀ W A   on S×S       (the Fisher curvature (5)),
          (1/τ) I  on the inactive block.                                   (20)

The Newton direction solves `∂G · dq = −G`. The inactive block is trivial —
it yields `dq_i = −τ G_i`, the forward–backward step, exactly right for
coordinates at the constraint boundary. The active block is a linear system in
`H_S = A_Sᵀ W A_S`, solved matrix-free by **conjugate gradients** (20
iterations, tolerance 10⁻³), with a Jacobi (diagonal) preconditioner supplied
through CG's `M=` argument — kept *outside* the operator because folding an
asymmetric scaling into it would break the symmetry CG requires (§5 of the
companion notes documents this failure mode). A relative ridge
`RIDGE_REL · diag(H)` regularises the active block; why the ridge must be
*relative* to `diag(H)` (units: `H` carries counts/amplitude², so a bare
additive constant means a different strength on every image) is discussed in
the code comment and §8.4 of the notes.

### 6.2 Globalisation: the line search

Newton steps are only locally reliable, so each one is tested against the true
objective: a backtracking line search accepts `q + s·dq` for the largest
`s ∈ {1, ½, ¼, …, 2⁻¹²}` that does not increase `J`. The decrease is computed
as a single sum of *per-pixel differences* (`obj_delta`), never as
`J(c_test) − J(c)` — two `O(|J|)` totals whose difference drowns below float32
resolution long before convergence (the comment above `obj_delta` quantifies
this; it stalled every solve on real-size frames).

### 6.3 The fallback: keeping the guarantee

When all 12 halvings fail — which provably happens on this dictionary, since
the active-set Newton system can be singular or even inconsistent (Theorem 3
of the notes: on a scale-space dictionary, exactly when the estimator is well
posed) — the solver does **not** stop and does not accept the bad step. It
takes the forward–backward step instead, via identity (18):

    accept = isfinite(d_try) & (d_try <= 0)
    q_fb   = q - tau_local * Gq            # = c - τ∇D(c), eq. (18)
    q⁺     = where(accept, q_try, q_fb)

By §4.1 this step is *guaranteed* to decrease `J` — `τ ≤ 1/L` holds globally
by (15) — so a rejected Newton step costs one slow iteration, never
correctness. The resulting hybrid is the textbook pattern for nonsmooth
Newton methods: a globally convergent first-order method supplies the
guarantee; Newton supplies the speed when its model is faithful. Measured on
the benchmark case (§8.7 of the notes): pure FB accepts every step but is far
from converged at 1000 iterations; the hybrid reaches `‖G‖` three orders of
magnitude smaller in 150.

---

## 7. Symbol–code dictionary

| math | meaning | code (`_solve_ssn_cg_global`) |
|---|---|---|
| `c` | nonnegative coefficients (peak amplitudes), `K×H×W` | `c` |
| `q` | pre-threshold iterate, `c = max(0, q − τλ)` | `q` |
| `A`, `Aᵀ` | dictionary synthesis / analysis (convolutions) | `_forward_op`, `_adjoint_op` with `self.K_weights` |
| `u = Ac + b` | modelled counts | `u` in `get_loss_grad_hess` |
| `D(c)` | Poisson/Gaussian NLL, eq. (3) | `nll` |
| `∇D(c)` | eq. (4) | `grad` |
| `W(u)` | Fisher weight `1/u`, eq. (5) | `W_diag` |
| `W(0)` | curvature bound at `c = 0`, eq. (15) | `W_ref` |
| `λ` | ℓ1 weight in objective units, eq. (6) | `lam` |
| `α_k` | threshold in noise-σ units | `alpha_vec` from `effective_alpha` |
| `L` | Lipschitz bound `λ_max(AᵀW(0)A)` | `L_max` via `power_step` |
| `τ` | step size `≤ 1/L` | `tau_local` |
| `τλ` | soft threshold | `tau_alpha` |
| `prox_{τR}` | one-sided soft-threshold, eq. (9) | `jnp.maximum(0.0, q - tau_alpha)` |
| `G(q)` | natural residual, eq. (17) | `Gq` |
| `S` | active set, eq. (19) | `D_mat` |
| `∂G` | generalised Jacobian, eq. (20) | `apply_jacobian` |
| — | Jacobi preconditioner | `jacobi`, `eta = 1/H_diag_local` |
| `τ‖G‖` | optimality measure / stop test | `step_norm` vs `1e-3` |
| eq. (18) | FB fallback step | `q_fb = q - tau_local * Gq` |

---

## 8. Further reading

Standard references for the material of §§3–5: Combettes & Wajs, *Signal
recovery by proximal forward–backward splitting*, Multiscale Model. Simul.
4(4), 2005; Parikh & Boyd, *Proximal algorithms*, Found. Trends Optim. 1(3),
2014; Beck & Teboulle, *A fast iterative shrinkage-thresholding algorithm*
(FISTA), SIAM J. Imaging Sci. 2(1), 2009. For §6: Qi & Sun, *A nonsmooth
version of Newton's method*, Math. Program. 58, 1993; Hintermüller, Ito &
Kunisch, *The primal-dual active set strategy as a semismooth Newton method*,
SIAM J. Optim. 13(3), 2003. For everything specific to this dictionary — the
scale degeneracy, the significance thresholds, the Newton inconsistency
theorem, and the control experiments — see
[`matrix_free_theory.md`](matrix_free_theory.md). A follow-up note,
[`augmented_penalty_newton.md`](augmented_penalty_newton.md), develops the
second-order theory further: an exact splitting of the Poisson likelihood
that makes the fidelity standard self-concordant, and a damped proximal
Newton analysis that remains valid — global convergence included — when the
Hessian is singular.
