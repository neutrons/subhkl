"""Matrix-free Poisson Newton steps with panel backgrounds eliminated."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def schur_direction(
    matrix,
    curvature,
    x,
    gradient,
    group_penalties,
    n_background,
    damping=1e-10,
    rtol=1e-9,
):
    """Solve the free-variable Newton system without a dense Hessian.

    Each row belongs to one background column. Its Hessian block is diagonal,
    so backgrounds can be eliminated exactly from the damped Newton system.
    Group-norm curvature is a diagonal minus one rank-one term per active group.
    The returned step still needs a projected line search on the true objective.
    """
    n_reflection = len(x) - n_background
    n_groups = len(group_penalties)
    size = n_reflection // n_groups
    groups = x[:n_reflection].reshape(n_groups, size)
    norms = np.linalg.norm(groups, axis=1)
    free = (x > 1e-9) | (gradient < 0)
    free[:n_reflection] &= np.repeat(norms > 0, size)
    reflections = np.flatnonzero(free[:n_reflection])
    backgrounds = n_reflection + np.flatnonzero(free[n_reflection:])
    F = matrix[:, reflections]
    B = matrix[:, backgrounds]
    if np.any(B.getnnz(axis=1) > 1):
        raise ValueError("background columns must have disjoint pixel support")
    df = np.asarray(F.power(2).T @ curvature).ravel()
    db = np.asarray(B.power(2).T @ curvature).ravel()
    blocks = []
    for group, norm in enumerate(norms):
        pos = np.flatnonzero(reflections // size == group)
        if len(pos):
            v = x[reflections[pos]] / norm
            alpha = group_penalties[group] / norm
            df[pos] += alpha * (1 - v * v)
            blocks.append((pos, v, alpha))
    ridge = damping * max(1.0, np.max(df, initial=0), np.max(db, initial=0))
    db = db + ridge
    cross = (F.T @ B.multiply(curvature[:, None])).tocsr()
    diagonal = df + ridge - np.asarray(cross.power(2) @ (1 / db)).ravel()

    def product(v):
        result = np.asarray(F.T @ (curvature * (F @ v))).ravel() + ridge * v
        for pos, u, alpha in blocks:
            result[pos] += alpha * (v[pos] - u * (u @ v[pos]))
        return result - cross @ ((cross.T @ v) / db)

    step = np.zeros_like(x)
    iterations = 0
    info = 0
    if len(reflections):
        operator = LinearOperator((len(reflections), len(reflections)), matvec=product)
        preconditioner = LinearOperator(
            operator.shape, matvec=lambda v: v / np.maximum(diagonal, ridge)
        )
        rhs = -gradient[reflections] + cross @ (gradient[backgrounds] / db)

        def count(_):
            nonlocal iterations
            iterations += 1

        dr, info = cg(
            operator,
            rhs,
            M=preconditioner,
            rtol=rtol,
            atol=1e-12,
            maxiter=max(100, min(5000, 5 * len(reflections))),
            callback=count,
        )
        step[reflections] = dr
    step[backgrounds] = (-gradient[backgrounds] - cross.T @ step[reflections]) / db
    return step, iterations, info
