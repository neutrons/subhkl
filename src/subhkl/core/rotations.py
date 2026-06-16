import jax.numpy as jnp


def rotation_from_axis_angle(axis, angle_rad):
    u = axis / jnp.linalg.norm(axis)
    ux, uy, uz = u
    K = jnp.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    eye = jnp.eye(3)
    R = eye + s[..., None, None] * K + (1.0 - c)[..., None, None] * (K @ K)
    return R


def rotation_from_rodrigues(w):
    theta = jnp.linalg.norm(w) + 1e-9
    k = w / theta
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    eye = jnp.eye(3)
    R = eye + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
    return R
