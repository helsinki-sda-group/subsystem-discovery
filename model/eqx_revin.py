# from GPT
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any

class RevIN(eqx.Module):
    num_sensors: int
    eps: float = 1e-5
    affine: bool = True
    affine_weight: Any
    affine_bias: Any

    def __init__(self, num_sensors, eps=1e-5, affine=True):
        self.num_sensors = num_sensors
        self.eps = eps
        self.affine = affine

        self.affine_weight = jnp.ones((1, self.num_sensors, 1, 1))
        self.affine_bias = jnp.zeros((1, self.num_sensors, 1, 1))

    def __call__(self, x, mode: str, **kwargs):
        if mode == 'norm':
            mean, stdev = self._get_statistics(x)
            x = self._normalize(x, mean, stdev)
            return x, mean, stdev
        elif mode == 'denorm':
            mean, stdev = kwargs.get('mean'), kwargs.get('stdev')
            if mean is None or stdev is None:
                raise ValueError("Mean and stdev must be provided for denormalization.")
            x = self._denormalize(x, mean, stdev)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = (-1, -2) # assuming seq is last dim
        mean = jnp.mean(x, axis=dim2reduce, keepdims=True)
        stdev = jnp.std(x, axis=dim2reduce, keepdims=True) + self.eps
        return mean, stdev

    def _normalize(self, x, mean, stdev):
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x, mean, stdev):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * stdev + mean
        return x
