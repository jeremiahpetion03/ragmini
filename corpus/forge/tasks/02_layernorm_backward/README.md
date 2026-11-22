# LayerNorm backward pass

A custom LayerNorm layer trains, but much worse than the framework version. Gradient checks point at the backward pass in `starter.py`.

The forward pass normalizes over the **last** axis with the biased variance:

```
mu   = mean(x, axis=-1)
var  = mean((x - mu)**2, axis=-1)
xhat = (x - mu) / sqrt(var + eps)
y    = gamma * xhat + beta
```

**Implement** `layernorm_backward(dy, x, gamma, eps=1e-5)` returning `(dx, dgamma, dbeta)`:

- `x` and `dy` have shape `(..., D)` with any number of leading dimensions. `gamma` has shape `(D,)`.
- `dx` has the shape of `x`. `dgamma` and `dbeta` have shape `(D,)`, summed over all leading dimensions.
- `dx` must be correct even when `var` is comparable to `eps`.

Hidden checks compare against central finite differences of the forward pass in float64, and test mathematical invariants of the gradient.
