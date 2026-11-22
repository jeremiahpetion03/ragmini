# Locate the first non-finite value in a forward pass

A model's output contains NaN. Finding where it came from by checking only the final output is useless: an `inf` produced early often turns into `NaN` several layers later (`inf - inf`, `0 * inf`), so the layer where NaN first *appears* is usually not the layer that *caused* it.

`starter.py` runs every layer and then reports the first layer whose output contains NaN. Fix it.

**Implement** `locate_first_nonfinite(layers, x)`:

- `layers`: list of `(name, fn)` pairs. Each `fn` takes an ndarray and returns an ndarray. **Some layers modify their input in place.**
- `x`: the input ndarray.

Run the layers in order. Return `None` if every output is finite. Otherwise stop **immediately** after the first layer whose output contains any NaN, `+inf` or `-inf`. Do not run later layers. Return a dict:

| key | value |
|---|---|
| `index` | position of that layer in `layers` (`-1` if `x` itself is non-finite; then no layer runs) |
| `name` | its name (`"<input>"` for index -1) |
| `input` | a copy of the layer's input **as it was before the layer ran** (for index -1: a copy of `x`) |
| `output` | a copy of the layer's output (for index -1: a copy of `x`) |
| `first_position` | tuple index of the first non-finite element of `output`, in row-major (C) order |
| `n_nan`, `n_posinf`, `n_neginf` | counts in `output` |

`x` must not be modified, even if a layer works in place.
