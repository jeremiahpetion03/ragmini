# Leak-free feature preprocessing

A model looks great offline and degrades in production. The cause: `starter.py` computes its scaling statistics from whatever batch it is transforming, so validation data leaks into the features, and a single request served alone gets different features than the same request inside a batch (train/serve skew).

**Implement** class `FeaturePipeline` with:

- `fit(num, cat)`: learn statistics from TRAINING data only. Returns `self`.
- `transform(num, cat)`: apply them to any data. Returns a float array of shape `(n, d_num + total_categories)`.

Inputs: `num` is an `(n, d_num)` float array that may contain NaN (missing). `cat` is an `(n, d_cat)` array of strings.

Numeric columns, in order:
1. From the training data, ignoring NaN: median, mean and population standard deviation (ddof = 0) of each column.
2. Replace NaN with the training median.
3. Standardize: `(x - mean) / std`. A column with std 0 uses std 1.

Categorical columns, in order: one-hot encode against the training categories of that column, in **sorted** order. A category never seen in training becomes all zeros.

Requirements:
- `transform` uses only statistics learned in `fit`: its output for a row never depends on the other rows in the batch.
- Unseen categories and constant columns never raise and never produce NaN.
- Never modifies its inputs.
