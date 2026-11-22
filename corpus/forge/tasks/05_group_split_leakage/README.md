# Leak-free train/validation split

A recommendation model scores far better on validation than in production. The cause: the split in `starter.py` shuffles individual **rows**, so the same user's interactions land in both train and validation. The model is graded partly on users it has already seen.

**Implement** `group_split(groups, val_fraction, seed)`, where `groups` is a 1-D integer array giving each row's user id. Return `(train_idx, val_idx)` as integer index arrays such that:

- **Partition:** every row index appears in exactly one of the two arrays.
- **No leakage:** no user id appears in both train and validation.
- **Size:** validation holds within ±0.05 of `val_fraction` of the **rows** (users have different numbers of rows).
- **Reproducible:** the same `seed` gives identical output, and different seeds give different splits.
- **Random:** which users go to validation must not depend on the order or value of their ids.
