from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

__all__ = ["compute_features"]


def compute_features(matrix: sp.spmatrix | np.ndarray):
    """Return a **dict** of structural & numerical properties of *matrix*."""

    # ––– ensure CSR –––––––––––––––––––––––––––––––––––––––––––––––––––
    if not sp.isspmatrix(matrix):
        matrix = sp.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()

    rows, cols = matrix.shape
    nnz = matrix.nnz
    density = nnz / (rows * cols) * 100.0 if rows and cols else 0.0

    features: dict[str, float | int | bool | None] = {
        "num_rows": rows,
        "num_cols": cols,
        "num_nonzeros": nnz,
        "density_percent": density,
    }

    # ––– symmetry –––––––––––––––––––––––––––––––––––––––––––––––––––––
    psym, nsym = calculate_psym_nsym(matrix)
    features |= {"pattern_symmetry": psym, "numerical_symmetry": nsym}

    # ––– NNZ per‑row / per‑col stats ––––––––––––––––––––––––––––––––
    nnz_row = matrix.getnnz(axis=1)
    nnz_col = matrix.getnnz(axis=0)

    features |= {
        "nonzeros_per_row_min": int(nnz_row.min()),
        "nonzeros_per_row_max": int(nnz_row.max()),
        "normalized_nonzeros_per_row_min": float(
            nnz_row.min() / cols if cols else np.nan
        ),
        "normalized_nonzeros_per_row_max": float(
            nnz_row.max() / cols if cols else np.nan
        ),
        "nonzeros_per_row_mean": float(nnz_row.mean()),
        "nonzeros_per_row_median": int(np.median(nnz_row)),
        "nonzeros_per_row_std": float(nnz_row.std()),
        "normalized_nonzeros_per_row_std": float(
            nnz_row.std() / cols if cols else np.nan
        ),
        "nonzeros_per_col_min": int(nnz_col.min()),
        "nonzeros_per_col_max": int(nnz_col.max()),
        "normalized_nonzeros_per_col_min": float(
            nnz_col.min() / rows if rows else np.nan
        ),
        "normalized_nonzeros_per_col_max": float(
            nnz_col.max() / rows if rows else np.nan
        ),
        "nonzeros_per_col_mean": float(nnz_col.mean()),
        "nonzeros_per_col_median": int(np.median(nnz_col)),
        "nonzeros_per_col_std": float(nnz_col.std()),
        "normalized_nonzeros_per_col_std": float(
            nnz_col.std() / rows if rows else np.nan
        ),
    }

    # ––– value stats ––––––––––––––––––––––––––––––––––––––––––––––––
    data = matrix.data if matrix.data.size else np.array([0])
    features |= {
        "value_min": float(data.min()),
        "value_max": float(data.max()),
        "value_mean": float(data.mean()),
        "value_std": float(data.std()),
    }

    # ––– per‑row detailed stats ––––––––––––––––––––––––––––––––––––
    csr = matrix
    row_min = np.zeros(rows)
    row_max = np.zeros(rows)
    row_mean = np.zeros(rows)
    row_std = np.zeros(rows)
    row_median = np.zeros(rows)

    for i in range(rows):
        seg = csr.data[csr.indptr[i] : csr.indptr[i + 1]]
        if seg.size:
            row_min[i] = seg.min()
            row_max[i] = seg.max()
            row_mean[i] = seg.mean()
            row_std[i] = seg.std()
            row_median[i] = np.median(seg)

    features |= {
        "row_min_min": float(row_min.min()),
        "row_min_max": float(row_min.max()),
        "row_min_mean": float(row_min.mean()),
        "row_min_std": float(row_min.std()),
        "row_max_min": float(row_max.min()),
        "row_max_max": float(row_max.max()),
        "row_max_mean": float(row_max.mean()),
        "row_max_std": float(row_max.std()),
        "row_mean_min": float(row_mean.min()),
        "row_mean_max": float(row_mean.max()),
        "row_mean_mean": float(row_mean.mean()),
        "row_mean_std": float(row_mean.std()),
        "row_std_min": float(row_std.min()),
        "row_std_max": float(row_std.max()),
        "row_std_mean": float(row_std.mean()),
        "row_std_std": float(row_std.std()),
        "row_median_min": float(row_median.min()),
        "row_median_max": float(row_median.max()),
        "row_median_mean": float(row_median.mean()),
        "row_median_std": float(row_median.std()),
    }

    # ––– per‑column detailed stats ––––––––––––––––––––––––––––––––––
    csc = matrix.tocsc()
    col_min = np.zeros(cols)
    col_max = np.zeros(cols)
    col_mean = np.zeros(cols)
    col_std = np.zeros(cols)
    col_median = np.zeros(cols)

    for j in range(cols):
        seg = csc.data[csc.indptr[j] : csc.indptr[j + 1]]
        if seg.size:
            col_min[j] = seg.min()
            col_max[j] = seg.max()
            col_mean[j] = seg.mean()
            col_std[j] = seg.std()
            col_median[j] = np.median(seg)

    features |= {
        "col_min_min": float(col_min.min()),
        "col_min_max": float(col_min.max()),
        "col_min_mean": float(col_min.mean()),
        "col_min_std": float(col_min.std()),
        "col_max_min": float(col_max.min()),
        "col_max_max": float(col_max.max()),
        "col_max_mean": float(col_max.mean()),
        "col_max_std": float(col_max.std()),
        "col_mean_min": float(col_mean.min()),
        "col_mean_max": float(col_mean.max()),
        "col_mean_mean": float(col_mean.mean()),
        "col_mean_std": float(col_mean.std()),
        "col_std_min": float(col_std.min()),
        "col_std_max": float(col_std.max()),
        "col_std_mean": float(col_std.mean()),
        "col_std_std": float(col_std.std()),
        "col_median_min": float(col_median.min()),
        "col_median_max": float(col_median.max()),
        "col_median_mean": float(col_median.mean()),
        "col_median_std": float(col_median.std()),
    }

    # ––– diagonal distances –––––––––––––––––––––––––––––––––––––––
    bandwidth, profile = calculate_bandwidth_and_total_profile(matrix)
    features |= {
        "bandwidth": bandwidth,
        "normalized_bandwidth": bandwidth / cols,
        "bandwidth_std": np.std(bandwidth),
        "profile": profile,
        "normalized_profile": profile / cols,
    }

    row_idx, col_idx = matrix.nonzero()
    if row_idx.size:
        dist = np.abs(row_idx - col_idx)
        nnz_diagonal = np.count_nonzero(row_idx == col_idx)
        nnz_off_diagonal = nnz - nnz_diagonal

        features |= {
            "avg_distance_to_diagonal": float(dist.mean()),
            "num_diagonals_with_nonzeros": int(np.unique(dist).size),
            "nnz_bandwidth_std": float(dist.std()) if dist.size else 0.0,
            "nnz_diagonal": int(nnz_diagonal),
            "nnz_off_diagonal": int(nnz_off_diagonal),
        }

    else:
        features |= {
            "avg_distance_to_diagonal": 0.0,
            "num_diagonals_with_nonzeros": 0.0,
            "nnz_bandwidth_std": 0.0,
            "nnz_diagonal": 0.0,
            "nnz_off_diagonal": 0.0,
        }

    # ––– structural unsymmetry ––––––––––––––––––––––––––––––––––––
    unsym = set(zip(row_idx, col_idx)) - set(zip(col_idx, row_idx))
    features |= {"num_structurally_unsymmetric_elements": int(len(unsym))}

    # ––– norms ––––––––––––––––––––––––––––––––––––––––––––––––––––
    try:
        features |= {
            "norm_1": float(splinalg.norm(matrix, 1)),
            "norm_inf": float(splinalg.norm(matrix, np.inf)),
            "frobenius_norm": float(splinalg.norm(matrix)),
        }
    except Exception:
        features |= {
            "norm_1": np.nan,
            "norm_inf": np.nan,
            "frobenius_norm": np.nan,
        }

    # ––– 1‑norm condition estimate ––––––––––––––––––––––––––––––––
    try:
        features |= {
            "estimated_condition_number": float(splinalg.onenormest(matrix))
        }
    except Exception:
        features |= {"estimated_condition_number": np.nan}

    # Sparsity profile
    features |= {
        "num_empty_rows": int(np.sum(nnz_row == 0)),
        "num_empty_cols": int(np.sum(nnz_col == 0)),
    }

    # Sparsity skew: compare nonzero spread across rows and cols
    features |= {
        "row_sparsity_skew": float(nnz_row.std() / (nnz_row.mean() + 1e-8)),
        "col_sparsity_skew": float(nnz_col.std() / (nnz_col.mean() + 1e-8)),
    }

    # Row/Col entropy
    features |= {
        "row_nnz_entropy": float(entropy(nnz_row)),
        "col_nnz_entropy": float(entropy(nnz_col)),
    }

    return features


def calculate_psym_nsym(matrix):
    # Only defined for square matrices.
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        return 0.0, 0.0

    # Convert to COO format for easier iteration.
    A = matrix.tocoo()

    # Exclude diagonal entries.
    offdiag_mask = A.row != A.col
    rows = A.row[offdiag_mask]
    cols = A.col[offdiag_mask]
    data = A.data[offdiag_mask]

    # Group entries by unordered pair (min(i,j), max(i,j)).
    pairs = {}
    for i, j, v in zip(rows, cols, data):
        key = (min(i, j), max(i, j))
        if key not in pairs:
            pairs[key] = [None, None]
        if i < j:
            pairs[key][0] = v
        else:
            pairs[key][1] = v

    total_keys = len(
        pairs
    )  # each key represents one unique off-diagonal position (could be 1 or 2 entries)
    complete_count = 0  # keys where both entries exist
    matching_numeric = 0  # complete keys where the values match

    for key, (v1, v2) in pairs.items():
        if v1 is not None and v2 is not None:
            complete_count += 1
            if np.isclose(v1, v2, atol=1e-64):
                matching_numeric += 1

    # Total off-diagonal entries:
    # Each complete pair contributes 2 entries; each incomplete pair contributes 1.
    total_offdiag = 2 * complete_count + (total_keys - complete_count)

    # Pattern symmetry: mirrored entries / total off-diagonals.
    psym = (2 * complete_count / total_offdiag) if total_offdiag > 0 else 0.0

    # Numerical symmetry: only consider complete pairs.
    nsym = (matching_numeric / complete_count) if complete_count > 0 else 0.0

    return psym, nsym


def calculate_bandwidth_and_total_profile(matrix):
    # Convert matrix to coordinate format for easy access to nonzero indices.
    coo = matrix.tocoo()
    rows = coo.row
    cols = coo.col
    n_rows, n_cols = matrix.shape

    # Bandwidth is defined as max(|i - j|) for all nonzero positions.
    bandwidth = np.max(np.abs(rows - cols)) if rows.size else 0

    # For the total profile, we compute the row-wise lower and upper profiles.
    # Lower profile: For each row i, difference between i and the minimum column index with a nonzero.
    # Upper profile: For each row i, difference between the maximum column index with a nonzero and i.

    # Initialize dictionaries for the first and last nonzero column indices for each row.
    # If no nonzero exists in a row, we keep default values.
    min_nonzero = {
        i: n_cols for i in range(n_rows)
    }  # default: no entry in row (n_cols is "infinite")
    max_nonzero = {
        i: -1 for i in range(n_rows)
    }  # default: no entry in row (-1 means no nonzero)

    # Update the dictionaries with the actual values.
    for i, j in zip(rows, cols):
        if j < min_nonzero[i]:
            min_nonzero[i] = j
        if j > max_nonzero[i]:
            max_nonzero[i] = j

    # Now accumulate the profile for each row.
    lower_profile = 0
    upper_profile = 0
    for i in range(n_rows):
        # Only if there is at least one nonzero entry in the row.
        if min_nonzero[i] < n_cols:
            lower_profile += i - min_nonzero[i]
        if max_nonzero[i] >= 0:
            upper_profile += max_nonzero[i] - i

    total_profile = lower_profile + upper_profile
    return bandwidth, total_profile


def entropy(x):
    p = x / np.sum(x)
    return -np.sum(p * np.log2(p + 1e-12))


def compute_average_features(features_list: list[dict]):
    """Element‑wise average of numeric entries across *features_list*."""
    if not features_list:
        return {}

    avg: dict[str, float | int | bool | None] = {}
    keys = features_list[0].keys()
    for k in keys:
        v0 = features_list[0][k]
        if isinstance(v0, (int, float, np.number, bool)):
            avg[k] = float(np.mean([f[k] for f in features_list]))
        else:
            avg[k] = v0
    return avg
