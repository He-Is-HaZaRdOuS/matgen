import multiprocessing as mp
import os
import sys
import time

from matrixgen import RESIZE_METHODS, load_matrix, resize_matrix, save_matrix

NUM_CORES = 8
INPUT_DIR = "matrices"
OUTPUT_DIR = "synthetic_generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Downscale-only methods
DOWNSCALE_METHODS = {
    name: meta["fn"]
    for name, meta in RESIZE_METHODS.items()
    if not meta["can_upscale"]
}

# Methods that can upscale (and usually downscale too)
UPSCALE_METHODS = {
    name: meta["fn"]
    for name, meta in RESIZE_METHODS.items()
    if meta["can_upscale"]
}


def process_variant(task):
    sys.stdout = open("output.log", "a", buffering=1)
    sys.stderr = sys.stdout
    method, matrix_filename, variant_type, new_dim = task
    orig_name = os.path.splitext(matrix_filename)[0]
    filepath = os.path.join(INPUT_DIR, matrix_filename)

    try:
        matrix = load_matrix(filepath)
    except Exception as e:
        print(f"[ERROR] Failed to load {matrix_filename}: {e}")
        return

    print(
        f"[INFO] Processing {matrix_filename} with {method}_{variant_type}_{new_dim}x{new_dim}"
    )
    start_time = time.time()
    scaled = resize_matrix(matrix, new_dim, method)
    end_time = time.time()
    elapsed_time = end_time - start_time

    new_name = (
        f"{orig_name}_synthetic_{method}_{variant_type}_"
        f"{new_dim}x{new_dim}_matchnnz_nnz{scaled.nnz}.mtx"
    )
    save_matrix_with_header(scaled, new_name, OUTPUT_DIR, elapsed_time)


def save_matrix_with_header(matrix, filename, output_dir, elapsed_time):
    filepath = os.path.join(output_dir, filename)
    # Save normally first
    save_matrix(matrix, filename, output_dir)

    # Now re-open, read, and re-write with header in correct spot
    with open(filepath, "r+") as f:
        lines = f.readlines()
        if not lines or not lines[0].startswith("%%MatrixMarket"):
            raise ValueError("Not a Matrix Market file. Missing banner.")

        # Insert elapsed_time as comment right after the banner
        lines.insert(1, f"% Generated in {elapsed_time:.6f} seconds\n")
        f.seek(0)
        f.writelines(lines)


def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mtx")]

    tasks = []
    for filename in files:
        orig_name = os.path.splitext(filename)[0]

        # Downscale variants
        for method in DOWNSCALE_METHODS:
            for variant_type, new_dim in [
                ("half", 0),  # dummy, to be replaced by real dims
                ("decrement", 0),
            ]:
                tasks.append((method, filename, variant_type, new_dim))

        # Upscale variants
        for method in UPSCALE_METHODS:
            for variant_type, new_dim in [
                ("quadruple", 0),
                ("double", 0),
                ("increment", 0),
                ("half", 0),
                ("decrement", 0),
            ]:
                tasks.append((method, filename, variant_type, new_dim))

    # Fix dims for real tasks
    final_tasks = []
    for method, filename, variant_type, _ in tasks:
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            matrix = load_matrix(filepath)
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

        orig_dim = matrix.shape[0]
        new_dim = {
            "quadruple": orig_dim * 4,
            "double": orig_dim * 2,
            "increment": orig_dim + 1,
            "half": max(1, orig_dim // 2),
            "decrement": max(1, orig_dim - 1),
        }[variant_type]

        final_tasks.append((method, filename, variant_type, new_dim))

    with mp.get_context("spawn").Pool(NUM_CORES) as pool:
        pool.map(process_variant, final_tasks)


if __name__ == "__main__":
    sys.stdout = open("output.log", "w", buffering=1)
    main()
    sys.exit(0)
