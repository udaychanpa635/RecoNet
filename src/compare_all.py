"""
Runs the popularity baseline, TF matrix factorization, and PyTorch two-tower
model on the SAME temporal split and prints a comparison table.

IMPORTANT: TensorFlow and PyTorch loaded in the same process can segfault
due to conflicting native runtime libraries (both ship their own OpenMP /
MKL runtimes). Rather than fight that, each model runs in its own
subprocess — a legitimate real-world pattern for multi-framework pipelines,
not a workaround to be embarrassed about. This is worth mentioning in an
interview: it's exactly the kind of practical friction that only shows up
once you actually run mixed-framework code, not in a tutorial.

    python src/compare_all.py
"""
import re
import subprocess
import sys
import os

SRC_DIR = os.path.dirname(__file__)
METRIC_PATTERN = re.compile(r"(precision@10|recall@10|ndcg@10):\s*([\d.]+)")


def run_and_parse(script_name: str) -> dict:
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, script_name)],
        capture_output=True, text=True,
    )
    metrics = {}
    for line in result.stdout.splitlines():
        match = METRIC_PATTERN.search(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))

    if not metrics:
        print(f"  [warning] Could not parse metrics from {script_name}. "
              f"Last 500 chars of output:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
    return metrics


def main():
    print("Running popularity baseline (from src/data_utils.py)...")
    pop_metrics = run_and_parse("data_utils.py")

    print("Running TF matrix factorization (isolated subprocess)...")
    tf_metrics = run_and_parse("tf_mf_model.py")

    print("Running PyTorch two-tower (isolated subprocess)...")
    tt_metrics = run_and_parse("pytorch_two_tower.py")

    print("\n" + "=" * 70)
    print(f"{'Metric':<15} {'Popularity':>15} {'TF Matrix-Fact':>18} {'PyTorch Two-Tower':>20}")
    print("=" * 70)
    for metric_name in ["precision@10", "recall@10", "ndcg@10"]:
        print(f"{metric_name:<15} {pop_metrics.get(metric_name, float('nan')):>15.4f} "
              f"{tf_metrics.get(metric_name, float('nan')):>18.4f} "
              f"{tt_metrics.get(metric_name, float('nan')):>20.4f}")
    print("=" * 70)
    print("\nNote: on the tiny synthetic dataset bundled with this repo, differences")
    print("between models are small / noisy — that's expected at this scale.")
    print("Run this on a real dataset (see notebooks/RecoNet_Training_Colab.ipynb)")
    print("with real sentence-transformer content embeddings for meaningful gaps.")


if __name__ == "__main__":
    main()
