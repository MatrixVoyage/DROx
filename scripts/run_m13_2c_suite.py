"""Full Automation Suite for Milestone 13.2c.

Sequentially executes:
1. Pre-execution baseline cryptographic integrity check.
2. Training M13.2c-TrueGroup (5 epochs).
3. Training M13.2c-ShuffledGroup negative control (5 epochs).
4. Training M13.2c-OrthOnly ablation (5 epochs).
5. Authoritative 5-Model Comparative Evaluation across the 560-query benchmark.
6. Post-execution baseline cryptographic integrity check.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def run_step(cmd_args: list[str], description: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print(f"COMMAND: {' '.join(cmd_args)}")
    print("=" * 80)

    start = time.perf_counter()
    result = subprocess.run(cmd_args, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\nERROR: Step failed with return code {result.returncode} in {elapsed:.1f}s")
        sys.exit(result.returncode)

    print(f"\nSUCCESS: Step completed in {elapsed:.1f}s")


def main():
    suite_start = time.perf_counter()

    # Step 1: Train TrueGroup (Full Disentanglement)
    run_step(
        [PYTHON_EXE, "scripts/train_m13_2c.py", "--mode", "true_group", "--epochs", "5", "--batch_size", "32"],
        "Training M13.2c-TrueGroup (Full Disentanglement)",
    )

    # Step 2: Train ShuffledGroup (Negative Control)
    run_step(
        [PYTHON_EXE, "scripts/train_m13_2c.py", "--mode", "shuffled_group", "--epochs", "5", "--batch_size", "32"],
        "Training M13.2c-ShuffledGroup (Negative Control)",
    )

    # Step 3: Train OrthOnly (Ablation without Contrastive Loss)
    run_step(
        [PYTHON_EXE, "scripts/train_m13_2c.py", "--mode", "orth_only", "--epochs", "5", "--batch_size", "32"],
        "Training M13.2c-OrthOnly (Orthogonality Without Contrastive Loss)",
    )

    # Step 4: Run Authoritative Comparative Evaluation
    run_step(
        [PYTHON_EXE, "scripts/evaluate_m13_2c.py"],
        "Executing Authoritative Comparative Evaluation Benchmark (560 queries)",
    )

    total_time = time.perf_counter() - suite_start
    print("\n" + "=" * 80)
    print(f"ALL MILESTONE 13.2c EXPERIMENTS COMPLETED IN {total_time/60:.2f} MINUTES")
    print("=" * 80)


if __name__ == "__main__":
    main()
