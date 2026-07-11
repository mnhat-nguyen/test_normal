"""
compare.py  –  compare baseline vs algorithm results
======================================================
Reads results/baseline_metrics.json and results/algo_metrics.json
produced by train_baseline.py and train.py respectively.

Prints:
  • Per-epoch table (time, train acc, test acc for both)
  • Summary: final accuracy, total training time, time-to-accuracy

Plots (saved to results/):
  • test_acc  vs cumulative training time
  • test_acc  vs epoch
  • train_loss vs epoch

Run:
    python compare.py

Optional args:
    python compare.py --target_acc 80.0   (time-to-accuracy threshold)
    python compare.py --results_dir ./my_results
"""

import os
import json
import argparse


def load(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run the corresponding training script first."
        )
    with open(path) as f:
        return json.load(f)


def time_to_accuracy(metrics: list, target_acc: float):
    """Return cumulative_train_s when test_acc first reaches target_acc."""
    for row in metrics:
        if row['test_acc'] >= target_acc:
            return row['cumulative_train_s'], row['epoch']
    return None, None


def print_table(baseline: list, algo: list) -> None:
    n = min(len(baseline), len(algo))
    header = (f"{'Epoch':>5}  "
              f"{'Base time(s)':>12}  {'Base testAcc':>12}  "
              f"{'Algo time(s)':>12}  {'Algo testAcc':>12}  "
              f"{'Δ testAcc':>9}  {'Speedup':>7}")
    print("\n" + header)
    print("─" * len(header))

    for i in range(n):
        b, a = baseline[i], algo[i]
        delta    = a['test_acc'] - b['test_acc']
        # time speedup: how much faster algo reached same epoch
        speedup  = b['cumulative_train_s'] / a['cumulative_train_s'] if a['cumulative_train_s'] > 0 else 0
        print(
            f"{b['epoch']:>5}  "
            f"{b['cumulative_train_s']:>12.1f}  {b['test_acc']:>11.2f}%  "
            f"{a['cumulative_train_s']:>12.1f}  {a['test_acc']:>11.2f}%  "
            f"{delta:>+8.2f}%  {speedup:>6.3f}x"
        )


def print_summary(baseline: list, algo: list, target_acc: float) -> None:
    b_last = baseline[-1]
    a_last = algo[-1]

    b_time, b_epoch = time_to_accuracy(baseline, target_acc)
    a_time, a_epoch = time_to_accuracy(algo,     target_acc)

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)

    print(f"\n  {'Metric':<35} {'Baseline':>10}  {'Algorithm':>10}")
    print(f"  {'─'*55}")
    print(f"  {'Final test accuracy':<35} {b_last['test_acc']:>9.2f}%  {a_last['test_acc']:>9.2f}%")
    print(f"  {'Final train accuracy':<35} {b_last['train_acc']:>9.2f}%  {a_last['train_acc']:>9.2f}%")
    print(f"  {'Total training time (s)':<35} {b_last['cumulative_train_s']:>10.1f}  {a_last['cumulative_train_s']:>10.1f}")
    print(f"  {'Epochs completed':<35} {b_last['epoch']+1:>10}  {a_last['epoch']+1:>10}")

    print(f"\n  Time-to-{target_acc:.0f}% test accuracy:")
    if b_time is not None:
        print(f"    Baseline  : {b_time:.1f}s  (epoch {b_epoch})")
    else:
        print(f"    Baseline  : never reached {target_acc}% within training")
    if a_time is not None:
        print(f"    Algorithm : {a_time:.1f}s  (epoch {a_epoch})")
    else:
        print(f"    Algorithm : never reached {target_acc}% within training")

    if b_time and a_time:
        speedup = b_time / a_time
        saved   = b_time - a_time
        print(f"\n  → Algorithm is {speedup:.3f}x faster to reach {target_acc:.0f}%")
        print(f"  → Time saved : {saved:.1f}s")

    overall_speedup = (b_last['cumulative_train_s'] /
                       a_last['cumulative_train_s']
                       if a_last['cumulative_train_s'] > 0 else 0)
    print(f"\n  → Overall training speedup : {overall_speedup:.3f}x")
    print(f"  → Final accuracy delta     : "
          f"{a_last['test_acc'] - b_last['test_acc']:+.2f}%")
    print("═" * 60 + "\n")


def plot_results(baseline: list, algo: list, results_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots.")
        print("  Install with:  pip install matplotlib")
        return

    os.makedirs(results_dir, exist_ok=True)

    b_time  = [r['cumulative_train_s'] for r in baseline]
    a_time  = [r['cumulative_train_s'] for r in algo]
    b_epoch = [r['epoch']    for r in baseline]
    a_epoch = [r['epoch']    for r in algo]

    b_test  = [r['test_acc']   for r in baseline]
    a_test  = [r['test_acc']   for r in algo]
    b_train = [r['train_loss'] for r in baseline]
    a_train = [r['train_loss'] for r in algo]

    # ── Plot 1: test accuracy vs training time ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b_time, b_test, label='Baseline (FP32)', color='steelblue', linewidth=1.5)
    ax.plot(a_time, a_test, label='Algorithm (EWMA+AMP)', color='tomato',  linewidth=1.5)
    ax.set_xlabel('Cumulative Training Time (s)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(results_dir, 'acc_vs_time.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Plot 2: test accuracy vs epoch ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b_epoch, b_test, label='Baseline (FP32)', color='steelblue', linewidth=1.5)
    ax.plot(a_epoch, a_test, label='Algorithm (EWMA+AMP)', color='tomato',  linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(results_dir, 'acc_vs_epoch.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Plot 3: train loss vs epoch ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b_epoch, b_train, label='Baseline (FP32)', color='steelblue', linewidth=1.5)
    ax.plot(a_epoch, a_train, label='Algorithm (EWMA+AMP)', color='tomato',  linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Train Loss vs Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(results_dir, 'loss_vs_epoch.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',  default='./results')
    parser.add_argument('--target_acc',   type=float, default=80.0,
                        help='Accuracy threshold for time-to-accuracy metric')
    parser.add_argument('--no_plot',      action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()

    baseline_path = os.path.join(args.results_dir, 'amp_disabled_metrics.json')
    algo_path     = os.path.join(args.results_dir, 'algo_metrics.json')

    print(f"\nLoading baseline : {baseline_path}")
    print(f"Loading algorithm: {algo_path}")

    baseline = load(baseline_path)
    algo     = load(algo_path)

    print_table(baseline, algo)
    print_summary(baseline, algo, target_acc=args.target_acc)

    if not args.no_plot:
        print("Generating plots ...")
        plot_results(baseline, algo, args.results_dir)


if __name__ == '__main__':
    main()
