"""
eval.py  —  standalone evaluation at node-0
=============================================
Reads results saved by rank-0 during training and produces a full report
covering training speed, accuracy, and straggler detection quality.

Reads from:
    results/algo_metrics.json        (written by train.py)
    results/baseline_metrics.json    (written by train_baseline.py, optional)
    results/detection_metrics.json   (written by train.py, optional)

Writes to:
    results/eval_report.txt
    results/eval_plots/  (PNG plots)

Run:
    python eval.py
    python eval.py --results_dir ./results --target_acc 80.0
"""

import os
import json
import argparse


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def epoch_times(metrics):
    if 'epoch_train_s' in metrics[0]:
        return [r['epoch_train_s'] for r in metrics]
    return [metrics[0]['cumulative_train_s']] + [
        metrics[i]['cumulative_train_s'] - metrics[i-1]['cumulative_train_s']
        for i in range(1, len(metrics))
    ]


def time_to_accuracy(metrics, target_acc):
    for r in metrics:
        if r['test_acc'] >= target_acc:
            return r['cumulative_train_s'], r['epoch']
    return None, None


def compute_speed(metrics, dataset_size):
    et = epoch_times(metrics)
    avg_t = sum(et) / len(et)
    return {
        'total_train_time_s': round(metrics[-1]['cumulative_train_s'], 2),
        'avg_epoch_time_s':   round(avg_t, 3),
        'min_epoch_time_s':   round(min(et), 3),
        'max_epoch_time_s':   round(max(et), 3),
        'avg_throughput_sps': round(dataset_size / avg_t, 1) if avg_t > 0 else 0,
        'epoch_times':        [round(t, 3) for t in et],
    }


def build_report(algo, baseline, detection, sa, sb, target_acc):
    sep = '═' * 62
    lines = [sep, '  TRAINING EVALUATION REPORT', sep, '']

    lines += ['  TRAINING SPEED', '  ' + '─' * 58]
    hdr = f"  {'Metric':<38} {'Algorithm':>10}"
    if baseline:
        hdr += f"  {'Baseline':>10}"
    lines.append(hdr)

    def row(label, av, bv=None, fmt='{:.2f}'):
        s = f"  {label:<38} {fmt.format(av):>10}"
        if bv is not None:
            s += f"  {fmt.format(bv):>10}"
        return s

    lines.append(row('Total training time (s)',   sa['total_train_time_s'], sb.get('total_train_time_s') if baseline else None))
    lines.append(row('Avg time per epoch (s)',     sa['avg_epoch_time_s'],   sb.get('avg_epoch_time_s')   if baseline else None, fmt='{:.3f}'))
    lines.append(row('Avg throughput (samples/s)', sa['avg_throughput_sps'], sb.get('avg_throughput_sps') if baseline else None, fmt='{:.1f}'))

    if baseline and sa['total_train_time_s'] > 0:
        lines.append(f"\n  → Overall speedup : {sb['total_train_time_s']/sa['total_train_time_s']:.3f}x")

    lines += ['', '  ACCURACY', '  ' + '─' * 58]
    lines.append(row('Final test accuracy (%)',  algo[-1]['test_acc'],  baseline[-1]['test_acc']  if baseline else None))
    lines.append(row('Final train accuracy (%)', algo[-1]['train_acc'], baseline[-1]['train_acc'] if baseline else None))

    lines.append(f"\n  Time to reach {target_acc:.0f}% test accuracy:")
    at, ae = time_to_accuracy(algo, target_acc)
    lines.append(f"    Algorithm : {at:.1f}s (epoch {ae})" if at else f"    Algorithm : did not reach {target_acc}%")
    if baseline:
        bt, be = time_to_accuracy(baseline, target_acc)
        lines.append(f"    Baseline  : {bt:.1f}s (epoch {be})" if bt else f"    Baseline  : did not reach {target_acc}%")
        if at and bt:
            lines.append(f"    → Algorithm is {bt/at:.3f}x faster to {target_acc:.0f}%")

    if detection:
        d = detection[-1]
        lines += ['', '  STRAGGLER DETECTION', '  ' + '─' * 58]
        lines.append(f"  {'Precision':<38} {d['precision']:>10.4f}")
        lines.append(f"  {'Recall':<38} {d['recall']:>10.4f}")
        lines.append(f"  {'F1':<38} {d['f1']:>10.4f}")
        lines.append(f"  {'Accuracy':<38} {d['accuracy']:>10.4f}")
        fp = d.get('cumulative_FP', d.get('FP', 'N/A'))
        lines.append(f"  Confusion — TP:{d['TP']}  FP:{fp}  TN:{d['TN']}  FN:{d['FN']}")
        if d.get('avg_detection_latency_batches'):
            lines.append(f"  Avg detection latency : {d['avg_detection_latency_batches']:.1f} batches")
        if d.get('avg_recovery_latency_batches'):
            lines.append(f"  Avg recovery  latency : {d['avg_recovery_latency_batches']:.1f} batches")

    lines += ['', sep]
    return '\n'.join(lines)


def plot_eval(algo, baseline, detection, sa, sb, results_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skip plots. pip install matplotlib")
        return

    plot_dir = os.path.join(results_dir, 'eval_plots')
    os.makedirs(plot_dir, exist_ok=True)

    def save(fig, name):
        path = os.path.join(plot_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    has_base  = baseline is not None
    a_epoch   = [r['epoch']             for r in algo]
    a_cumtime = [r['cumulative_train_s'] for r in algo]
    a_test    = [r['test_acc']           for r in algo]
    a_loss    = [r['train_loss']         for r in algo]

    # Plot 1: test accuracy vs epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_epoch, a_test, label='Algorithm', color='tomato', linewidth=1.5)
    if has_base:
        ax.plot([r['epoch'] for r in baseline], [r['test_acc'] for r in baseline],
                label='Baseline', color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Test Accuracy vs Epoch')
    ax.legend(); ax.grid(True, alpha=0.3); save(fig, 'acc_vs_epoch.png')

    # Plot 2: epoch vs training time (x=time, y=epoch)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_cumtime, a_epoch, label='Algorithm', color='tomato', linewidth=1.5)
    if has_base:
        ax.plot([r['cumulative_train_s'] for r in baseline], [r['epoch'] for r in baseline],
                label='Baseline', color='steelblue', linewidth=1.5)
    ax.set_xlabel('Cumulative Training Time (s)'); ax.set_ylabel('Epoch')
    ax.set_title('Epochs Completed vs Training Time')
    ax.legend(); ax.grid(True, alpha=0.3); save(fig, 'epoch_vs_time.png')

    # Plot 3: training time per epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_epoch, sa['epoch_times'], label='Algorithm', color='tomato', linewidth=1.5)
    if has_base:
        ax.plot([r['epoch'] for r in baseline], sb['epoch_times'],
                label='Baseline', color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Time (s)'); ax.set_title('Training Time per Epoch')
    ax.legend(); ax.grid(True, alpha=0.3); save(fig, 'time_per_epoch.png')

    # Plot 4: train loss vs epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_epoch, a_loss, label='Algorithm', color='tomato', linewidth=1.5)
    if has_base:
        ax.plot([r['epoch'] for r in baseline], [r['train_loss'] for r in baseline],
                label='Baseline', color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss'); ax.set_title('Train Loss vs Epoch')
    ax.legend(); ax.grid(True, alpha=0.3); save(fig, 'loss_vs_epoch.png')

    # Plot 5: detection TP vs FP + P/R/F1
    if detection:
        det_epochs = [r['epoch'] for r in detection]
        fp_per_epoch = [r.get('epoch_FP', r.get('FP', 0)) for r in detection]
        tp_per_epoch = [r.get('epoch_TP', r.get('TP', 0)) for r in detection]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(det_epochs)); w = 0.4
        axes[0].bar(x - w/2, tp_per_epoch, w, label='True Positive (real straggler)', color='seagreen', alpha=0.8)
        axes[0].bar(x + w/2, fp_per_epoch, w, label='False Positive (wrong detection)', color='tomato',   alpha=0.8)
        axes[0].set_xticks(x[::max(1, len(x)//20)])
        axes[0].set_xticklabels(det_epochs[::max(1, len(det_epochs)//20)])
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Detection Count')
        axes[0].set_title('Straggler Detections: Correct vs False Alarms')
        axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].plot(det_epochs, [r['precision'] for r in detection], label='Precision', color='steelblue', linewidth=1.5)
        axes[1].plot(det_epochs, [r['recall']    for r in detection], label='Recall',    color='tomato',    linewidth=1.5)
        axes[1].plot(det_epochs, [r['f1']        for r in detection], label='F1',        color='seagreen',  linewidth=1.5)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score'); axes[1].set_title('Detection Metrics per Epoch')
        axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.tight_layout(); save(fig, 'detection_eval.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',  default=os.environ.get('RESULTS_DIR', './results'))
    parser.add_argument('--target_acc',   type=float, default=80.0)
    parser.add_argument('--dataset_size', type=int,   default=50000)
    parser.add_argument('--no_plot',      action='store_true')
    args = parser.parse_args()

    algo      = load_json(os.path.join(args.results_dir, 'algo_metrics.json'))
    baseline  = load_json(os.path.join(args.results_dir, 'baseline_metrics.json'))
    detection = load_json(os.path.join(args.results_dir, 'detection_metrics.json'))

    if algo is None:
        print(f"ERROR: algo_metrics.json not found in {args.results_dir}")
        return

    sa = compute_speed(algo,     args.dataset_size)
    sb = compute_speed(baseline, args.dataset_size) if baseline else {}

    report = build_report(algo, baseline, detection, sa, sb, args.target_acc)
    print(report)

    path = os.path.join(args.results_dir, 'eval_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {path}")

    if not args.no_plot:
        print("\nGenerating plots ...")
        plot_eval(algo, baseline, detection, sa, sb, args.results_dir)


if __name__ == '__main__':
    main()
