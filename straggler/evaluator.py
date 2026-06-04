"""
straggler/evaluator.py
-----------------------
Tracks per-batch straggler detection accuracy by comparing:
  ground truth : injector.is_sleeping  (was worker actually a straggler?)
  prediction   : detector.amp_flag     (did the detector fire?)

Metrics:
  Precision  = TP / (TP + FP)   how often detector fires correctly
  Recall     = TP / (TP + FN)   how often actual stragglers are caught
  F1         = harmonic mean of precision and recall
  Accuracy   = (TP + TN) / total
  Avg detection latency  = batches from sleep-start until AMP activates
  Avg recovery  latency  = batches from sleep-end   until AMP deactivates
"""

import json
import os
from typing import Optional


class DetectionEvaluator:

    def __init__(self) -> None:
        self.TP = 0   # sleeping=True,  amp=True
        self.FP = 0   # sleeping=False, amp=True   (false alarm)
        self.TN = 0   # sleeping=False, amp=False
        self.FN = 0   # sleeping=True,  amp=False  (missed)

        self._sleep_start_batch:  Optional[int] = None
        self._detected_batch:     Optional[int] = None
        self._wake_batch:         Optional[int] = None
        self._recovered_batch:    Optional[int] = None

        self._detection_latencies: list = []
        self._recovery_latencies:  list = []
        self._batch_idx: int = 0

        self.epoch_metrics: list = []

    # ─────────────────────────────────────────────────────────────────────────
    def record(self, actually_sleeping: bool, amp_active: bool) -> None:
        """Call once per non-boundary batch after the training step."""
        # confusion matrix
        if actually_sleeping and amp_active:
            self.TP += 1
        elif not actually_sleeping and amp_active:
            self.FP += 1
        elif not actually_sleeping and not amp_active:
            self.TN += 1
        else:
            self.FN += 1

        # detection latency
        prev_sleeping = (self._sleep_start_batch is not None and
                         self._wake_batch is None)

        if actually_sleeping and not prev_sleeping:
            self._sleep_start_batch = self._batch_idx
            self._detected_batch    = None

        if (actually_sleeping and amp_active and
                self._detected_batch is None and
                self._sleep_start_batch is not None):
            self._detected_batch = self._batch_idx
            self._detection_latencies.append(
                self._detected_batch - self._sleep_start_batch)

        # recovery latency
        if not actually_sleeping and prev_sleeping:
            self._wake_batch      = self._batch_idx
            self._recovered_batch = None

        if (not actually_sleeping and not amp_active and
                self._wake_batch is not None and
                self._recovered_batch is None):
            self._recovered_batch = self._batch_idx
            self._recovery_latencies.append(
                self._recovered_batch - self._wake_batch)
            self._sleep_start_batch = None
            self._wake_batch        = None

        self._batch_idx += 1

    # ─────────────────────────────────────────────────────────────────────────
    def compute(self) -> dict:
        total     = self.TP + self.FP + self.TN + self.FN
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0
        recall    = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (self.TP + self.TN) / total if total > 0 else 0.0

        avg_det = (sum(self._detection_latencies) / len(self._detection_latencies)
                   if self._detection_latencies else None)
        avg_rec = (sum(self._recovery_latencies)  / len(self._recovery_latencies)
                   if self._recovery_latencies  else None)

        return {
            'TP': self.TP, 'FP': self.FP,
            'TN': self.TN, 'FN': self.FN,
            'precision':                     round(precision, 4),
            'recall':                        round(recall,    4),
            'f1':                            round(f1,        4),
            'accuracy':                      round(accuracy,  4),
            'avg_detection_latency_batches': round(avg_det, 2) if avg_det else None,
            'avg_recovery_latency_batches':  round(avg_rec, 2) if avg_rec else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def log_epoch(self, epoch: int) -> dict:
        """Snapshot cumulative metrics at end of epoch and track per-epoch FP/TP."""
        prev_FP = self.epoch_metrics[-1]['cumulative_FP'] if self.epoch_metrics else 0
        prev_TP = self.epoch_metrics[-1]['TP']            if self.epoch_metrics else 0
        m = self.compute()
        m['epoch']        = epoch
        m['cumulative_FP'] = m.pop('FP')
        m['epoch_FP']      = m['cumulative_FP'] - prev_FP
        m['epoch_TP']      = m['TP'] - prev_TP
        self.epoch_metrics.append(m)
        return m

    def report(self, logger) -> None:
        m = self.compute()
        logger.info("── Detection Accuracy ──────────────────────────────")
        logger.info(f"   TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")
        logger.info(f"   Precision : {m['precision']:.4f}")
        logger.info(f"   Recall    : {m['recall']:.4f}")
        logger.info(f"   F1        : {m['f1']:.4f}")
        logger.info(f"   Accuracy  : {m['accuracy']:.4f}")
        if m['avg_detection_latency_batches'] is not None:
            logger.info(f"   Avg detection latency : "
                        f"{m['avg_detection_latency_batches']:.1f} batches")
        if m['avg_recovery_latency_batches'] is not None:
            logger.info(f"   Avg recovery  latency : "
                        f"{m['avg_recovery_latency_batches']:.1f} batches")
        logger.info("────────────────────────────────────────────────────")

    def save(self, results_dir: str) -> None:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, 'detection_metrics.json')
        with open(path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
