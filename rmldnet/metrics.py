import numpy as np


class Confusion:
    def __init__(self):
        self.matrix = np.zeros((2, 2), dtype=np.int64)

    def update(self, pred, target):
        pred = np.asarray(pred, dtype=bool).ravel()
        target = np.asarray(target, dtype=bool).ravel()
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")
        self.matrix += np.bincount(
            2 * target.astype(int) + pred.astype(int), minlength=4
        ).reshape(2, 2)

    def compute(self):
        tn, fp, fn, tp = self.matrix.ravel().tolist()

        def ratio(a, b):
            return a / b if b else None

        return {
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy": ratio(tp + tn, tp + tn + fp + fn),
            "lane_iou": ratio(tp, tp + fp + fn),
            "miou": float(
                np.mean(
                    [
                        x
                        for x in [ratio(tn, tn + fp + fn), ratio(tp, tp + fp + fn)]
                        if x is not None
                    ]
                )
            ),
            "confusion": self.matrix.tolist(),
            "protocol": "binary pixel confusion; not official lane benchmark",
        }
