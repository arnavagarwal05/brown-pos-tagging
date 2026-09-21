"""Token-level accuracy, per-tag accuracy, and a confusion matrix."""

from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix

from .data import UNIVERSAL_TAGS


def score(y_true, y_pred, tags=UNIVERSAL_TAGS):
    y_true, y_pred = list(y_true), list(y_pred)
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    total, correct = Counter(y_true), Counter(t for t, p in zip(y_true, y_pred) if t == p)
    per_tag = {t: (correct[t] / total[t] if total[t] else float("nan")) for t in tags}
    cm = confusion_matrix(y_true, y_pred, labels=tags)
    return {"accuracy": acc, "per_tag": per_tag, "confusion": cm, "support": dict(total)}


def format_report(name, folds, tags=UNIVERSAL_TAGS):
    accs = [f["accuracy"] for f in folds]
    lines = [f"{name}: {100*np.mean(accs):.2f}% ± {100*np.std(accs):.2f}  ({len(folds)}-fold, token accuracy)", "", f"{'tag':6} {'acc':>7}  support"]
    support = Counter()
    for f in folds:
        support.update(f["support"])
    for t in tags:
        acc = np.nanmean([f["per_tag"][t] for f in folds])
        lines.append(f"{t:6} {100*acc:6.2f}%  {support[t]}")
    return "\n".join(lines)
