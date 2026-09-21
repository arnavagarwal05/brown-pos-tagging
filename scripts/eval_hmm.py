"""5-fold cross-validation of the Viterbi HMM tagger on Brown / universal tagset."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np

from postag.data import UNIVERSAL_TAGS, kfold, load_brown
from postag.evaluate import format_report, score
from postag.hmm import HMMTagger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--max-suffix", type=int, default=3)
    ap.add_argument("--out", default="results/hmm_cv.json")
    args = ap.parse_args()

    sents = load_brown()
    folds, t0 = [], time.time()
    for i, (train, test) in enumerate(kfold(sents, k=args.folds), 1):
        tagger = HMMTagger(UNIVERSAL_TAGS, max_suffix=args.max_suffix, alpha=args.alpha).fit(train)
        y_true, y_pred, unk = [], [], 0
        for s in test:
            words = [w for w, _ in s]
            unk += sum(w not in tagger.vocab for w in words)
            y_true += [t for _, t in s]
            y_pred += [t for _, t in tagger.tag(words)]
        f = score(y_true, y_pred)
        f["unknown_rate"] = unk / len(y_true)
        folds.append(f)
        print(f"fold {i}: {100*f['accuracy']:.2f}%  (unknown words {100*f['unknown_rate']:.2f}%)")
    print(f"\n{format_report('HMM (Viterbi)', folds)}\n({time.time()-t0:.0f}s total)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({
        "model": "hmm_viterbi", "alpha": args.alpha, "max_suffix": args.max_suffix, "folds": args.folds,
        "accuracy_mean": float(np.mean([f["accuracy"] for f in folds])),
        "accuracy_std": float(np.std([f["accuracy"] for f in folds])),
        "per_tag": {t: float(np.nanmean([f["per_tag"][t] for f in folds])) for t in UNIVERSAL_TAGS},
        "confusion": np.sum([f["confusion"] for f in folds], axis=0).tolist(),
        "tags": UNIVERSAL_TAGS,
    }, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
