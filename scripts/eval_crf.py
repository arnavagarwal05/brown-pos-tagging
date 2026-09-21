"""Train the stacked CRF on an 80/20 sentence split and report both stages."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

from sklearn.model_selection import train_test_split

from postag.crf import StackedCRF
from postag.data import UNIVERSAL_TAGS, load_brown
from postag.evaluate import format_report, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="results/crf_test.json")
    args = ap.parse_args()

    sents = load_brown()
    train, test = train_test_split(sents, test_size=args.test_size, random_state=args.seed)
    t0 = time.time()
    crf = StackedCRF(args.model_dir).fit(train)
    print(f"trained both stages on {len(train)} sentences in {time.time()-t0:.0f}s")

    out = {"train_sentences": len(train), "test_sentences": len(test), "seed": args.seed}
    for name, fn in (("stage1", crf.tag_stage1), ("stacked", crf.tag)):
        y_true, y_pred = [], []
        for s in test:
            y_true += [t for _, t in s]
            y_pred += [t for _, t in fn([w for w, _ in s])]
        f = score(y_true, y_pred)
        print(f"\n{format_report(f'CRF {name}', [f])}")
        out[name] = {"accuracy": f["accuracy"], "per_tag": f["per_tag"], "confusion": f["confusion"].tolist()}
    out["tags"] = UNIVERSAL_TAGS
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
