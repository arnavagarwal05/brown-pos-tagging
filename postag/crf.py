"""Two-stage stacked CRF tagger (python-crfsuite).

Stage 1 tags a sentence from lexical and positional features. Stage 2 sees the
same features plus stage 1's predicted tag for the previous word, so the linear
chain gets a second, learned view of left context on top of its own transitions.
"""

import os

import pycrfsuite
from nltk.stem import PorterStemmer

_stem = PorterStemmer().stem


def word_features(words, i):
    w = words[i]
    stem = _stem(w)
    return {
        "stem": stem,
        "suffix": w[len(stem):],
        "len": len(w),
        "pos": i + 1,
        "prev": words[i - 1] if i > 0 else "^",
        "next": words[i + 1] if i < len(words) - 1 else "$",
        "first": int(i == 0),
        "last": int(i == len(words) - 1),
        "cap": int(w[:1].isupper()),
    }


def sentence_features(words, prev_tags=None):
    feats = [word_features(words, i) for i in range(len(words))]
    if prev_tags is not None:
        for i, f in enumerate(feats):
            f["prev_tag"] = prev_tags[i - 1] if i > 0 else "^"
    return feats


class StackedCRF:
    def __init__(self, model_dir="models", c1=0.1, c2=0.1, max_iter=100):
        self.model_dir = model_dir
        self.params = {"c1": c1, "c2": c2, "max_iterations": max_iter, "feature.possible_transitions": True}
        self.stage1 = os.path.join(model_dir, "crf_stage1.crfsuite")
        self.stage2 = os.path.join(model_dir, "crf_stage2.crfsuite")
        self._t1 = self._t2 = None

    def _train(self, X, y, path):
        tr = pycrfsuite.Trainer(verbose=False)
        for xs, ys in zip(X, y):
            tr.append(xs, ys)
        tr.set_params(self.params)
        tr.train(path)

    def _open(self, path):
        t = pycrfsuite.Tagger()
        t.open(path)
        return t

    def fit(self, sentences):
        os.makedirs(self.model_dir, exist_ok=True)
        words = [[w for w, _ in s] for s in sentences]
        tags = [[t for _, t in s] for s in sentences]
        self._train([sentence_features(w) for w in words], tags, self.stage1)
        self._t1 = self._open(self.stage1)
        # stage 2 features use stage 1's own predictions on the training data
        stacked = [sentence_features(w, self._t1.tag(sentence_features(w))) for w in words]
        self._train(stacked, tags, self.stage2)
        self._t2 = self._open(self.stage2)
        return self

    def load(self):
        self._t1, self._t2 = self._open(self.stage1), self._open(self.stage2)
        return self

    def tag(self, words):
        first = self._t1.tag(sentence_features(words))
        return list(zip(words, self._t2.tag(sentence_features(words, first))))

    def tag_stage1(self, words):
        return list(zip(words, self._t1.tag(sentence_features(words))))
