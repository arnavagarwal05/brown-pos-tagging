"""Brown corpus with the 12-tag universal tagset, and k-fold splitting."""

import nltk
from sklearn.model_selection import KFold

UNIVERSAL_TAGS = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON", "PRT", "VERB", ".", "X"]


def ensure_nltk():
    for pkg, path in (("brown", "corpora/brown"), ("universal_tagset", "taggers/universal_tagset")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def load_brown():
    """List of sentences, each a list of (word, tag) with the universal tagset."""
    ensure_nltk()
    return [list(s) for s in nltk.corpus.brown.tagged_sents(tagset="universal")]


def kfold(sentences, k=5, seed=42):
    """Yields (train_sentences, test_sentences) per fold. Split by sentence, never by token."""
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(sentences):
        yield [sentences[i] for i in tr], [sentences[i] for i in te]
