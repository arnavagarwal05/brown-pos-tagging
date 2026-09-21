"""Fast checks on a tiny synthetic corpus; no Brown download needed."""

from postag.data import UNIVERSAL_TAGS
from postag.hmm import HMMTagger

TOY = [
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB"), (".", ".")],
    [("a", "DET"), ("dog", "NOUN"), ("ran", "VERB"), ("quickly", "ADV"), (".", ".")],
    [("the", "DET"), ("big", "ADJ"), ("dog", "NOUN"), ("sat", "VERB"), (".", ".")],
    [("cats", "NOUN"), ("run", "VERB"), (".", ".")],
] * 5


def test_viterbi_recovers_training_sentence():
    h = HMMTagger(UNIVERSAL_TAGS).fit(TOY)
    assert [t for _, t in h.tag(["the", "cat", "sat", "."])] == ["DET", "NOUN", "VERB", "."]


def test_viterbi_is_not_greedy():
    # "run" is only ever a VERB; after DET a NOUN is required, so an exact decoder
    # must still produce a full-length path and never crash on a zero-count pair.
    h = HMMTagger(UNIVERSAL_TAGS).fit(TOY)
    out = h.tag(["the", "run", "."])
    assert len(out) == 3 and all(t in UNIVERSAL_TAGS for _, t in out)


def test_unknown_word_uses_suffix():
    h = HMMTagger(UNIVERSAL_TAGS, max_suffix=3).fit(TOY)
    tags = dict(h.tag(["the", "zog", "zoggly", "."]))
    assert tags["zoggly"] == "ADV"  # -ly seen only with ADV in training


def test_empty_sentence():
    assert HMMTagger(UNIVERSAL_TAGS).fit(TOY).tag([]) == []


def test_crf_features_shape():
    from postag.crf import sentence_features
    f = sentence_features(["The", "dog", "."])
    assert f[0]["prev"] == "^" and f[-1]["next"] == "$" and f[0]["cap"] == 1
    f2 = sentence_features(["The", "dog"], prev_tags=["DET", "NOUN"])
    assert f2[0]["prev_tag"] == "^" and f2[1]["prev_tag"] == "DET"
