"""Gradio demo: type a sentence, see HMM and CRF tags side by side."""

import gradio as gr

from postag.crf import StackedCRF
from postag.data import UNIVERSAL_TAGS, load_brown
from postag.hmm import HMMTagger

print("training HMM on Brown (a few seconds)...")
hmm = HMMTagger(UNIVERSAL_TAGS).fit(load_brown())
try:
    crf = StackedCRF("models").load()
except Exception:
    crf = None
    print("no models/crf_*.crfsuite found; run scripts/eval_crf.py first to enable the CRF column")


def tag(sentence):
    words = sentence.split()
    rows = [(w, t, "") for w, t in hmm.tag(words)]
    if crf:
        rows = [(w, t, c) for (w, t, _), (_, c) in zip(rows, crf.tag(words))]
    return rows


gr.Interface(
    fn=tag,
    inputs=gr.Textbox(label="Sentence", value="The quick brown fox jumps over the lazy dog ."),
    outputs=gr.Dataframe(headers=["word", "HMM", "CRF"], label="Tags"),
    title="Part-of-speech tagger",
    description="Brown corpus, universal tagset. HMM with Viterbi decoding; two-stage CRF.",
).launch()
