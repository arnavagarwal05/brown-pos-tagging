"""A bigram hidden Markov model tagger with exact Viterbi decoding.

Transition and emission tables are estimated from training data only. Words not
seen in training fall back to a suffix model: P(word | tag) is approximated by
P(suffix | tag) for the longest suffix (up to 3 characters) seen with that tag,
with a capitalisation flag, which is what actually lets the tagger guess that an
unseen "-ing" word is a VERB or an unseen "-ly" word is an ADV.
"""

import math
from collections import Counter, defaultdict

START = "<s>"


class HMMTagger:
    def __init__(self, tags, max_suffix=3, alpha=0.01):
        self.tags = list(tags)
        self.max_suffix = max_suffix
        self.alpha = alpha
        self.log_trans = {}
        self.log_emit = defaultdict(dict)
        self.log_suffix = {}
        self.log_unk = {}
        self.vocab = set()

    @staticmethod
    def _shape(word):
        return "Cap" if word[:1].isupper() else "low"

    def _suffixes(self, word):
        w = word.lower()
        return [f"{self._shape(word)}|{w[-n:]}" for n in range(min(self.max_suffix, len(w)), 0, -1)] + [f"{self._shape(word)}|"]

    def fit(self, sentences):
        trans, emit, tag_count = Counter(), defaultdict(Counter), Counter()
        suffix = defaultdict(Counter)
        word_freq = Counter(w for s in sentences for w, _ in s)
        for s in sentences:
            prev = START
            for w, t in s:
                trans[(prev, t)] += 1
                emit[t][w] += 1
                tag_count[t] += 1
                # suffix statistics come from rare words, which behave like unknown words
                if word_freq[w] <= 5:
                    for sfx in self._suffixes(w):
                        suffix[sfx][t] += 1
                prev = t
        self.vocab = set(word_freq)
        n_tags = len(self.tags)
        prev_count = Counter()
        for (p, _), c in trans.items():
            prev_count[p] += c
        for p in [START] + self.tags:
            for t in self.tags:
                self.log_trans[(p, t)] = math.log((trans[(p, t)] + self.alpha) / (prev_count[p] + self.alpha * n_tags))
        for t in self.tags:
            denom = tag_count[t] + self.alpha * len(self.vocab)
            for w, c in emit[t].items():
                self.log_emit[t][w] = math.log((c + self.alpha) / denom)
            self.log_unk[t] = math.log(self.alpha / denom)
        # P(tag | suffix), stored per suffix, later combined with P(tag) to get P(suffix | tag)
        self.log_tag_prior = {t: math.log((tag_count[t] + 1) / (sum(tag_count.values()) + n_tags)) for t in self.tags}
        for sfx, cnt in suffix.items():
            total = sum(cnt.values())
            self.log_suffix[sfx] = {t: math.log((cnt[t] + 0.5) / (total + 0.5 * n_tags)) for t in self.tags}
        return self

    def _emission(self, word, tag):
        if word in self.vocab:
            return self.log_emit[tag].get(word, self.log_unk[tag])
        for sfx in self._suffixes(word):
            if sfx in self.log_suffix:
                # Bayes: log P(sfx | tag) = log P(tag | sfx) - log P(tag) + const
                return self.log_suffix[sfx][tag] - self.log_tag_prior[tag]
        return self.log_unk[tag]

    def tag(self, words):
        """Exact Viterbi: best tag sequence under the bigram model, in log space."""
        if not words:
            return []
        n, T = len(words), self.tags
        score = [{t: self.log_trans[(START, t)] + self._emission(words[0], t) for t in T}]
        back = [{}]
        for i in range(1, n):
            cur, bp = {}, {}
            em = {t: self._emission(words[i], t) for t in T}
            for t in T:
                best_prev = max(T, key=lambda p: score[-1][p] + self.log_trans[(p, t)])
                cur[t] = score[-1][best_prev] + self.log_trans[(best_prev, t)] + em[t]
                bp[t] = best_prev
            score.append(cur)
            back.append(bp)
        last = max(T, key=lambda t: score[-1][t])
        path = [last]
        for i in range(n - 1, 0, -1):
            path.append(back[i][path[-1]])
        return list(zip(words, reversed(path)))
