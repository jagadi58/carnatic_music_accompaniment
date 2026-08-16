#!/usr/bin/env python3
"""Diagnostic: score a cached .npz, show clean stream + mined phrases."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core.raga_identifier import RagaIdentifier

try:
    from src.core.raga_identifier import TOKENS
except ImportError:
    TOKENS = ["S","R1","R2","G2","G3","M1","M2","P","D1","D2","N2","N3"]

npz = Path(input("Path to .npz cache: ").strip().strip('"'))
data = np.load(npz, allow_pickle=True)
sruti = float(data["sruti"])

ident = RagaIdentifier(str(ROOT / "config" / "databases/ /ragas.json".replace("/ /", "/")))
ident.set_sruti(sruti)
results = ident.identify_raga(data["freqs"], data["confs"])

mask = data["confs"] >= 0.4
vf = data["freqs"][mask]; vf = vf[vf > 0]
semis = 12.0 * np.log2(vf / sruti)

def clean_at(shift):
    raw = [TOKENS[int(np.round(x - shift)) % 12] for x in semis]
    held = ident._hold_filter(ident._collapse_gamakas(raw), 12)
    if hasattr(ident, "_absorb_overshoot"):
        held = ident._absorb_overshoot(held)
    return ident._compress_swara_stream(held, 10), held

def raga_score(raga, stream, held):
    counts = {t: held.count(t) for t in set(held) if t != "-"}
    total = sum(counts.values()) or 1
    det = set(stream)
    if hasattr(ident, "_score_raga_v6"):
        return ident._score_raga_v6(raga, stream, det, counts, total, 0.5)
    if hasattr(ident, "_score_raga_v5"):
        return ident._score_raga_v5(raga, stream, det, counts, total, 0.5)
    return 0.0

best_s, best_top = 0, -1.0
for s in range(12):
    stream, held = clean_at(s)
    if not stream:
        continue
    top = max(raga_score(r, stream, held) for r in ident.db.get_all("ragas"))
    if top > best_top:
        best_top, best_s = top, s

stream, _ = clean_at(best_s)
print("reference shift :", best_s)
print("clean stream    :", " ".join(stream))
print("\nTop 5:")
for r in results[:5]:
    print(f"  {r['name']:20s} {r['score']:.3f}")

c = Counter()
for n in (4, 5, 6):
    c.update(tuple(stream[i:i+n]) for i in range(len(stream)-n+1))
print("\nRepeated phrases (candidate prayogas):")
shown = 0
for g, k in c.most_common(40):
    if k >= 2 and shown < 12:
        print("  ", " ".join(g), " x", k)
        shown += 1