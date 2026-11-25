"""Algorithm 1: invariants, agreement with the line-by-line reference, merge operator (Eq. 10)."""

import math

import pytest
import torch

from lofi.frequency import derotate, low_high_channels, rope_thetas
from lofi.reference import select_one_head
from lofi.selection import local_maxima, next_prev_active, pair_scores, protected_mask, select

BASE, DIM = 1e4, 32
LOW, HIGH = low_high_channels(BASE, DIM)


def make(n_heads, n, seed, smooth=0.0):
    """Random keys, values and importance. ``smooth`` adds correlation between neighbours."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(n_heads, n, DIM, generator=g)
    if smooth > 0:
        for j in range(1, n):
            keys[:, j] = smooth * keys[:, j - 1] + (1 - smooth) * keys[:, j]
    values = torch.randn(n_heads, n, DIM, generator=g)
    importance = torch.rand(n_heads, n, generator=g)
    return keys, values, importance


CASES = [
    # heads, |D|, n_keep, protect, seed, smooth
    (4, 200, 40, 0.1, 0, 0.0),
    (2, 517, 103, 0.1, 1, 0.5),
    (3, 64, 60, 0.1, 2, 0.0),       # one pass, a few removals
    (3, 64, 3, 0.1, 3, 0.8),        # deep compression, many passes
    (2, 301, 30, 0.5, 4, 0.3),
    (1, 2, 1, 0.1, 5, 0.0),
    (2, 150, 149, 0.1, 6, 0.0),
    (2, 400, 80, 0.0, 7, 0.6),
]


@pytest.mark.parametrize("n_heads,n,n_keep,protect,seed,smooth", CASES)
def test_invariants(n_heads, n, n_keep, protect, seed, smooth):
    keys, values, importance = make(n_heads, n, seed, smooth)
    out = select(keys, values, importance, n_keep, LOW, HIGH, protect=protect)
    assert out.active.sum(dim=1).tolist() == [n_keep] * n_heads           # exactly B - n_s - W per head
    protected = protected_mask(importance, n_keep, protect)
    assert protected.sum(dim=1).tolist() == [math.ceil(protect * n_keep) if protect > 0 else 0] * n_heads
    assert bool((out.active | ~protected).all())                          # protected tokens are never removed
    assert out.fallback_removed.sum() == 0                                # the pair rule alone reaches the budget
    assert (out.pair_removed == n - n_keep).all()
    assert torch.equal(out.keys, keys) and torch.equal(out.values, values)   # eviction leaves the cache entries intact
    assert (out.passes >= math.ceil(math.log2(n / n_keep)) if n > n_keep else out.passes == 0).all()


@pytest.mark.parametrize("n_heads,n,n_keep,protect,seed,smooth", CASES)
def test_matches_reference(n_heads, n, n_keep, protect, seed, smooth):
    keys, values, importance = make(n_heads, n, seed, smooth)
    out = select(keys, values, importance, n_keep, LOW, HIGH, protect=protect)
    for h in range(n_heads):
        kept, passes = select_one_head(keys[h], importance[h].tolist(), n_keep, LOW, protect=protect)
        assert out.active[h].nonzero().squeeze(-1).tolist() == kept
        assert int(out.passes[h]) == passes


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_with_ties(seed):
    """Tied importances and tied scores follow the documented rules in both implementations."""
    g = torch.Generator().manual_seed(seed)
    n = 120
    keys = torch.randn(2, n, DIM, generator=g)
    keys[:, 30:50] = keys[:, 30:31]                      # a run of identical keys: a plateau of equal scores
    importance = torch.randint(0, 4, (2, n), generator=g).float()     # many equal importances
    out = select(keys, keys.clone(), importance, 24, LOW, HIGH)
    for h in range(2):
        kept, passes = select_one_head(keys[h], importance[h].tolist(), 24, LOW)
        assert out.active[h].nonzero().squeeze(-1).tolist() == kept
        assert int(out.passes[h]) == passes


def test_no_importance_matches_reference():
    keys, values, _ = make(2, 180, 11, 0.4)
    out = select(keys, values, None, 36, LOW, HIGH)
    for h in range(2):
        kept, _ = select_one_head(keys[h], None, 36, LOW)
        assert out.active[h].nonzero().squeeze(-1).tolist() == kept


def test_heads_are_independent():
    """Running all heads together gives each head the result it gets alone."""
    keys, values, importance = make(4, 256, 21, 0.5)
    joint = select(keys, values, importance, 50, LOW, HIGH)
    for h in range(4):
        alone = select(keys[h : h + 1], values[h : h + 1], importance[h : h + 1], 50, LOW, HIGH)
        assert torch.equal(joint.active[h], alone.active[0])


def test_candidates_are_local_maxima_and_disjoint():
    keys, _, _ = make(3, 300, 31, 0.5)
    active = torch.ones(3, 300, dtype=torch.bool)
    active[:, 5::7] = False                              # a non-trivial active set
    nxt, prv = next_prev_active(active)
    valid = active & (nxt < 300)
    feats = torch.nn.functional.normalize(keys[..., LOW], dim=-1)
    scores = pair_scores(feats, nxt, valid)
    cand = local_maxima(scores, nxt, prv, valid)
    for h in range(3):
        order = active[h].nonzero().squeeze(-1).tolist()
        s = [float(scores[h, j]) for j in order[:-1]]
        expected = [
            order[i] for i in range(len(s))
            if s[i] > (s[i - 1] if i > 0 else -math.inf) and s[i] >= (s[i + 1] if i + 1 < len(s) else -math.inf)
        ]
        assert cand[h].nonzero().squeeze(-1).tolist() == expected
        used = []
        for j in expected:
            used += [j, int(nxt[h, j])]
        assert len(used) == len(set(used))               # no two candidates share a token


def test_one_pass_removes_at_most_half():
    keys, values, importance = make(2, 401, 41, 0.5)
    out = select(keys, values, importance, 10, LOW, HIGH, max_passes=1)
    assert (out.pair_removed <= 401 // 2).all()
    assert (out.passes == 1).all()
    # the rest goes through the fallback of line 13: lowest importance among the unprotected tokens
    assert (out.fallback_removed == 391 - out.pair_removed).all()
    assert out.active.sum(dim=1).tolist() == [10, 10]
    protected = protected_mask(importance, 10, 0.1)
    assert bool((out.active | ~protected).all())


def test_fallback_evicts_lowest_importance():
    keys, values, importance = make(1, 50, 51)
    out = select(keys, values, importance, 20, LOW, HIGH, max_passes=0)      # fallback only
    expected = importance[0].topk(20).indices.sort().values
    assert out.active[0].nonzero().squeeze(-1).tolist() == expected.tolist()


def test_score_is_eq_5():
    keys, _, _ = make(1, 6, 61)
    active = torch.ones(1, 6, dtype=torch.bool)
    nxt, _ = next_prev_active(active)
    feats = torch.nn.functional.normalize(keys[..., LOW], dim=-1)
    scores = pair_scores(feats, nxt, active & (nxt < 6))
    for j in range(5):
        a, b = keys[0, j, LOW].double(), keys[0, j + 1, LOW].double()
        assert abs(float(scores[0, j]) - float(a @ b / (a.norm() * b.norm()))) < 1e-6
    assert scores[0, 5] == -math.inf


# ------------------------------------------------------------------ merge operator (LoFi-M)


def test_merge_single_pair_is_eq_10():
    keys, values, _ = make(1, 2, 71)
    importance = torch.tensor([[0.2, 0.9]])              # the right token is the target
    out = select(keys, values, importance, 1, LOW, HIGH, protect=0.0, operator="merge")
    assert out.active.tolist() == [[False, True]]
    kt, ku, vt, vu = keys[0, 1].double(), keys[0, 0].double(), values[0, 1].double(), values[0, 0].double()
    s = float(kt[LOW] @ ku[LOW] / (kt[LOW].norm() * ku[LOW].norm()))
    alpha = 0.2 + 0.3 * min(max(s, 0.0), 1.0)
    alpha_v = 0.15 + 0.15 * min(max(s, 0.0), 1.0)
    high = (1 - alpha) * kt[HIGH] + alpha * ku[HIGH]
    high = high * kt[HIGH].norm() / high.norm()
    assert torch.equal(out.keys[0, 1, LOW], keys[0, 1, LOW])
    assert torch.allclose(out.keys[0, 1, HIGH].double(), high, atol=1e-5)
    assert torch.allclose(out.values[0, 1].double(), (1 - alpha_v) * vt + alpha_v * vu, atol=1e-5)
    assert out.counts.tolist() == [[1.0, 2.0]]


@pytest.mark.parametrize("n_heads,n,n_keep,protect,seed,smooth", CASES)
def test_merge_properties(n_heads, n, n_keep, protect, seed, smooth):
    keys, values, importance = make(n_heads, n, seed, smooth)
    evict = select(keys, values, importance, n_keep, LOW, HIGH, protect=protect)
    merge = select(keys, values, importance, n_keep, LOW, HIGH, protect=protect, operator="merge")
    # on L the merge coincides with eviction, so both operators keep the same positions
    assert torch.equal(merge.active, evict.active)
    # low-frequency channels are untouched, bit for bit
    assert torch.equal(merge.keys[..., LOW], keys[..., LOW])
    # the high-frequency part of every key keeps its norm
    assert torch.allclose(merge.keys[..., HIGH].norm(dim=-1), keys[..., HIGH].norm(dim=-1), rtol=1e-4, atol=1e-5)
    # every original token is counted once among the kept positions
    assert torch.allclose((merge.counts * merge.active).sum(dim=1), torch.full((n_heads,), float(n)))
    # the inputs are not modified
    k2, v2, _ = make(n_heads, n, seed, smooth)
    assert torch.equal(keys, k2) and torch.equal(values, v2)
    if n > n_keep:
        assert not torch.equal(merge.values, values)


# ------------------------------------------------------------------ controls of the selection-rule comparison


def test_controls_run_and_meet_the_budget():
    keys, values, importance = make(2, 300, 81, 0.5)
    inv_freq = rope_thetas(BASE, DIM)
    derotated = derotate(keys, torch.arange(32, 332), inv_freq)
    g = torch.Generator().manual_seed(0)
    settings = [
        dict(criterion="full"),
        dict(criterion="derotated", derotated_keys=derotated),
        dict(criterion="random", generator=g),
        dict(criterion="low", first_criterion="full"),
        dict(criterion="full", first_criterion="low"),
        dict(criterion="full", operator="merge"),
    ]
    kept = []
    for kwargs in settings:
        out = select(keys, values, importance, 60, LOW, HIGH, **kwargs)
        assert out.active.sum(dim=1).tolist() == [60, 60]
        kept.append(out.active)
    base = select(keys, values, importance, 60, LOW, HIGH).active
    assert all(not torch.equal(k, base) for k in kept[:5])       # each control changes the kept set
    out = select(keys, values, None, 60, LOW, HIGH)              # no importance
    assert out.active.sum(dim=1).tolist() == [60, 60]


def test_random_control_is_reproducible():
    keys, values, importance = make(2, 200, 91)
    a = select(keys, values, importance, 40, LOW, HIGH, criterion="random", generator=torch.Generator().manual_seed(3))
    b = select(keys, values, importance, 40, LOW, HIGH, criterion="random", generator=torch.Generator().manual_seed(3))
    c = select(keys, values, importance, 40, LOW, HIGH, criterion="random", generator=torch.Generator().manual_seed(4))
    assert torch.equal(a.active, b.active) and not torch.equal(a.active, c.active)


def test_first_pass_criterion_only_affects_first_pass():
    keys, values, importance = make(2, 200, 101, 0.5)
    low = select(keys, values, importance, 150, LOW, HIGH)                               # one pass is enough
    mixed = select(keys, values, importance, 150, LOW, HIGH, criterion="full", first_criterion="low")
    assert (low.passes == 1).all() and torch.equal(low.active, mixed.active)


def test_argument_checks():
    keys, values, importance = make(1, 20, 111)
    with pytest.raises(ValueError):
        select(keys, values, importance, 0, LOW, HIGH)
    with pytest.raises(ValueError):
        select(keys, values, importance, 21, LOW, HIGH)
    with pytest.raises(ValueError):
        select(keys, values, importance, 5, LOW, HIGH, criterion="derotated")
    with pytest.raises(ValueError):
        select(keys, values, importance, 5, LOW, HIGH, criterion="cosine")


# ------------------------------------------------------------------ pass cap


@pytest.mark.parametrize("cap", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize("n_heads,n,n_keep,protect,seed,smooth", CASES[:5])
def test_pass_cap_matches_reference(cap, n_heads, n, n_keep, protect, seed, smooth):
    """With a cap, the passes that run are unchanged and the fallback removes exactly the rest."""
    keys, values, importance = make(n_heads, n, seed, smooth)
    out = select(keys, values, importance, n_keep, LOW, HIGH, protect=protect, max_passes=cap)
    free = select(keys, values, importance, n_keep, LOW, HIGH, protect=protect)
    protected = protected_mask(importance, n_keep, protect)
    assert out.active.sum(dim=1).tolist() == [n_keep] * n_heads           # the budget is met either way
    assert bool((out.active | ~protected).all())                          # the fallback spares protected tokens
    assert (out.passes <= cap).all()
    assert (out.pair_removed + out.fallback_removed == n - n_keep).all()
    for h in range(n_heads):
        kept, passes = select_one_head(keys[h], importance[h].tolist(), n_keep, LOW, protect=protect, max_passes=cap)
        assert out.active[h].nonzero().squeeze(-1).tolist() == kept
        assert int(out.passes[h]) == passes
        if int(free.passes[h]) <= cap:                                    # a cap that does not bind changes nothing
            assert torch.equal(out.active[h], free.active[h]) and int(out.fallback_removed[h]) == 0
        else:
            assert int(out.fallback_removed[h]) > 0
