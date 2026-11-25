import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
shape = pytest.importorskip("apertus_tree.shape")
agg = pytest.importorskip("apertus_tree.shape_aggregate")


def test_walk_counts_leading_matches_and_jumps():
    nan = np.nan
    match = np.array(
        [
            [1, 1, 0, 1, 1, 1, 1],  # step 0: 2 accepted -> jump 3
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, nan],  # step at 3: k=3 caps at 3 -> jump 4
            [0] * 7,
            [0] * 7,
            [1, nan, nan, nan, nan, nan, nan],
        ],
        float,
    )
    assert agg.walk(match, 3) == [(0, 2), (3, 3)]
    assert agg.walk(match, 7) == [(0, 2), (3, 6)]
    assert agg.walk(match, 1) == [(0, 1), (2, 0), (3, 1), (5, 0), (6, 1)]


def test_dist_metrics_identities():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    lp = torch.log_softmax(logits, -1)
    truth = lp.argmax(-1)
    m = shape.dist_metrics(lp, lp, truth, truth)
    assert torch.allclose(m["tv"], torch.zeros(4), atol=1e-6)
    assert torch.allclose(m["kl"], torch.zeros(4), atol=1e-6)
    assert m["match"].eq(1).all() and m["agree"].eq(1).all()
    uniform = torch.full((1, 50), -math.log(50))
    point = torch.full((1, 50), -1e4)
    point[0, 0] = 0.0
    point = torch.log_softmax(point, -1)
    m = shape.dist_metrics(point, uniform, torch.tensor([0]), torch.tensor([0]))
    assert m["ent"].item() == pytest.approx(math.log(50), rel=1e-5)
    assert m["tv"].item() == pytest.approx(1 - 1 / 50, rel=1e-5)
    assert m["kl"].item() == pytest.approx(math.log(50), rel=1e-4)
