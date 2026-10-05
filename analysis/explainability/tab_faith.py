"""Deletion / insertion faithfulness for tabular explainers (numpy only).

For one subject x (K model inputs), a background B (50 train rows, rng 0) and an order of the K inputs:
  deletion step k: the first k inputs of the order are replaced by each background row; output = mean over B
                   of P(target class);
  insertion step k: start from the background rows, insert the subject's first k inputs.
Area = trapezoid over k = 0..K divided by K. The statistic per subject is area(attribution order) - mean area
(50 random orders, seed SeedSequence([RID, code])). Deletion should be lower (negative difference),
insertion higher (positive).
"""
import numpy as np

N_RAND = 50


def attribution_order(phi_target):
    """Descending attribution toward the explained class; ties by input index."""
    return np.lexsort((np.arange(len(phi_target)), -np.asarray(phi_target)))


def random_orders(rid, code, K):
    rng = np.random.default_rng(np.random.SeedSequence([int(rid), int(code)]))
    return np.stack([rng.permutation(K) for _ in range(N_RAND)])


def curves(x, B, orders, predict, target):
    """orders (n_orders, K) -> deletion, insertion curves (n_orders, K+1) of mean P(target)."""
    n_o, K = orders.shape
    nb = len(B)
    keep = np.zeros((n_o, K + 1, K), bool)                       # keep[o, k, j]: input j replaced/inserted at step k
    for o in range(n_o):
        for k in range(1, K + 1):
            keep[o, k, orders[o, :k]] = True
    Xd = np.where(keep[:, :, None, :], B[None, None], x[None, None, None])      # (n_o, K+1, nb, K)
    Xi = np.where(keep[:, :, None, :], x[None, None, None], B[None, None])
    P = predict(np.concatenate([Xd.reshape(-1, K), Xi.reshape(-1, K)]))[:, target]
    P = P.reshape(2, n_o, K + 1, nb).mean(-1)
    return P[0], P[1]


def area(c):
    K = c.shape[-1] - 1
    return np.trapezoid(c, dx=1.0, axis=-1) / K


def subject_faithfulness(x, B, phi_target, rid, code, predict, target):
    K = len(x)
    orders = np.vstack([attribution_order(phi_target)[None], random_orders(rid, code, K)])
    d, i = curves(x, B, orders, predict, target)
    ad, ai = area(d), area(i)
    return {"del_diff": float(ad[0] - ad[1:].mean()), "ins_diff": float(ai[0] - ai[1:].mean()),
            "del_attr": d[0], "del_rand": d[1:].mean(0), "ins_attr": i[0], "ins_rand": i[1:].mean(0)}


def verdict(del_ci_hi, ins_ci_lo):
    dp, ip = bool(del_ci_hi < 0), bool(ins_ci_lo > 0)
    return dp, ip, ("PASS" if dp and ip else ("WEAK" if dp or ip else "FAIL"))
