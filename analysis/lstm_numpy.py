"""
lstm_numpy.py — Project 15 PHASE 5. A compact, time-aware LSTM implemented from scratch in NumPy
(no TF/PyTorch installed; pure-NumPy keeps it reproducible — Tenet 8).

Single-layer LSTM, input = [value_t, dt_t] (dt = years to the NEXT observation, so the model is
Δt-aware over the irregular DHS wave spacing), scalar next-value output. Trained by full-batch BPTT
+ Adam over the pooled 16-region panel. `gradient_check()` validates the analytic BPTT against finite
differences (run once during development).

This is deliberately small: with 16 sequences of length 9 the honest expectation is that statistical
baselines are competitive or better (evidence bank S14/S15). The LSTM is the title method and a
benchmarked comparator, not assumed to win.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TimeAwareLSTM:
    def __init__(self, hidden=8, seed=0):
        self.H = hidden
        rng = np.random.default_rng(seed)
        D = 2  # [value, dt]
        s = 0.1
        # input->gate (D), hidden->gate (H), biases; order gates f,i,g,o
        self.p = {
            "Wf": rng.normal(0, s, (self.H, D)), "Uf": rng.normal(0, s, (self.H, self.H)), "bf": np.zeros(self.H),
            "Wi": rng.normal(0, s, (self.H, D)), "Ui": rng.normal(0, s, (self.H, self.H)), "bi": np.zeros(self.H),
            "Wg": rng.normal(0, s, (self.H, D)), "Ug": rng.normal(0, s, (self.H, self.H)), "bg": np.zeros(self.H),
            "Wo": rng.normal(0, s, (self.H, D)), "Uo": rng.normal(0, s, (self.H, self.H)), "bo": np.zeros(self.H),
            "Wy": rng.normal(0, s, (1, self.H)), "by": np.zeros(1),
        }
        self.p["bf"] += 1.0  # forget-gate bias = 1 (standard, helps gradient flow)

    def _seq_forward(self, X):
        """X: [T, 2]. Returns predictions yhat [T] and a cache for BPTT."""
        H = self.H
        T = X.shape[0]
        h, c = np.zeros(H), np.zeros(H)
        cache = []
        yhat = np.zeros(T)
        for t in range(T):
            x = X[t]
            f = _sigmoid(self.p["Wf"] @ x + self.p["Uf"] @ h + self.p["bf"])
            i = _sigmoid(self.p["Wi"] @ x + self.p["Ui"] @ h + self.p["bi"])
            g = np.tanh(self.p["Wg"] @ x + self.p["Ug"] @ h + self.p["bg"])
            o = _sigmoid(self.p["Wo"] @ x + self.p["Uo"] @ h + self.p["bo"])
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            y = self.p["Wy"] @ h_new + self.p["by"]
            cache.append((x, h, c, f, i, g, o, c_new, h_new))
            h, c = h_new, c_new
            yhat[t] = y[0]
        return yhat, cache

    def _seq_backward(self, cache, dy):
        """dy: [T] gradient of loss wrt each yhat[t]. Returns grad dict (summed over t)."""
        H = self.H
        g_ = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh_next, dc_next = np.zeros(H), np.zeros(H)
        for t in reversed(range(len(cache))):
            x, h_prev, c_prev, f, i, gg, o, c_new, h_new = cache[t]
            g_["Wy"] += np.outer([dy[t]], h_new)
            g_["by"] += dy[t]
            dh = self.p["Wy"].T[:, 0] * dy[t] + dh_next
            do = dh * np.tanh(c_new)
            dc = dh * o * (1 - np.tanh(c_new) ** 2) + dc_next
            df = dc * c_prev
            di = dc * gg
            dgg = dc * i
            dc_prev = dc * f
            # gate pre-activation grads
            af = df * f * (1 - f)
            ai = di * i * (1 - i)
            ag = dgg * (1 - gg ** 2)
            ao = do * o * (1 - o)
            for name, a in (("f", af), ("i", ai), ("g", ag), ("o", ao)):
                g_["W" + name] += np.outer(a, x)
                g_["U" + name] += np.outer(a, h_prev)
                g_["b" + name] += a
            dh_next = (self.p["Uf"].T @ af + self.p["Ui"].T @ ai +
                       self.p["Ug"].T @ ag + self.p["Uo"].T @ ao)
            dc_next = dc_prev
        return g_

    def loss_and_grad(self, seqs):
        """seqs: list of (X[T,2], target[T]). MSE over all predicted steps. Returns loss, grads."""
        total, n = 0.0, 0
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        for X, target in seqs:
            yhat, cache = self._seq_forward(X)
            err = yhat - target
            total += np.sum(err ** 2)
            n += len(target)
            self_g = self._seq_backward(cache, 2 * err)
            for k in grads:
                grads[k] += self_g[k]
        for k in grads:
            grads[k] /= n
        return total / n, grads

    def fit(self, seqs, epochs=600, lr=0.05, seed=0):
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            loss, g = self.loss_and_grad(seqs)
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mhat = m[k] / (1 - b1 ** t)
                vhat = v[k] / (1 - b2 ** t)
                self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        return loss

    def forecast(self, X_hist, last_value, future_dts):
        """Roll the LSTM state over the observed history, then forecast recursively.
        X_hist: [T,2] observed inputs; last_value: last observed (std) value;
        future_dts: list of std-dt to each future target. Returns list of std predictions."""
        _, cache = self._seq_forward(X_hist)
        _, h, c, *_ , c_new, h_new = cache[-1]
        h, c = h_new, c_new
        val = last_value
        out = []
        for dt in future_dts:
            x = np.array([val, dt])
            f = _sigmoid(self.p["Wf"] @ x + self.p["Uf"] @ h + self.p["bf"])
            i = _sigmoid(self.p["Wi"] @ x + self.p["Ui"] @ h + self.p["bi"])
            g = np.tanh(self.p["Wg"] @ x + self.p["Ug"] @ h + self.p["bg"])
            o = _sigmoid(self.p["Wo"] @ x + self.p["Uo"] @ h + self.p["bo"])
            c = f * c + i * g
            h = o * np.tanh(c)
            val = (self.p["Wy"] @ h + self.p["by"])[0]
            out.append(val)
        return out


def gradient_check(seed=1):
    """Finite-difference check of analytic BPTT on a tiny random sequence."""
    rng = np.random.default_rng(seed)
    net = TimeAwareLSTM(hidden=4, seed=seed)
    seqs = [(rng.normal(0, 1, (6, 2)), rng.normal(0, 1, 6))]
    _, grads = net.loss_and_grad(seqs)
    worst = 0.0
    for k in net.p:
        flat = net.p[k].ravel()
        for j in range(min(flat.size, 5)):
            orig = flat[j]
            eps = 1e-5
            flat[j] = orig + eps
            lp, _ = net.loss_and_grad(seqs)
            flat[j] = orig - eps
            lm, _ = net.loss_and_grad(seqs)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[k].ravel()[j]
            # floor the denominator so finite-difference noise on near-zero gradients
            # (|grad| ~ 1e-8) doesn't inflate the relative error
            denom = max(1e-3, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
    return worst


if __name__ == "__main__":
    w = gradient_check()
    print(f"gradient check max relative error: {w:.2e}  ->", "PASS" if w < 1e-4 else "FAIL")
