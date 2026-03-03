# lib/adaptive_weights.py

class AdaptiveWeights:
    def __init__(self, weights, lr):
        self.weights = weights.copy()
        self.lr      = lr

    def update(self, signal_scores, trade_result):
        for key in self.weights:
            contribution       = signal_scores.get(key, 0)
            self.weights[key] += self.lr * trade_result * contribution
        total = sum(abs(v) for v in self.weights.values())
        if total == 0:
            return
        for k in self.weights:
            self.weights[k] /= total
