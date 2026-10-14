"""The 18 type labels and decoding. Numpy only, so the deployed app can import it."""

import numpy as np

# types.csv ids 1-18, in id order; index i is output unit i.
TYPES = ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
         "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark", "fairy"]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}


def decode(probs, threshold):
    """(n, 18) confidences -> (n, 18) 0/1 predictions. Always predict the top type; add
    the runner-up only if it clears `threshold`. Never zero or three types."""
    probs = np.asarray(probs)
    order = np.argsort(-probs, axis=1)
    pred = np.zeros_like(probs, dtype=np.int8)
    rows = np.arange(len(probs))
    pred[rows, order[:, 0]] = 1
    second = probs[rows, order[:, 1]] >= threshold
    pred[rows[second], order[second, 1]] = 1
    return pred
