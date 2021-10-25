
import numpy as np


def choose_examples(x, examples_to_use, y=None):
    """Chooses exampels from x and y"""
    if isinstance(examples_to_use, int):
        # randomly chose x values from test_x
        x, index = choose_n_imp_exs(x, examples_to_use, y)
    elif isinstance(examples_to_use, float):
        assert examples_to_use < 1.0
        # randomly choose x fraction from test_x
        x, index = choose_n_imp_exs(x, int(examples_to_use * len(x)), y)

    elif hasattr(examples_to_use, '__len__'):
        index = np.array(examples_to_use)
        x = x[index]
    else:
        raise ValueError(f"unrecognized value of examples_to_use: {examples_to_use}")

    return x, index


def choose_n_imp_exs(x:np.ndarray, n:int, y=None):
    """Chooses the n important examples from x and y"""

    n = min(len(x), n)

    st = n // 2
    en = n - st

    if y is None:
        idx = np.random.randint(0, len(x), n)
    else:
        st = np.argsort(y, axis=0)[0:st].reshape(-1,)
        en = np.argsort(y, axis=0)[-en:].reshape(-1,)
        idx = np.hstack([st, en])

    x = x[idx]

    return x, idx