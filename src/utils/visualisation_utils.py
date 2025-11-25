import numpy as np
import matplotlib.pyplot as plt


def plot_rhythm_from_sequence(seq, title="Rhythm (step & duration)"):
    """
    seq: (T,3) denorm or norm (we just plot values)
    рисует шаги и длительности по time index
    """
    seq = np.array(seq)
    pitches = seq[:, 0]
    steps = seq[:, 1]
    durs = seq[:, 2]
    T = len(steps)
    times = np.cumsum(np.concatenate([[0.0], steps[:-1]]))  # onset times
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax[0].plot(times, steps, marker='o', linestyle='-')
    ax[0].set_ylabel("step (delta to prev)")
    ax[0].grid(True)

    ax[1].plot(times, durs, marker='o', linestyle='-')
    ax[1].set_ylabel("duration (quarter lengths)")
    ax[1].set_xlabel("onset time (quarter lengths)")
    ax[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
