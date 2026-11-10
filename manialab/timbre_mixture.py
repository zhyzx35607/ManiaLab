"""Nonnegative local timbre evidence; research only, no exclusive class choice."""
import numpy as np
from scipy.optimize import nnls
from scipy.signal import stft


def _matrix(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError('Expected finite nonnegative feature rows')
    return values


def attack_power(samples, rate, times_ms):
    """Linear band power above the local background, retaining attack evolution."""
    samples = np.asarray(samples)
    times_ms = np.asarray(times_ms, dtype=float)
    if (samples.ndim != 1 or len(samples) < 1024 or rate < 8000
            or not np.isfinite(samples).all() or times_ms.ndim != 1
            or not np.isfinite(times_ms).all() or (times_ms < 0).any()
            or (times_ms >= len(samples) / rate * 1000).any()):
        raise ValueError('Invalid mono audio or candidate clock')
    frequency, clock, spectrum = stft(samples, rate, nperseg=1024, noverlap=896)
    power = np.abs(spectrum) ** 2
    edges = np.geomspace(35, rate / 2, 65)
    bands = np.asarray([power[(frequency >= a) & (frequency < b)].sum(axis=0)
                        for a, b in zip(edges, edges[1:])])
    rows = []
    for time in times_ms / 1000:
        vectors = []
        for low, high in ((-.045, -.005), (-.005, .025), (.025, .070), (.070, .140)):
            lo, hi = np.searchsorted(clock, [time + low, time + high])
            vectors.append(bands[:, lo:hi].mean(axis=1) if hi > lo else np.zeros(64))
        rows.append(np.maximum(0, np.asarray(vectors[1:]) - vectors[0]).ravel())
    return np.asarray(rows, dtype=float).reshape(len(times_ms), 192)


def prototypes(values, count):
    """Deterministic spherical clusters of annotated single-class examples."""
    values = _matrix(values)
    norms = np.linalg.norm(values, axis=1)
    values = values[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(values) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Nonzero examples and a positive template count are required')
    count = min(count, len(values))
    center = values.mean(axis=0)
    selected = [int(np.argmax(values @ center))]
    for _ in range(count - 1):
        distances = 1 - (values @ values[selected].T).max(axis=1)
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    atoms = values[selected].copy()
    for _ in range(100):
        assigned = np.argmax(values @ atoms.T, axis=1)
        updated = atoms.copy()
        for index in range(count):
            if np.any(assigned == index):
                row = values[assigned == index].mean(axis=0)
                updated[index] = row / np.linalg.norm(row)
        if np.max(np.abs(updated - atoms)) < 1e-10:
            return updated
        atoms = updated
    raise ValueError('Timbre template clustering did not converge')


def decompose(values, atoms, class_ids, class_count):
    """Allow simultaneous drum components and retain unexplained energy.

    Scores are normalized component coefficients, not calibrated probabilities.
    They are intentionally not made to sum to one or forced into one label.
    """
    values, atoms = _matrix(values), _matrix(atoms)
    class_ids = np.asarray(class_ids)
    if (not len(atoms) or values.shape[1] != atoms.shape[1]
            or class_ids.shape != (len(atoms),) or class_ids.dtype.kind not in 'iu'
            or class_count < 1 or (class_ids < 0).any() or (class_ids >= class_count).any()
            or (np.linalg.norm(atoms, axis=1) < 1e-12).any()):
        raise ValueError('Invalid timbre dictionary')
    scores = np.zeros((len(values), class_count))
    residual = np.zeros(len(values))
    coefficients = np.zeros((len(values), len(atoms)))
    for index, row in enumerate(values):
        norm = np.linalg.norm(row)
        if norm < 1e-12:
            continue
        weights, error = nnls(atoms.T, row / norm, maxiter=max(100, 10 * len(atoms)))
        coefficients[index] = weights
        residual[index] = error
        for category in range(class_count):
            scores[index, category] = weights[class_ids == category].sum()
    return scores, residual, coefficients
