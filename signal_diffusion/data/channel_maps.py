"""EEG channel orderings for each supported dataset."""
from __future__ import annotations

from typing import Sequence, Tuple

ChannelMap = Sequence[Tuple[str, int]]

math_channels: ChannelMap = [
    ("Fp1", 0),
    ("Fp2", 1),
    ("F3", 2),
    ("F4", 3),
    ("Fz", 16),
    ("F7", 4),
    ("F8", 5),
    ("C3", 8),
    ("C4", 9),
    ("Cz", 17),
    ("P3", 12),
    ("P4", 13),
    ("Pz", 18),
    ("O1", 14),
    ("O2", 15),
    ("T3", 6),
    ("T4", 7),
    ("T5", 10),
    ("T6", 11),
    ("A2-A1", 19),
]

parkinsons_channels: ChannelMap = [
    ("Fp1", 0),
    ("Fp2", 30),
    ("F3", 2),
    ("F4", 28),
    ("Fz", 1),
    ("F7", 3),
    ("F8", 29),
    ("C3", 7),
    ("C4", 23),
    ("Cz", 22),
    ("P3", 12),
    ("P4", 17),
    ("Pz", 51),
    ("O1", 14),
    ("O2", 16),
    ("T7", 8),  # T3 equiv
    ("T8", 24),  # T4 equiv
    ("TP7", 40),  # T5 equiv
    ("TP8", 53),  # T6 equiv
    ("TP10", 19),  # A0 equiv
]

seed_channels: ChannelMap = [
    ("FP1", 0),
    ("FP2", 2),
    ("F3", 7),
    ("F4", 11),
    ("FZ", 9),
    ("F7", 5),
    ("F8", 13),
    ("C3", 25),
    ("C4", 29),
    ("CZ", 27),
    ("P3", 45),
    ("P4", 49),
    ("PZ", 47),
    ("O1", 60),
    ("O2", 62),
    ("T7", 23),  # T3 equiv
    ("T8", 31),  # T4 equiv
    ("TP7", 33),  # T5 equiv
    ("TP8", 41),  # T6 equiv
    ("P8", 51),  # A0 equiv
]

longitudinal_channels: ChannelMap = [
    ("Fp1", 0),
    ("Fp2", 1),
    ("F3", 3),
    ("F4", 5),
    ("Fz", 4),
    ("F7", 2),
    ("F8", 6),
    ("C3", 12),
    ("C4", 14),
    ("Cz", 13),
    ("P3", 23),
    ("P4", 25),
    ("Pz", 24),
    ("O1", 28),
    ("O2", 30),
    ("T7", 11),
    ("T8", 15),
    ("TP7", 50),
    ("TP8", 54),
    ("TP10", 21),
]

__all__ = [
    "ChannelMap",
    "math_channels",
    "parkinsons_channels",
    "seed_channels",
    "longitudinal_channels",
]
