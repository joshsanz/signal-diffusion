"""EEG channel orderings for each supported dataset."""
from __future__ import annotations

from typing import Sequence, Tuple

ChannelMap = Sequence[Tuple[str, int]]

math_channels: ChannelMap = [
    ("Fp1", 0),
    ("Fp2", 1),
    ("F3", 2),
    ("F4", 3),
    ("Fz", 4),
    ("F7", 5),
    ("F8", 6),
    ("C3", 7),
    ("C4", 8),
    ("Cz", 9),
    ("P3", 10),
    ("P4", 11),
    ("Pz", 12),
    ("O1", 13),
    ("O2", 14),
    ("T3", 15),
    ("T4", 16),
    ("T5", 17),
    ("T6", 18),
    ("A", 19),
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
    ("Fp1", 0),
    ("Fp2", 2),
    ("F3", 7),
    ("F4", 11),
    ("Fz", 9),
    ("F7", 5),
    ("F8", 13),
    ("C3", 25),
    ("C4", 29),
    ("Cz", 27),
    ("P3", 45),
    ("P4", 49),
    ("Pz", 47),
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

mit_channels: ChannelMap = [
    ("FP1-F7", 0),
    ("FP1-F3", 4),
    ("FP2-F4", 8),
    ("FP2-F8", 12),
    ("F3-C3", 5),
    ("FZ-CZ", 16),
    ("F7-T7", 1),
    ("C3-P3", 6),
    ("C4-P4", 10),
    ("CZ-PZ", 17),
    ("P3-O1", 7),
    ("P4-O2", 11),
    ("T7-FT9", 19),
    ("T7-P7", 2),
    ("T8-P8", 14),
    ("T8-P8", 22),
    ("P8-O2", 15),
    ("P7-O1", 3),
    ("F4-C4", 9),
    ("F8-T8", 13),
    ("P7-T7", 18),
    ("FT9-FT10", 20),
    ("FT10-T8", 21),
]

__all__ = [
    "ChannelMap",
    "math_channels",
    "parkinsons_channels",
    "seed_channels",
    "longitudinal_channels",
    "mit_channels",
]
