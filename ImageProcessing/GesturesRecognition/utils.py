from dataclasses import dataclass
from enum import Enum


LANDMARKS_LINKS = {
    0: [1, 5, 17],
    1: [2],
    2: [3],
    3: [4],
    5: [6, 9],
    6: [7],
    7: [8],
    9: [10, 13],
    10: [11],
    11: [12],
    13: [14, 17],
    14: [15],
    15: [16],
    17: [18],
    18: [19],
    19: [20]
}


class Gestures(Enum):
    ONE = 0
    FIST = 1
    STOP = 2
    PEACE = 3
    FOUR = 4
    THREE2 = 5
    ROCK = 6


@dataclass
class ImageShape:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y
        