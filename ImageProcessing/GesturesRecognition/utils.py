from dataclasses import dataclass


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
GESTURES = {
    0: 'one',
    1: 'fist',
    2: 'stop',
    3: 'peace',
    4: 'four',
    5: 'three2',
    6: 'rock'
}


@dataclass
class ImageShape:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y
        