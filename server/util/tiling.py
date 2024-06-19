#
# tiling.py
# Bart Trzynadlowski
#
# Square tiling. Given N tiles, arranges them to be as square as possible.
#

from math import ceil, sqrt
from typing import Tuple


class SquareTiling:
    width: int
    height: int

    def __init__(self, num_tiles: int):
        self.width = ceil(sqrt(num_tiles))
        self.height = num_tiles // self.width

    def index_to_coordinate(self, idx: int) -> Tuple[int, int]:
        if idx >= self.width * self.height:
            raise ValueError(f"Index {idx} exceeds maximum index of {self.width * self.height - 1} for {self.width}x{self.height} tiling")
        y = (idx // self.width)
        x = (idx % self.width)
        return (x, y)

    def coordinate_to_index(self, coordinate: Tuple[int, int]) -> int:
        x = coordinate[0]
        y = coordinate[1]
        if x >= self.width or y >= self.height:
            raise ValueError(f"Coordinate ({x},{y}) is out of bounds for {self.width}x{self.height} tiling")
        return y * self.width + x