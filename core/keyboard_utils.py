import math


def pinch_distance(thumb_tip: tuple[int,int], index_tip: tuple[int,int]) -> float:
    return math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
