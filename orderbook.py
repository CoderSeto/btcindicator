"""
Order-book based signal: bid/ask volume imbalance near the top of book.
"""


def compute_imbalance(order_book: dict, depth: int) -> float:
    """
    Returns a value in [-1, 1]:
      +1  -> all volume in the top `depth` levels is on the bid side (buy pressure)
      -1  -> all volume is on the ask side (sell pressure)
       0  -> balanced
    """
    bids = order_book.get("bids", [])[:depth]
    asks = order_book.get("asks", [])[:depth]

    bid_volume = sum(float(level[1]) for level in bids)
    ask_volume = sum(float(level[1]) for level in asks)

    total = bid_volume + ask_volume
    if total == 0:
        return 0.0

    return (bid_volume - ask_volume) / total
