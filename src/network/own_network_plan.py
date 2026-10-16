"""Pure bounded planning: arbitrary identities, fair windows, measured capacity."""
import math

from src.learning.swarm_policy import IDENTIFIER


def choose_window(ready, history, *, limit=8):
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('Invalid own window bound')
    if (len(set(ready)) != len(ready) or len(ready) > 4096
            or any(not isinstance(w, str) or not IDENTIFIER.fullmatch(w) for w in ready)):
        raise ValueError('Invalid own ready identities')
    # Least recently completed first: a large colony cannot permanently lock
    # its first eight identities into every update. No hardware claim grants priority.
    return sorted(ready, key=lambda w: (history.get(w, {}).get('last_round', -1), w))[:limit]


def allocate(window, heads, history, *, start, base_count=64, training_units=4096):
    if (not window or len(window) > 8 or len(set(window)) != len(window)
            or type(training_units) is not int or not 1<=training_units<=4096
            or type(start) is not int or not 0 <= start < training_units
            or type(base_count) is not int or not 16 <= base_count <= 256):
        raise ValueError('Invalid own dynamic shard plan')
    rates = {}
    for worker in window:
        row = history.get(worker, {})
        # Only requester-observed completions earn unequal training assignments.
        # New workers receive a bounded equal probe, regardless of claimed hardware.
        rate = row.get('documents_per_second')
        rates[worker] = rate if row.get('samples', 0) >= 2 else None
        if rates[worker] is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0):
            raise ValueError('Invalid observed own capacity')
    observed = [v for v in rates.values() if v is not None]
    reference = min(observed) if observed else 1.
    result = []; cursor = start
    for worker in window:
        if cursor >= training_units: break
        count = min(256, max(16, round(base_count * (min(4., rates[worker] / reference) if rates[worker] else 1.))))
        count = min(count, training_units - cursor)
        result.append({'shard_id': str(cursor), 'worker_id': worker, 'start': cursor, 'count': count})
        cursor += count
    return result


def replacement(ready, assignments, contributions, failed):
    """Do not duplicate a completed shard or queue two jobs on the same worker."""
    busy = {r['worker_id'] for r in assignments if r['shard_id'] not in contributions}
    candidates = [w for w in ready if w not in failed and w not in busy]
    return min(candidates) if candidates else None
