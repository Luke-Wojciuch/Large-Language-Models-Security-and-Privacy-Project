from decimal import Decimal
from fractions import Fraction
from math import ceil


def select_tasks(task_ids, limit):
    if limit is None:
        return None
    if isinstance(limit, Decimal):
        limit = ceil(Fraction(limit) * len(task_ids))
    return list(task_ids)[:limit]
