"""Logical error-rate and threshold-style collection APIs."""

from faultscope.collection._collect import collect, iter_collect
from faultscope.collection._types import CollectionOptions, CollectionTask, TaskStats

__all__ = [
    "CollectionOptions",
    "CollectionTask",
    "TaskStats",
    "collect",
    "iter_collect",
]
