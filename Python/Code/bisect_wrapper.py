"""
Utilities for fast bisect-based range searching over lists of dicts based on the original bisect module.
Primary use case: searching matching m/z values between datasets using SortedBisectableWrapper.get_indexes_between().


Last update:  2026  Apr 27
upload only, assuming it works lol
"""
# ---------------------------------------------------------------------------

import bisect
import numpy as np


# ---------------------------------------------------------------------------
# Low-level bisect helpers
# ---------------------------------------------------------------------------

def lt_index(input_list: list, x) -> int:
    """Return the rightmost index whose value is strictly less than x."""
    i = bisect.bisect_left(input_list, x)
    if i:
        return i - 1
    raise ValueError(f"No value less than {x} found in list.")


def gt_index(input_list: list, x) -> int:
    """Return the leftmost index whose value is strictly greater than x."""
    i = bisect.bisect_right(input_list, x)
    if i != len(input_list):
        return i
    raise ValueError(f"No value greater than {x} found in list.")


def lt_or_eq_index(input_list: list, x) -> int:
    """Return the rightmost index whose value is less than or equal to x."""
    i = bisect.bisect_right(input_list, x)
    if i:
        return i - 1
    raise ValueError(f"No value less than or equal to {x} found in list.")


def gt_or_eq_index(input_list: list, x) -> int:
    """Return the leftmost index whose value is greater than or equal to x."""
    i = bisect.bisect_left(input_list, x)
    if i != len(input_list):
        return i
    raise ValueError(f"No value greater than or equal to {x} found in list.")


# ---------------------------------------------------------------------------
# DictListIndexWrapper  (sorted input)
# ---------------------------------------------------------------------------

class DictListIndexWrapper:
    """
    Wraps a list of dicts so that bisect can operate on a single key field,
    assuming the list is already sorted by that key (or can be sorted in place).
    """

    def __init__(self, dict_list: list[dict], key: str, check_sorted: bool = False):
        self.dict_list = dict_list
        self.key = key

        if check_sorted:
            is_sorted = all(
                dict_list[i][key] <= dict_list[i + 1][key]
                for i in range(len(dict_list) - 1)
            )
            if not is_sorted:
                self.dict_list.sort(key=lambda d: d[key])

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.dict_list)

    def __getitem__(self, index: int):
        return self.dict_list[index][self.key]

    def get(self, index: int, key: str):
        return self.dict_list[index][key]

    def get_dict_list(self) -> list[dict]:
        return self.dict_list

    # -- internal range helper ----------------------------------------------

    def _get_boundary_indexes(
        self,
        lower_limit,
        upper_limit,
        greater_or_equal: bool = False,
        less_than_or_equal: bool = False,
    ) -> tuple[int, int]:
        """
        Return (lower_index, upper_index) for values within [lower_limit, upper_limit].

        Returns (-1, -2) when no matching range exists so that
        range(-1, -2 + 1) yields an empty sequence.
        """
        try:
            lower_index = (
                gt_or_eq_index(self, lower_limit)
                if greater_or_equal
                else gt_index(self, lower_limit)
            )
            upper_index = (
                lt_or_eq_index(self, upper_limit)
                if less_than_or_equal
                else lt_index(self, upper_limit)
            )
        except ValueError:
            return -1, -2

        return lower_index, upper_index

    # -- public range queries -----------------------------------------------

    def get_values_between(
        self,
        lower_limit,
        upper_limit,
        return_generator: bool = False,
        greater_or_equal: bool = False,
        less_than_or_equal: bool = False,
    ):
        """Return key values whose value falls between lower_limit and upper_limit."""
        lo, hi = self._get_boundary_indexes(
            lower_limit, upper_limit, greater_or_equal, less_than_or_equal
        )
        indices = range(lo, hi + 1)
        result = (self[i] for i in indices)
        return result if return_generator else list(result)

    def get_items_between(
        self,
        lower_limit,
        upper_limit,
        return_generator: bool = False,
        greater_or_equal: bool = False,
        less_than_or_equal: bool = False,
    ):
        """Return original dicts whose key value falls between lower_limit and upper_limit."""
        lo, hi = self._get_boundary_indexes(
            lower_limit, upper_limit, greater_or_equal, less_than_or_equal
        )
        indices = range(lo, hi + 1)
        result = (self.dict_list[i] for i in indices)
        return result if return_generator else list(result)

    def get_indexes_between(
        self,
        lower_limit,
        upper_limit,
        return_generator: bool = False,
        greater_or_equal: bool = False,
        less_than_or_equal: bool = False,
    ):
        """Return original list indices whose key value falls between lower_limit and upper_limit."""
        lo, hi = self._get_boundary_indexes(
            lower_limit, upper_limit, greater_or_equal, less_than_or_equal
        )
        indices = range(lo, hi + 1)
        result = (i for i in indices)
        return result if return_generator else list(result)

    # -- nearest-value helpers ----------------------------------------------

    def get_closest_value(self, value):
        """Return the key value nearest to the given value."""
        index = bisect.bisect_left(self, value)

        if index == 0:
            return self[0]
        if index == len(self):
            return self[-1]

        before, after = self[index - 1], self[index]
        return after if (after - value) < (value - before) else before

    def get_closest_item(self, value) -> dict:
        """Return the dict whose key value is nearest to the given value."""
        index = bisect.bisect_left(self, value)

        if index == 0:
            return self.dict_list[0]
        if index == len(self):
            return self.dict_list[-1]

        before, after = self[index - 1], self[index]
        return (
            self.dict_list[index]
            if (after - value) < (value - before)
            else self.dict_list[index - 1]
        )


# ---------------------------------------------------------------------------
# SortedBisectableWrapper  (unsorted input — key use case)
# ---------------------------------------------------------------------------

class SortedBisectableWrapper(DictListIndexWrapper):
    """
    Extends DictListIndexWrapper to handle an *unsorted* list of dicts.

    Computes a sorted index array (via numpy argsort) so the original list
    is never mutated, while still allowing O(log n) bisect searches.

    Typical usage — fast m/z range lookup:

        wrapper = SortedBisectableWrapper(records, key="mz")
        hits = wrapper.get_indexes_between(
            lower_limit=precursor_mz - tolerance,
            upper_limit=precursor_mz + tolerance,
            greater_or_equal=True,
            less_than_or_equal=True,
        )
        matched = [records[i] for i in hits]

    Note: sorted_indexes stores numpy.int64 values internally; they are
    converted to plain Python ints when returned to callers.
    """

    def __init__(self, dict_list: list[dict], key: str):
        super().__init__(dict_list, key, check_sorted=False)

        # Argsort over a plain wrapper (no re-sorting) to get sorted order.
        plain = DictListIndexWrapper(dict_list, key, check_sorted=False)
        self.sorted_indexes: np.ndarray = np.argsort(plain)

    # -- sequence protocol (sorted view) ------------------------------------

    def __getitem__(self, index: int):
        return self.dict_list[self.sorted_indexes[index]][self.key]

    def get(self, index: int, key: str):
        return self.dict_list[self.sorted_indexes[index]][key]

    # -- public range queries -----------------------------------------------

    def get_items_between(
        self,
        lower_limit,
        upper_limit,
        return_generator: bool = False,
        greater_or_equal: bool = False,
        less_than_or_equal: bool = False,
    ):
        """Return original dicts whose key value falls between lower_limit and upper_limit."""
        lo, hi = self._get_boundary_indexes(
            lower_limit, upper_limit, greater_or_equal, less_than_or_equal
        )
        result = (self.dict_list[self.sorted_indexes[i]] for i in range(lo, hi + 1))
        return result if return_generator else list(result)

    def get_indexes_between(
        self,
        lower_limit,
        upper_limit,
        return_generator: bool = False,
        greater_or_equal: bool = False,
        less_than_or_equal: bool = False,
        skip_indexes: list[int] | set[int] = (),
    ) -> list[int]:
        """
        Return original list indices whose key value falls between lower_limit and upper_limit.

        Parameters
        ----------
        lower_limit, upper_limit : numeric
            Search bounds (e.g. precursor_mz ± tolerance).
        greater_or_equal : bool
            Include values equal to lower_limit (default: strictly greater).
        less_than_or_equal : bool
            Include values equal to upper_limit (default: strictly less).
        skip_indexes : list or set of int
            Original indices to exclude from results.
        return_generator : bool
            Return a generator instead of a list (default: False).
        """
        lo, hi = self._get_boundary_indexes(
            lower_limit, upper_limit, greater_or_equal, less_than_or_equal
        )
        skip = set(skip_indexes)  # O(1) lookup
        result = (
            int(self.sorted_indexes[i])
            for i in range(lo, hi + 1)
            if int(self.sorted_indexes[i]) not in skip
        )
        return result if return_generator else list(result)
