# Copyright 2021 The Distla Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Utility functions for sharded probability function classes.
"""


def to_index(targets, labels):
    return tuple([labels.index(t) for t in targets if t in labels])


def permute(lst, perm):
    """Permute the given list by the permutation.

    Args:
      lst: The given list.
      perm: The permutation. The integer values are the source indices, and the
        index of the integer is the destination index.

    Returns:
      A permutation copy of lst.
    """
    return tuple([lst[i] for i in perm])


def relative_permutation(orig, new):
    """Calculates the permutation between orig and new.

    For example, if
    orig = ["a", "b", "c", "d", "e"]
    and new = ["c", "e", "d", "b", "a"],
    then this function will return [2, 4, 3, 1, 0].

    Args:
      orig: A list of hashable objects (usually ints).
      new: A list of hashable objects (usually ints).

    Returns:
      The permutation between the two lists.
    """
    if set(orig) != set(new):
        raise ValueError("Given lists do not have the same elements")
    position_mapping = {x: i for i, x in enumerate(orig)}
    return tuple([position_mapping[x] for x in new])


def invert_permutation(perm):
    """Invert a permutation list.

    This will take a permutation list and invert the permutation.

    Examples:
      [0, 1, 2] -> [0, 1, 2]
      [1, 2, 0] -> [2, 0, 1]
      [3, 0, 1, 2] -> [1, 2, 3, 0]

    Note that `inv_perm(inv_perm((perm)) == perm`.

    Args:
      perm: A list of integers. The following must hold `set(range(len(perm))) ==
        set(perm)`

    Returns:
      The inverted permutation
    """
    return relative_permutation(perm, list(range(len(perm))))


def remove_and_reduce(perm, values):
    """Remove the values in perm and reduce the remaining values.

    This function removes the elements in `values` from `perm` and reducing the
    remaining values of `perm` so that they create a valid (and smaller)
    permutation.

    Example:
      perm =  [1, 4, 3, 0, 2]
      values = [1, 3]
      reduce_and_remove(perm, values) # -> [2, 0, 1]

    Args:
      perm: The original permutation.
      values: The elements to remove from perm.

    Returns:
      A new smaller permutation.
    """
    assert set(perm) == set(range(len(perm))), f"{perm} is not a permutation"
    values = list(reversed(sorted(values)))
    perm = [x for x in perm if x not in values]
    for x in values:
        for i in range(len(perm)):
            if x <= perm[i]:
                perm[i] -= 1
    return tuple(perm)


def send_to_left_side(targets, values):
    """Send the given target values to the left of all other values.

    Example:
      targets = ["b", "x","c"]
      values = ["a", "b", "c", "x", "y", "z"]
      send_to_left_side(targets, values) # ->  ["b", "x", "c", "a", "y", "z"]

    Args:
      targets: Values to send to left side.
      values: The values of all elements.

    Returns:
      A list of elements of values in the desired permutation.
    """
    target_set = set(targets)
    return tuple(targets) + tuple([x for x in values if x not in target_set])


def send_to_right_side(targets, values):
    """Send the given target values to the right of all other values.

    Example:
      targets = ["b", "x","c"]
      values = ["a", "b", "c", "x", "y", "z"]
      send_to_right_side(targets, values) # ->  ["a", "y", "z", "b", "x", "c"]

    Args:
      targets: Values to send to right side.
      values: The values of all elements.

    Returns:
      A list of elements of values in the desired permutation.
    """
    target_set = set(targets)
    return tuple([x for x in values if x not in target_set] + list(targets))
