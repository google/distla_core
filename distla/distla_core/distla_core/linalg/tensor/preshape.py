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
from typing import (Tuple, Sequence, Text, Any, List)

import jax
from jax.interpreters import pxla
import numpy as np

from distla_core.utils import config
from distla_core.utils import misc

AXIS_NAME = config.get_axis_name()


def apply_forward_transform(tensor: pxla.ShardedDeviceArray,
                            op: Tuple[Text, Tuple[Any, Any]]):
  """
  Helper function for `pravel` and `preshape`.
  Apply `op` to tensor. `op[0]` can by any of
  {'reshape', 'transpose', 'pswapaxes'}.
  `op[1]` contains the arguments to
  the corresponding jax function, e.g.
  for op[0]=='reshape', op[1][0] should
  contain the shape of `tensor` prior to reshaping it,
  and op[1][1] the final shape of the reshaping operation,
  such that jax.numpy.reshape(tensor, op[1][1]) gives the
  desired outcome.

  Args:
    tensor: An input tensor.
    op: Tuple[Text, Tuple[Any, Any]] describing the
      the actual operation and the arguments to it,
      such that both the forward in inverse operation
      can be reconstructed from it.

  Returns:
    ShardedDeviceArray: `op` applied to `tensor`.
  """
  if op[0] == 'reshape':
    return tensor.reshape(op[1][1])
  if op[0] == 'transpose':
    return tensor.transpose(op[1])
  if op[0] == 'pswapaxes':
    return jax.lax.pswapaxes(tensor, **op[1])
  raise ValueError(f"unknown op {op[0]}")


def apply_reverse_transform(tensor, op):
  """
  Helper function for `pravel` and `preshape`.
  Apply inverse `op` to tensor. `op[0]` can by any of
  {'reshape', 'transpose', 'pswapaxes'}.
  `op[1]` contains the arguments to
  the corresponding jax function, e.g.
  for op[0]=='reshape', op[1][0] should
  contain the shape of `tensor` prior to reshaping it,
  and op[1][1] the final shape of the reshaping operation
  This functions will then
  apply the inverse reshape operaetion,
  i.e. jax.numpy.reshape(tensor, op[1][0]).

  Args:
    tensor: An input tensor.
    op: Tuple[Text, Tuple[Any, Any]] describing the
      the actual operation and the arguments to it,
      such that both the forward in inverse operation
      can be reconstructed from it.

  Returns:
    ShardedDeviceArray: inverse of `op` applied to `tensor`.
  """

  if op[0] == 'reshape':
    return tensor.reshape(op[1][0])
  if op[0] == 'transpose':
    return tensor.transpose(misc.inverse_permutation(op[1]))
  if op[0] == 'pswapaxes':
    return jax.lax.pswapaxes(tensor, **op[1])
  raise ValueError("unknown op")


def apply_ops(tensor: pxla.ShardedDeviceArray,
              ops: List[Tuple[Text, Tuple[Any, Any]]]):
  """
  Helper function. Applies all `ops` to `tensor`.
  """
  for op in ops:
    tensor = apply_forward_transform(tensor, op)
  return tensor


def apply_reverse_ops(tensor: pxla.ShardedDeviceArray,
                      ops: List[Tuple[Text, Tuple[Any, Any]]]):
  """
  Helper function. Applies inverse of all `ops` to `tensor`,
  in reversed order.
  """

  for op in reversed(ops):
    tensor = apply_reverse_transform(tensor, op)
  return tensor


def compute_transformation_sequence_case_3(cumprod, local_shape, ind,
                                           sharded_leg_pos, pgrid):
  """
  Helper function for `pravel`, see `pravel` for more details.
  """
  ops = []
  ndev = np.prod(pgrid)
  orig_left_shape = tuple(local_shape[:sharded_leg_pos])
  orig_right_shape = tuple(local_shape[sharded_leg_pos:])
  shape_1 = orig_left_shape + (ndev,) + (np.prod(orig_right_shape) // ndev,)
  ops.append(('reshape', [local_shape, shape_1]))
  ops.append(('pswapaxes', {
      'axis_name': AXIS_NAME,
      'axis': len(orig_left_shape)
  }))

  remainder = ndev // cumprod[ind - 1]
  split_pgrid = tuple(pgrid[:sharded_leg_pos]) + (
      remainder, pgrid[sharded_leg_pos] // remainder) + tuple(
          pgrid[sharded_leg_pos + 1:])
  remaining_pgrid = (pgrid[sharded_leg_pos] // remainder,) + tuple(
      pgrid[sharded_leg_pos + 1:])
  shape_2 = orig_left_shape + tuple(split_pgrid) + (
      np.prod(orig_right_shape) // ndev,)
  l = list(range(len(shape_2)))
  left = l[:len(orig_left_shape)]
  right = l[len(orig_left_shape):]
  perm_1 = misc.flatten([[r, l] for l, r in zip(left, right[:len(left)])
                        ]) + right[len(left):]
  shape_3 = (np.prod(shape_2[:2 * len(orig_left_shape) + 1]),) + tuple(
      shape_2[2 * len(orig_left_shape) + 1:])

  ops.append(('reshape', [shape_1, shape_2]))
  ops.append(('transpose', perm_1))
  perm_shape_2 = [shape_2[p] for p in perm_1]
  ops.append(('reshape', [perm_shape_2, shape_3]))

  # swap the first local axis with the sharded one
  ops.append(('pswapaxes', {'axis_name': AXIS_NAME, 'axis': 0}))
  shape_4 = (ndev, np.prod(remaining_pgrid), np.prod(orig_right_shape) // ndev)
  shape_5 = remaining_pgrid + orig_right_shape
  ops.append(('reshape', [shape_3, shape_4]))
  ops.append(('transpose', [1, 0, 2]))
  perm_shape_4 = [shape_4[p] for p in [1, 0, 2]]
  ops.append(('reshape', [perm_shape_4, shape_5]))

  # now we have the sharded legs in the right order
  # next we need to fix the order of the localized legs
  left = list(range(len(remaining_pgrid)))
  right = list(range(len(remaining_pgrid), 2 * len(remaining_pgrid)))
  perm_2 = misc.flatten([[l, r] for l, r in zip(left, right)])
  ops.append(('transpose', perm_2))

  perm_shape_5 = [shape_5[p] for p in perm_2]
  shape_6 = misc.maybe_ravel_shape(perm_shape_5)
  ops.append(('reshape', [perm_shape_5, shape_6]))
  return ops


def compute_transformation_sequence_case_2(local_shape, sharded_leg_pos, pgrid):
  """
  Helper function for `pravel`, see `pravel` for more details.
  """
  ops = []
  ndev = np.prod(pgrid)
  orig_left_shape = tuple(local_shape[:sharded_leg_pos])
  orig_right_shape = tuple(local_shape[sharded_leg_pos:])
  shape_1 = orig_left_shape + (ndev,) + (np.prod(orig_right_shape) // ndev,)
  ops.append(('reshape', [local_shape, shape_1]))
  ops.append(('pswapaxes', {
      'axis_name': AXIS_NAME,
      'axis': len(orig_left_shape)
  }))

  shape_2 = orig_left_shape + tuple(pgrid) + (
      np.prod(orig_right_shape) // ndev,)
  l = list(range(len(shape_2)))
  left = l[:len(orig_left_shape)]
  right = l[len(orig_left_shape):]
  perm_1 = misc.flatten([[r, l] for l, r in zip(left, right[:len(left)])
                        ]) + right[len(left):]
  shape_3 = (np.prod(shape_2[:2 * len(orig_left_shape) + 1]),) + tuple(
      shape_2[2 * len(orig_left_shape) + 1:])
  ops.append(('reshape', [shape_1, shape_2]))
  ops.append(('transpose', perm_1))
  perm_shape_2 = [shape_2[p] for p in perm_1]
  ops.append(('reshape', [perm_shape_2, shape_3]))

  # swap the first local axis with the sharded one
  ops.append(('pswapaxes', {'axis_name': AXIS_NAME, 'axis': 0}))
  # now we have the sharded legs in the right order
  # next we need to fix the order of the localized legs
  perm_2 = list(range(1,
                      len(pgrid[sharded_leg_pos + 1:]) + 1)) + [0] + [
                          len(pgrid[sharded_leg_pos + 1:]) + 1
                      ]
  shape_4 = tuple(pgrid[sharded_leg_pos + 1:]) + orig_right_shape
  ops.append(('transpose', perm_2))
  perm_shape_3 = [shape_3[p] for p in perm_2]
  ops.append(('reshape', [perm_shape_3, shape_4]))

  p = len(pgrid[sharded_leg_pos + 1:])
  left = list(range(p))
  right = list(range(p, len(shape_4)))

  perm_3 = [right[0]] + misc.flatten([
      [l, r] for l, r in zip(left, right[1:len(left) + 1])
  ]) + right[len(left) + 1:]

  ops.append(('transpose', perm_3))
  perm_shape_4 = [shape_4[p] for p in perm_3]
  shape_5 = misc.maybe_ravel_shape(perm_shape_4)
  ops.append(('reshape', [perm_shape_4, shape_5]))
  return ops


def compute_transformation_sequence_case_1(cumprod, local_shape, ind,
                                           sharded_leg_pos, pgrid):
  """
  Helper function for `pravel`, see `pravel` for more details.
  """
  ops = []
  ndev = np.prod(pgrid)
  if ndev % cumprod[ind - 1] != 0:
    raise ValueError("reshaping not possible")
  remainder = ndev // cumprod[ind - 1]
  # the local leg has to be divisible by the remainder,
  # otherwise we can't place the sharded legs that need to be
  # localized at their respective positions
  if local_shape[sharded_leg_pos] % remainder != 0:
    raise ValueError(
        f"tensor.shape[{sharded_leg_pos}] = {local_shape[sharded_leg_pos]}"
        f" is not divisible by a local remainder of {remainder}. "
        f"Try using a different shape for the input tensor")
  if np.prod(local_shape[sharded_leg_pos:]) % remainder != 0:
    raise ValueError("reshaping not possible 2")
  # the first index group contains all legs that are going to be sharded
  # the second index group contain is swapped with the currently sharded legs
  # the third group remains unchanged

  orig_left_shape = tuple(local_shape[:sharded_leg_pos],) + (remainder,)
  orig_right_shape = (local_shape[sharded_leg_pos] // remainder,) + tuple(
      local_shape[sharded_leg_pos + 1:])
  shape_1 = orig_left_shape + (ndev,) + (np.prod(orig_right_shape) // ndev,)
  ops.append(('reshape', [local_shape, shape_1]))
  ops.append(('pswapaxes', {
      'axis_name': AXIS_NAME,
      'axis': sharded_leg_pos + 1
  }))

  # the previously sharded legs are now localized at position
  # sharded_leg_pos + 1 we now split off the legs that need
  # to be distributed again and move them to the right of their
  # corresponding local legs
  shape_2 = orig_left_shape + tuple(pgrid) + (
      np.prod(orig_right_shape) // ndev,)
  l = list(range(len(shape_2)))
  left = l[:len(orig_left_shape)]
  right = l[len(orig_left_shape):]
  perm_1 = misc.flatten([[r, l] for l, r in zip(left, right[:len(left)])
                        ]) + right[len(left):]
  shape_3 = (np.prod(shape_2[:2 * len(orig_left_shape)]),) + tuple(
      shape_2[2 * len(orig_left_shape):])
  ops.append(('reshape', [shape_1, shape_2]))
  ops.append(('transpose', perm_1))
  perm_shape_2 = [shape_2[p] for p in perm_1]
  ops.append(('reshape', [perm_shape_2, shape_3]))
  ops.append(('pswapaxes', {'axis_name': AXIS_NAME, 'axis': 0}))
  # swap the first local axis with the sharded one

  # now we have the harded legs in the right order
  # next we need to fix the order of the localized legs
  perm_2 = list(range(1, len(
      pgrid[sharded_leg_pos:]))) + [0] + [len(pgrid[sharded_leg_pos:])]
  shape_4 = tuple(pgrid[sharded_leg_pos + 1:]) + orig_right_shape
  ops.append(('transpose', perm_2))
  perm_shape_3 = [shape_3[p] for p in perm_2]
  ops.append(('reshape', [perm_shape_3, shape_4]))

  p = len(pgrid[sharded_leg_pos + 1:])

  left = list(range(p))
  right = list(range(p + 1, len(shape_4)))
  perm_3 = [p] + misc.flatten([[l, r] for l, r in zip(left, right[:len(left)])
                              ]) + right[len(left):]

  ops.append(('transpose', perm_3))

  perm_shape_4 = [shape_4[p] for p in perm_3]
  shape_5 = misc.maybe_ravel_shape(perm_shape_4)
  ops.append(('reshape', [perm_shape_4, shape_5]))
  return ops


def compute_transformation_sequence_case_0(local_shape, sharded_leg_pos, pgrid):
  """
  Helper function for `pravel`, see `pravel` for more details.
  """
  ndev = np.prod(pgrid)
  ops = []

  orig_left_shape = tuple(local_shape[:sharded_leg_pos + 1])
  orig_right_shape = tuple(local_shape[sharded_leg_pos + 1:])
  shape_1 = orig_left_shape + (ndev,) + (np.prod(orig_right_shape) // ndev,)
  ops.append(('reshape', [local_shape, shape_1]))
  ops.append(('pswapaxes', {
      'axis_name': AXIS_NAME,
      'axis': len(orig_left_shape)
  }))

  shape_2 = orig_left_shape + tuple(pgrid) + (
      np.prod(orig_right_shape) // ndev,)

  l = list(range(len(shape_2)))
  left = l[:len(orig_left_shape)]
  right = l[len(orig_left_shape):]
  perm_1 = misc.flatten([[r, l] for l, r in zip(left, right[:len(left)])
                        ]) + right[len(left):]
  shape_3 = (np.prod(shape_2[:2 * len(orig_left_shape)]),) + tuple(
      shape_2[2 * len(orig_left_shape):])
  ops.append(('reshape', [shape_1, shape_2]))
  ops.append(('transpose', perm_1))
  perm_shape_2 = [shape_2[p] for p in perm_1]
  ops.append(('reshape', [perm_shape_2, shape_3]))

  # swap the first local axis with the sharded one
  ops.append(('pswapaxes', {'axis_name': AXIS_NAME, 'axis': 0}))
  # now we have the sharded legs in the right order
  # next we need to fix the order of the localized legs
  perm_2 = list(range(1,
                      len(pgrid[sharded_leg_pos + 1:]) + 1)) + [0] + [
                          len(pgrid[sharded_leg_pos + 1:]) + 1
                      ]
  shape_4 = tuple(pgrid[sharded_leg_pos + 1:]) + orig_right_shape
  ops.append(('transpose', perm_2))
  perm_shape_3 = [shape_3[p] for p in perm_2]
  ops.append(('reshape', [perm_shape_3, shape_4]))
  p = len(pgrid[sharded_leg_pos + 1:])

  left = list(range(p))
  right = list(range(p, len(shape_4)))

  perm_3 = misc.flatten([[l, r] for l, r in zip(left, right[:len(left)])
                        ]) + right[len(left):]
  ops.append(('transpose', perm_3))

  perm_shape_4 = [shape_4[p] for p in perm_3]
  shape_5 = misc.maybe_ravel_shape(perm_shape_4)
  ops.append(('reshape', [perm_shape_4, shape_5]))
  return ops


def _pravel(tensor: pxla.ShardedDeviceArray,
            processor_grid_shape: Sequence[int]):
  """
  Ravel a distributed array `tensor`, distributed on a processor-grid
  of shape `processor_grid_shape` in row-major order.
  This returns the "ravelled" tensor in a ShardedDeviceArray  `ravelled`
  such that np.ndarray(ravelled).ravel() is identical to
  to_numpy(tensor, processor_grid_shape).ravel().
  The method in essence localizes the distributed legs of `tensor`, places them
  at the correct local position, reshapes the local part of `tensor` into
  a 2d arrray and distributes the correct fraction of first index to
  all devices.
  The returned value `ravelled` may or may not be actually ravelled. The method
  uses a heuristic to keep the output in a shape close to (m*4,n*128) to
  avoid/minimize padding on ASIC.

  Args:
    tensor: The distributed tensor to be ravelled.
    processor_grid_shape: The shape of the processor grid
      according to which `tensor` is distributed. The method
      assumes that the processor labels are arranged into a grid
      of shape `processor_grid_shape` in row-major order, i.e.
      `np.arange(np.prod(processor_grid_shape).reshape(processor_grid_shape)`.

  Returns:
    ShardedDeviceArray: The ravelled tensor, distributed. The actual *local*
      shape of the result may or may not be 1d: the method tries to reshape the
      the local tensors into a 2d array to avoid/minimize zero padding on ASICs.
  """
  pgrid = np.array(processor_grid_shape)
  local_shape = np.array(tensor.shape)
  ndev = np.prod(pgrid)

  # determine which local legs and which sharded legs need to be distributed
  # there are four cases to be considered:
  #  case 0: no sharded leg needs to be split and no local leg is split,
  #          and the number of devices `ndev` fits exactly into an even product
  #          of sharded and local leg-dimensions (i.e. the number of sharded and
  #          local legs is equal.
  #  case 1: no sharded leg needs to be split, but the number of devices
  #          `ndev` does not fit exactly into an even product of sharded and
  #          local leg-dimensions. Additionally, a local leg has to be split
  #          in order to obtain a dimension of `ndev`,
  #  case 2: no sharded leg needs to be split and no local leg has to be split,
  #          and the number of devices `ndev` fits exactly into an odd product
  #          of sharded and local leg-dimensions (i.e. number of sharded legs
  #          is one larger than the number of local legs.
  #  case 3: no local leg needs to be split, but the number of devices
  #          `ndev` does not fit exactly into an even product of sharded and
  #          local leg-dimensions. In this case, a sharded leg has to be split
  #          in order to obtain a dimension of `ndev`,

  # once we know which combination of currently sharded and local legs is
  # going to be finally sharded, we split off a suitable index of local legs
  # and swap them with the currently global legs. The thus localized,
  # formerly global legs are moved to their corresponding local position and
  # are merged with their local counterpart. We then split off a leg of
  # dimension `ndev` from the first index, and swap it with the currently
  # sharded indices. The now localized leg is split up again into
  # its previously localized parts, each part is moved to its corresponding
  # position within the localized tensor and merged with it.
  # At this point the tensor is correctly ravelled.

  # Notes(mganahl): The process assumes certain properties for the leg
  #                 dimensions.
  #                 The code should be cleaned up and shortened by code reusage.

  combined_dims = np.array(
      misc.flatten([[p, s] for p, s in zip(pgrid, local_shape[:len(pgrid)])]))
  cumprod = np.cumprod(combined_dims)
  inds = np.nonzero(cumprod == ndev)[0]
  # sharded_leg_pos is the index of the last currently sharded leg
  # (counting from the left) that needs to participate in the final
  # sharding of the the ravelled tensor shard_local_ind is an integer
  # denoting whether (1) or not (0) the local index at position
  # `sharded_leg_pos` also participates in the sharding of the final
  # ravelled array.

  if len(inds) > 1:
    # some local dimension or sharded dimensions are 1
    x, y = np.divmod(inds, 2)
    _mask = y == 1
    if np.any(_mask):
      i = np.nonzero(_mask)[0][-1]
      sharded_leg_pos, shard_local_ind = x[i], y[i]
    else:
      sharded_leg_pos, shard_local_ind = x[0], y[0]
  else:
    ind = np.nonzero(cumprod >= ndev)[0][0]
    sharded_leg_pos, shard_local_ind = np.divmod(ind, 2)
  numel = np.prod(local_shape)

  if numel == np.prod(pgrid):
    raise ValueError(f"the number of local elements = {numel}"
                     f" equals the number of sharded elements {ndev}."
                     f" Use a larger dimensions or a smaller number of devices")

  if np.any(cumprod == ndev) and shard_local_ind == 1:
    # case 0
    ops = compute_transformation_sequence_case_0(local_shape, sharded_leg_pos,
                                                 pgrid)
  elif np.all(cumprod != ndev) and shard_local_ind == 1:
    # case 1
    ops = compute_transformation_sequence_case_1(cumprod, local_shape, ind,
                                                 sharded_leg_pos, pgrid)
  elif np.any(cumprod == ndev) and shard_local_ind == 0:
    # case 2
    ops = compute_transformation_sequence_case_2(local_shape, sharded_leg_pos,
                                                 pgrid)
  elif np.all(cumprod != ndev) and shard_local_ind == 0:
    # case 3
    ops = compute_transformation_sequence_case_3(cumprod, local_shape, ind,
                                                 sharded_leg_pos, pgrid)
  return apply_ops(tensor, ops)


def pravel(tensor: pxla.ShardedDeviceArray,
           processor_grid_shape: Sequence[int]) -> pxla.ShardedDeviceArray:
  """
  Ravel a distributed array `tensor`, distributed on a processor-grid
  of shape `processor_grid_shape` in row-major order.
  This returns the "ravelled" tensor in a ShardedDeviceArray  `ravelled`
  such that np.ndarray(ravelled).ravel() is identical to
  to_numpy(tensor, processor_grid_shape).ravel().
  The method in essence localizes the distributed legs of `tensor`, places them
  at the correct local position, reshapes the local part of `tensor` into
  a 2d arrray and distributes the correct fraction of first index to
  all devices.
  The local tensors of the returned array are guaranteed to be 1d, which
  can cause subtantial zero-padding on ASICs. To avoid this, use `_pravel`.

  Args:
    tensor: The distributed tensor to be ravelled.
    processor_grid_shape: The shape of the processor grid
      according to which `tensor` is distributed. The method
      assumes that the processor labels are arranged into a grid
      of shape `processor_grid_shape` in row-major order, i.e.
      `np.arange(np.prod(processor_grid_shape).reshape(processor_grid_shape)`.

  Returns:
    ShardedDeviceArray: The ravelled tensor, distributed. The resulting local
      part of the returned array is 1d!
  """
  maybe_ravelled = _pravel(tensor, processor_grid_shape)
  return maybe_ravelled.ravel()


def preshape(
    tensor: pxla.ShardedDeviceArray, shape: Sequence[int],
    processor_grid_shape: Sequence[int],
    old_processor_grid_shape: Sequence[int]) -> pxla.ShardedDeviceArray:
  """
  Reshape a ShardedDeviceArray `tensor` into `shape`. This method computes
  the operations necessary to ravel `tensor` and the operations necessary
  to ravel the desired final tensor with shape `shape` and
  distributed according `processor_grid_shape`.

  Args:
    tensor: A sharded array.
    shape: The final shape into which `tensor` should be reshaped.
    processor_grid_shape: The final processor grid shape according
      to which the outcome should be distributed.
    old_processor_grid_shape: The processor grid shape according
      to which `tensor` is currently distributed.

  Returns:
    ShardedDeviceArray: The result of the reshape operation.
  """
  pgrid = np.asarray(processor_grid_shape)
  local_shape = np.array(shape) // pgrid
  ndev = np.prod(pgrid)

  # The reshape is determined by the following strategy:
  # 1) determine the operations needed to ravel `tensor`.
  # 2) determine the operations needed to ravel a tensor with
  #    shape `shape` and processor grid shape `processor_grid_shape`.
  # 3) apply forward operations of 1) to `tensor`
  # 4) apply reverse operations of 2) to the outcome of 3).

  # see _pravel of more details on how to determine the necessary operations.

  combined_dims = np.array(
      misc.flatten([[p, s] for p, s in zip(pgrid, local_shape[:len(pgrid)])]))
  cumprod = np.cumprod(combined_dims)
  inds = np.nonzero(cumprod == ndev)[0]
  # sharded_leg_pos is the index of the last currently sharded leg
  # (counting from the left) that needs to participate in the final
  # sharding of the the ravelled tensor shard_local_ind is an integer
  # denoting whether (1) or not (0) the local index at position
  # `sharded_leg_pos` also participates in the sharding of the final
  # ravelled array.

  if len(inds) > 1:
    x, y = np.divmod(inds, 2)
    _mask = y == 1
    if np.any(_mask):
      i = np.nonzero(_mask)[0][-1]
      sharded_leg_pos, shard_local_ind = x[i], y[i]
    else:
      sharded_leg_pos, shard_local_ind = x[0], y[0]
  else:
    ind = np.nonzero(cumprod >= ndev)[0][0]
    sharded_leg_pos, shard_local_ind = np.divmod(ind, 2)
  numel = np.prod(local_shape)

  if numel == np.prod(pgrid):
    raise ValueError(f"the number of local elements ={numel}"
                     f" equals the number of sharded elements {ndev}."
                     f" Use a larger dimensions or a smaller number of devices")

  if np.any(cumprod == ndev) and shard_local_ind == 1:
    # case 0
    ops = compute_transformation_sequence_case_0(local_shape, sharded_leg_pos,
                                                 pgrid)
  elif np.all(cumprod != ndev) and shard_local_ind == 1:
    # case 1
    ops = compute_transformation_sequence_case_1(cumprod, local_shape, ind,
                                                 sharded_leg_pos, pgrid)
  elif np.any(cumprod == ndev) and shard_local_ind == 0:
    # case 2
    ops = compute_transformation_sequence_case_2(local_shape, sharded_leg_pos,
                                                 pgrid)
  elif np.all(cumprod != ndev) and shard_local_ind == 0:
    # case 3
    ops = compute_transformation_sequence_case_3(cumprod, local_shape, ind,
                                                 sharded_leg_pos, pgrid)

  return apply_reverse_ops(
      _pravel(tensor, np.asarray(old_processor_grid_shape)), ops)
