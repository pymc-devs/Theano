import aesara
from aesara.graph.opt import copy_stack_trace, local_optimizer
from aesara.tensor.basic_opt import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from aesara.tensor.math import Dot, dot
from aesara.tensor.shape import shape_tuple
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    Subtensor,
    advanced_subtensor1,
    get_idx_list,
    inc_subtensor,
)
from aesara.tensor.type_other import NoneTypeT, SliceConstant, SliceType
from aesara.tensor.var import TensorConstant, TensorVariable


def transform_take(a, indices, axis):
    r"""Transform ``arr[:,:,:,indices,...]``-like operations into single-dimensional, vector index operations.

    This effectively converts certain `AdvancedSubtensor` `Op`\s into a
    combination of `AdvancedSubtensor1`, `Dimshuffle`, and `Reshape` `Op`\s,
    which can be more efficient.

    Parameters
    ----------
    a : TensorVariable
        The source array.
    indices : TensorVariable, ndarray, list, tuple
        The indices of the values to extract.
    axis : int
        The axis over which to select values. By default, the flattened
        input array is used.

    """
    a = aesara.tensor.as_tensor_variable(a)
    indices = aesara.tensor.as_tensor_variable(indices)
    # We can use the more efficient `AdvancedSubtensor1` if `indices` is a vector
    if indices.ndim == 1:
        if axis == 0:
            return advanced_subtensor1(a, indices)
        else:
            shuffle = list(range(a.ndim))
            shuffle[0] = axis
            shuffle[axis] = 0
            res = advanced_subtensor1(a.dimshuffle(shuffle), indices).dimshuffle(
                shuffle
            )
            return res

    # We can reshape and flatten the indices in order to use an
    # `AdvancedSubtensor1` `Op` per the above
    indices_shape = shape_tuple(indices)
    a_shape = shape_tuple(a)

    shape_parts = [
        a_shape[:axis],
        indices_shape,
        a_shape[axis + 1 :],
    ]

    shape_parts = [sp for sp in shape_parts if len(sp) > 0]

    assert len(shape_parts) > 0

    if len(shape_parts) > 1:
        shape = aesara.tensor.concatenate(shape_parts)
    else:
        shape = shape_parts[0]

    ndim = a.ndim + indices.ndim - 1

    return transform_take(a, indices.flatten(), axis).reshape(shape, ndim)


def is_full_slice(x):
    """Determine if `x` is a ``slice(None)`` or a symbolic equivalent."""
    if (
        (isinstance(x, slice) and x == slice(None))
        or (isinstance(x, SliceConstant) and x.value == slice(None))
        or (
            not isinstance(x, SliceConstant)
            and isinstance(getattr(x, "type", None), SliceType)
            and x.owner is not None
            and all(
                isinstance(getattr(i, "type", None), NoneTypeT) for i in x.owner.inputs
            )
        )
    ):
        return True
    return False


def get_advsubtensor_axis(indices):
    """Determine the axis at which an array index is applied.

    This only works for ``take``-like indices: e.g. ``x[:, :, idx, ...]``.  For
    the above example, `get_advsubtensor_axis` would return ``2``.  If it
    encounters anything other than a set of `indices` containing full slices
    and an array/tensor index, it will return ``None``.

    """
    found_idx = False
    axis = 0
    for idx in indices:
        if not found_idx and is_full_slice(idx):
            # Preceding full slices
            axis += 1
        elif found_idx and not is_full_slice(idx):
            # We don't handle multiple indices
            return
        elif found_idx and is_full_slice(idx):
            # Trailing full slices
            continue
        else:
            found_idx = True

    if isinstance(
        indices[axis], (TensorConstant, TensorVariable, TensorSharedVariable)
    ):
        return axis


@register_specialize
@local_optimizer([AdvancedSubtensor])
def local_replace_AdvancedSubtensor(fgraph, node):
    r"""
    This rewrite converts expressions like ``X[..., y]`` into ``X.T[y].T``, for
    a vector ``y``, and ``X[z, ...]`` into ``X[z.flatten()].reshape(...)``, for a
    matrix ``z``.

    These rewrites replace `AdvancedSubtensor`\s with the more efficient
    `AdvancedSubtensor1` and `Subtensor` `Op`\s.
    """

    if not isinstance(node.op, AdvancedSubtensor):
        return

    indexed_var = node.inputs[0]
    indices = node.inputs[1:]

    axis = get_advsubtensor_axis(indices)

    if axis is None or indices[axis].dtype == "bool":
        # Booleans aren't handled
        return

    new_res = transform_take(indexed_var, indices[axis], axis)

    assert new_res.broadcastable == node.outputs[0].broadcastable

    copy_stack_trace(node.outputs[0], new_res)
    return [new_res]


@register_specialize
@local_optimizer([AdvancedIncSubtensor])
def local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1(fgraph, node):
    r"""Replace `AdvancedIncSubtensor`\s with `AdvancedIncSubtensor1`\s.

    This is only done when there's a single vector index.
    """

    if not isinstance(node.op, AdvancedIncSubtensor):
        return

    res = node.inputs[0]
    val = node.inputs[1]
    indices = node.inputs[2:]

    axis = get_advsubtensor_axis(indices)

    if axis is None or indices[axis].dtype == "bool":
        # Booleans aren't handled
        return

    new_subtensor = transform_take(res, indices[axis], axis)
    set_instead_of_inc = node.op.set_instead_of_inc
    inplace = node.op.inplace

    new_res = inc_subtensor(
        new_subtensor, val, inplace=inplace, set_instead_of_inc=set_instead_of_inc
    )
    copy_stack_trace(node.outputs[0], new_res)
    return [new_res]


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([Subtensor])
def local_subtensor_of_dot(fgraph, node):
    """Rewrite ``aet.dot(A, B)[idxs]`` into ``aet.dot(A[idxs_a], B[idxs_b])``.

    ``idxs_a`` is the first ``A.ndim-1`` entries of ``idxs``, and ``idxs_b`` is
    the remaining entries of ``idxs`` (if any), modified to skip the
    second-to-last dimension of ``B`` (because dot sums over this dimension).

    """
    if not isinstance(node.op, Subtensor):
        return
    if not node.inputs[0].owner or not isinstance(node.inputs[0].owner.op, Dot):
        return
    # If there is other node that use the outputs of the dot
    # We don't want to compute twice the sub part.
    if len(fgraph.clients[node.inputs[0]]) > 1:
        return

    a = node.inputs[0].owner.inputs[0]
    b = node.inputs[0].owner.inputs[1]

    idx_list = get_idx_list(node.inputs, node.op.idx_list)

    num_a_indices = min(a.ndim - 1, len(idx_list))
    a_indices = idx_list[:num_a_indices]
    b_indices = idx_list[num_a_indices:]

    # This is necessary because np.dot sums the last index of a with the second to last of b
    # so we want to skip the second-to-last index into b.
    # This wasn't necessary for a, because we just omitted the last index.
    # We skip this if b.ndim = 1, since then we just want b_sub = b, not b_sub = b[:]
    # (dot also handles b.ndim < 2 as a special case)
    if b.ndim > 1 and len(b_indices) >= b.ndim - 1:
        b_indices = (
            b_indices[: b.ndim - 2]
            + (slice(None, None, None),)
            + b_indices[b.ndim - 2 :]
        )

    a_sub = a.__getitem__(tuple(a_indices))
    b_sub = b.__getitem__(tuple(b_indices)) if b_indices else b

    # Copy over previous output stacktrace to a_sub and b_sub,
    # because an error in the subtensor operation (e.g. an index error)
    # on either a or b must correspond to an error in the
    # subtensor operation on their dot product.
    copy_stack_trace(node.outputs[0], [a_sub, b_sub])

    # Copy over previous output stacktrace and previous dot product stacktrace,
    # because an error here may correspond to an either in either the original
    # dot product, or in the dot product after the subtensor operation.
    r = dot(a_sub, b_sub)
    copy_stack_trace([node.outputs[0], node.inputs[0]], r)

    return [r]
