import inspect
from functools import singledispatch
from numbers import Number
from textwrap import indent
from typing import Union

import numba
import numpy as np
from numba.cpython.unsafe.tuple import tuple_setitem

from aesara import config
from aesara.graph.op import Op
from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import (
    create_numba_signature,
    create_tuple_creator,
    numba_funcify,
)
from aesara.link.utils import (
    compile_function_src,
    get_name_for_object,
    unique_name_generator,
)
from aesara.scalar.basic import (
    AND,
    OR,
    XOR,
    Add,
    IntDiv,
    Mul,
    ScalarMaximum,
    Sub,
    TrueDiv,
)
from aesara.scalar.basic import add as add_as
from aesara.scalar.basic import scalar_maximum
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.math import MaxAndArgmax
from aesara.tensor.nnet.basic import LogSoftmax, Softmax, SoftmaxGrad


def create_vectorize_func(op, node, use_signature=False, identity=None, **kwargs):
    scalar_op_fn = numba_funcify(op.scalar_op, node=node, inline="always", **kwargs)

    if len(node.outputs) > 1:
        raise NotImplementedError(
            "Multi-output Elemwise Ops are not supported by the Numba backend"
        )

    if use_signature:
        signature = [create_numba_signature(node, force_scalar=True)]
    else:
        signature = []

    target = (
        getattr(node.tag, "numba__vectorize_target", None)
        or config.numba__vectorize_target
    )

    numba_vectorized_fn = numba_basic.numba_vectorize(
        signature, identity=identity, target=target, fastmath=config.numba__fastmath
    )

    py_scalar_func = getattr(scalar_op_fn, "py_func", scalar_op_fn)

    elemwise_fn = numba_vectorized_fn(scalar_op_fn)
    elemwise_fn.py_scalar_func = py_scalar_func

    return elemwise_fn


@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, **kwargs):

    elemwise_fn = create_vectorize_func(op, node, use_signature=False)
    elemwise_fn_name = elemwise_fn.__name__

    if op.inplace_pattern:
        input_idx = op.inplace_pattern[0]
        sign_obj = inspect.signature(elemwise_fn.py_scalar_func)
        input_names = list(sign_obj.parameters.keys())

        unique_names = unique_name_generator([elemwise_fn_name, "np"], suffix_sep="_")
        input_names = [unique_names(i, force_unique=True) for i in input_names]

        updated_input_name = input_names[input_idx]

        inplace_global_env = {elemwise_fn_name: elemwise_fn, "np": np}

        inplace_elemwise_fn_name = f"{elemwise_fn_name}_inplace"

        input_signature_str = ", ".join(input_names)

        if node.inputs[input_idx].ndim > 0:
            inplace_elemwise_src = f"""
def {inplace_elemwise_fn_name}({input_signature_str}):
    return {elemwise_fn_name}({input_signature_str + ", " + updated_input_name})
            """
        else:
            # We can't perform in-place updates on Numba scalars, so we need to
            # convert them to NumPy scalars.
            # TODO: We should really prevent the rewrites from creating
            # in-place updates on scalars when the Numba mode is selected (or
            # in general?).
            inplace_elemwise_src = f"""
def {inplace_elemwise_fn_name}({input_signature_str}):
    {updated_input_name}_scalar = np.asarray({updated_input_name})
    return {elemwise_fn_name}({input_signature_str + ", " + updated_input_name}_scalar).item()
            """

        inplace_elemwise_fn = compile_function_src(
            inplace_elemwise_src,
            inplace_elemwise_fn_name,
            {**globals(), **inplace_global_env},
        )
        return numba_basic.numba_njit(inline="always", fastmath=config.numba__fastmath)(
            inplace_elemwise_fn
        )

    return elemwise_fn


@singledispatch
def scalar_in_place_fn(op: Op, idx: str, res: str, arr: str):
    """Return code for an in-place update on an array using a binary scalar :class:`Op`.

    Parameters
    ----------
    op
        The scalar :class:`Op`
    idx
        The index of `res` that needs to be updated.
    res
        The symbol name for the first input and results/output.
    arr
        The symbol name for the second input.
    """
    return f"{res}[{idx}] = {op.nfunc_spec[0]}({res}[{idx}], arr)"


@scalar_in_place_fn.register(Add)
def scalar_in_place_fn_Add(op, idx, res, arr):
    return f"{res}[{idx}] += {arr}"


@scalar_in_place_fn.register(Mul)
def scalar_in_place_fn_Mul(op, idx, res, arr):
    return f"{res}[{idx}] *= {arr}"


@scalar_in_place_fn.register(Sub)
def scalar_in_place_fn_Sub(op, idx, res, arr):
    return f"{res}[{idx}] -= {arr}"


@scalar_in_place_fn.register(AND)
def scalar_in_place_fn_AND(op, idx, res, arr):
    return f"{res}[{idx}] &= {arr}"


@scalar_in_place_fn.register(OR)
def scalar_in_place_fn_OR(op, idx, res, arr):
    return f"{res}[{idx}] |= {arr}"


@scalar_in_place_fn.register(XOR)
def scalar_in_place_fn_XOR(op, idx, res, arr):
    return f"{res}[{idx}] ^= {arr}"


@scalar_in_place_fn.register(TrueDiv)
def scalar_in_place_fn_TrueDiv(op, idx, res, arr):
    return f"{res}[{idx}] /= {arr}"


@scalar_in_place_fn.register(IntDiv)
def scalar_in_place_fn_IntDiv(op, idx, res, arr):
    return f"{res}[{idx}] //= {arr}"


@scalar_in_place_fn.register(ScalarMaximum)
def scalar_in_place_fn_ScalarMaximum(op, idx, res, arr):
    return f"""
if {res}[{idx}] < {arr}:
    {res}[{idx}] = {arr}
"""


def create_axis_reducer(
    scalar_op: Op,
    identity: Union[np.ndarray, Number],
    axis: int,
    ndim: int,
    dtype: numba.types.Type,
    keepdims: bool = False,
) -> numba.core.dispatcher.Dispatcher:
    r"""Create a Numba JITed function that performs a NumPy reduction on a given axis.

    The functions generated by this function take the following form:

    .. code-block:: python

        def careduce_axis(x):
            res_shape = tuple(shape[i] if i < axis else shape[i + 1] for i in range(ndim - 1))
            res = np.full(res_shape, identity, dtype=dtype)

            x_axis_first = x.transpose(reaxis_first)

            for m in range(x.shape[axis]):
                reduce_fn(res, x_axis_first[m], res)

            if keepdims:
                return np.expand_dims(res, axis)
            else:
                return res


    This can be removed/replaced when
    https://github.com/numba/numba/issues/4504 is implemented.

    Parameters
    ==========
    scalar_op:
        The scalar :class:`Op` that performs the desired reduction.
    identity:
        The identity value for the reduction.
    axis:
        The axis to reduce.
    ndim:
        The number of dimensions of the result.
    dtype:
        The data type of the result.
    keepdims:
        Determines whether or not the reduced dimension is retained.
    """

    reduce_elemwise_fn_name = "careduce_axis"

    if ndim > 1:
        res_shape_tuple_ctor = create_tuple_creator(
            lambda i, shape: shape[i] if i < axis else shape[i + 1], ndim - 1
        )
        if keepdims:
            set_out_dims = numba_basic.numba_njit(
                lambda x: np.expand_dims(x, axis), inline="always"
            )
        else:
            set_out_dims = numba_basic.numba_njit(lambda x: x, inline="always")

    else:

        @numba_basic.numba_njit
        def res_shape_tuple_ctor(args):
            return 1

        if keepdims:
            set_out_dims = numba_basic.numba_njit(
                lambda x: numba_basic.direct_cast(x, dtype), inline="always"
            )
        else:
            set_out_dims = numba_basic.numba_njit(
                lambda x: numba_basic.direct_cast(x[0], dtype), inline="always"
            )

    identity = str(identity)
    if identity == "inf":
        identity = "np.inf"
    elif identity == "-inf":
        identity = "-np.inf"

    if ndim > 1:
        res_indices = []
        arr_indices = []
        count = 0

        for i in range(ndim):
            if i == axis:
                arr_indices.append("i")
            else:
                res_indices.append(f"idx_arr[{count}]")
                arr_indices.append(f"idx_arr[{count}]")
                count = count + 1

        res_indices = ", ".join(res_indices)
        arr_indices = ", ".join(arr_indices)

        inplace_update_statement = scalar_in_place_fn(
            scalar_op, res_indices, "res", f"x[{arr_indices}]"
        )
        inplace_update_statement = indent(inplace_update_statement, " " * 4 * 3)

        reduce_elemwise_def_src = f"""
def {reduce_elemwise_fn_name}(x):

    res_shape = res_shape_tuple_ctor(x.shape)
    res = np.full(res_shape, numba_basic.to_scalar({identity}))

    axis_shape = x.shape[{axis}]

    for idx_arr in np.ndindex(res_shape):
        for i in range(axis_shape):
            {inplace_update_statement}

    return set_out_dims(res)
        """
    else:
        inplace_update_statement = scalar_in_place_fn(scalar_op, "0", "res", "x[i]")
        inplace_update_statement = indent(inplace_update_statement, " " * 4 * 3)

        reduce_elemwise_def_src = f"""
def {reduce_elemwise_fn_name}(x):

    res_shape = res_shape_tuple_ctor(x.shape)
    res = np.full(res_shape, numba_basic.to_scalar({identity}))

    axis_shape = x.shape[{axis}]

    for i in range(axis_shape):
        {inplace_update_statement}

    return set_out_dims(res)
        """

    global_env = {
        "np": np,
        "res_shape_tuple_ctor": res_shape_tuple_ctor,
        "numba_basic": numba_basic,
        "set_out_dims": set_out_dims,
    }
    reduce_elemwise_fn_py = compile_function_src(
        reduce_elemwise_def_src, reduce_elemwise_fn_name, global_env
    )
    return numba_basic.numba_njit(boundscheck=False)(reduce_elemwise_fn_py)


def create_multiaxis_reducer(
    scalar_op, identity, axes, ndim, dtype, input_name="input"
):
    r"""Construct a function that reduces multiple axes.

    The functions generated by this function take the following form:

    .. code-block:: python

        def careduce_maximum(input):
            axis_0_res = careduce_axes_fn_0(input)
            axis_1_res = careduce_axes_fn_1(axis_0_res)
            ...
            axis_N_res = careduce_axes_fn_N(axis_N_minus_1_res)
            return axis_N_res

    The range 0-N is determined by the `axes` argument (i.e. the
    axes to be reduced).


    Parameters
    ==========
    scalar_op:
        The scalar :class:`Op` that performs the desired reduction.
    identity:
        The identity value for the reduction.
    axes:
        The axes to reduce.
    ndim:
        The number of dimensions of the result.
    dtype:
        The data type of the result.

    """
    if len(axes) == 1:
        return create_axis_reducer(scalar_op, identity, axes[0], ndim, dtype)

    careduce_fn_name = f"careduce_{scalar_op}"
    global_env = {}
    to_reduce = reversed(sorted(axes))
    careduce_lines_src = []
    var_name = input_name

    for i, axis in enumerate(to_reduce):
        careducer_axes_fn_name = f"careduce_axes_fn_{i}"
        global_env[careducer_axes_fn_name] = create_axis_reducer(
            scalar_op, identity, axis, ndim, dtype
        )
        ndim -= 1
        last_var_name = var_name
        var_name = f"axis_{i}_res"
        careduce_lines_src.append(
            f"{var_name} = {careducer_axes_fn_name}({last_var_name})"
        )

    careduce_assign_lines = indent("\n".join(careduce_lines_src), " " * 4)
    careduce_def_src = f"""
def {careduce_fn_name}({input_name}):
{careduce_assign_lines}
    return {var_name}
    """

    careduce_fn = compile_function_src(
        careduce_def_src, careduce_fn_name, {**globals(), **global_env}
    )
    return numba_basic.numba_njit(fastmath=config.numba__fastmath)(careduce_fn)


def create_axis_apply_fn(fn, axis, ndim, dtype):
    reaxis_first = tuple(i for i in range(ndim) if i != axis) + (axis,)

    @numba_basic.numba_njit(boundscheck=False)
    def axis_apply_fn(x):
        x_reaxis = x.transpose(reaxis_first)

        res = np.zeros(x_reaxis.shape[:-1], dtype=dtype)
        for m in np.ndindex(res.shape):
            v = fn(x_reaxis[m])
            res[m] = v
        return res

    return axis_apply_fn


@numba_funcify.register(CAReduce)
def numba_funcify_CAReduce(op, node, **kwargs):
    axes = op.axis
    if axes is None:
        axes = list(range(node.inputs[0].ndim))

    if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
        acc_dtype = op.acc_dtype
    else:
        acc_dtype = node.outputs[0].type.dtype

    np_acc_dtype = np.dtype(acc_dtype)

    scalar_op_identity = np.asarray(op.scalar_op.identity, dtype=np_acc_dtype)

    input_name = get_name_for_object(node.inputs[0])
    ndim = node.inputs[0].ndim
    careduce_fn = create_multiaxis_reducer(
        op.scalar_op,
        scalar_op_identity,
        axes,
        ndim,
        np_acc_dtype,
        input_name=input_name,
    )

    return careduce_fn


@numba_funcify.register(DimShuffle)
def numba_funcify_DimShuffle(op, **kwargs):
    shuffle = tuple(op.shuffle)
    transposition = tuple(op.transposition)
    augment = tuple(op.augment)
    inplace = op.inplace

    ndim_new_shape = len(shuffle) + len(augment)

    if len(shuffle) > 0:

        @numba_basic.numba_njit
        def populate_new_shape(i, j, new_shape, shuffle_shape):
            if i in augment:
                new_shape = tuple_setitem(new_shape, i, 1)
                return j, new_shape
            else:
                new_shape = tuple_setitem(new_shape, i, shuffle_shape[j])
                return j + 1, new_shape

    else:
        # When `len(shuffle) == 0`, the `shuffle_shape[j]` expression above is
        # is typed as `getitem(Tuple(), int)`, which has no implementation
        # (since getting an item from an empty sequence doesn't make sense).
        # To avoid this compile-time error, we omit the expression altogether.
        @numba_basic.numba_njit(inline="always")
        def populate_new_shape(i, j, new_shape, shuffle_shape):
            return j, tuple_setitem(new_shape, i, 1)

    if ndim_new_shape > 0:
        create_zeros_tuple = numba_basic.create_tuple_creator(
            lambda _: 0, ndim_new_shape
        )

        @numba_basic.numba_njit
        def dimshuffle_inner(x, shuffle):
            res = np.transpose(x, transposition)
            shuffle_shape = res.shape[: len(shuffle)]

            new_shape = create_zeros_tuple()

            j = 0
            for i in range(len(new_shape)):
                j, new_shape = populate_new_shape(i, j, new_shape, shuffle_shape)

            # FIXME: Numba's `array.reshape` only accepts C arrays.
            res_reshape = np.reshape(np.ascontiguousarray(res), new_shape)

            if not inplace:
                return res_reshape.copy()
            else:
                return res_reshape

    else:

        @numba_basic.numba_njit
        def dimshuffle_inner(x, shuffle):
            return x.item()

    # Without the following wrapper function we would see this error:
    # E   No implementation of function Function(<built-in function getitem>) found for signature:
    # E
    # E    >>> getitem(UniTuple(int64 x 2), slice<a:b>)
    # E
    # E   There are 22 candidate implementations:
    # E      - Of which 22 did not match due to:
    # E      Overload of function 'getitem': File: <numerous>: Line N/A.
    # E        With argument(s): '(UniTuple(int64 x 2), slice<a:b>)':
    # E       No match.
    # ...(on this line)...
    # E           shuffle_shape = res.shape[: len(shuffle)]
    @numba_basic.numba_njit(inline="always")
    def dimshuffle(x):
        return dimshuffle_inner(np.asarray(x), shuffle)

    return dimshuffle


@numba_funcify.register(Softmax)
def numba_funcify_Softmax(op, node, **kwargs):

    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    axis = op.axis

    if axis is not None:
        reduce_max = create_axis_reducer(
            scalar_maximum, -np.inf, axis, x_at.ndim, x_dtype, keepdims=True
        )
        reduce_sum = create_axis_reducer(
            add_as, 0.0, axis, x_at.ndim, x_dtype, keepdims=True
        )
    else:
        reduce_max = np.max
        reduce_sum = np.sum

    @numba_basic.numba_njit
    def softmax(x):
        z = reduce_max(x)
        e_x = np.exp(x - z)
        w = reduce_sum(e_x)
        sm = e_x / w
        return sm

    return softmax


@numba_funcify.register(SoftmaxGrad)
def numba_funcify_SoftmaxGrad(op, node, **kwargs):

    sm_at = node.inputs[1]
    sm_dtype = sm_at.type.numpy_dtype
    sm_dtype = numba.np.numpy_support.from_dtype(sm_dtype)

    axis = op.axis
    if axis is not None:
        reduce_sum = create_axis_reducer(
            add_as, 0.0, axis, sm_at.ndim, sm_dtype, keepdims=True
        )
    else:
        reduce_sum = np.sum

    @numba_basic.numba_njit
    def softmax_grad(dy, sm):
        dy_times_sm = dy * sm
        sum_dy_times_sm = reduce_sum(dy_times_sm)
        dx = dy_times_sm - sum_dy_times_sm * sm
        return dx

    return softmax_grad


@numba_funcify.register(LogSoftmax)
def numba_funcify_LogSoftmax(op, node, **kwargs):

    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    axis = op.axis

    if axis is not None:
        reduce_max = create_axis_reducer(
            scalar_maximum, -np.inf, axis, x_at.ndim, x_dtype, keepdims=True
        )
        reduce_sum = create_axis_reducer(
            add_as, 0.0, axis, x_at.ndim, x_dtype, keepdims=True
        )
    else:
        reduce_max = np.max
        reduce_sum = np.sum

    @numba_basic.numba_njit
    def log_softmax(x):
        xdev = x - reduce_max(x)
        lsm = xdev - np.log(reduce_sum(np.exp(xdev)))
        return lsm

    return log_softmax


@numba_funcify.register(MaxAndArgmax)
def numba_funcify_MaxAndArgmax(op, node, **kwargs):
    axis = op.axis
    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    x_ndim = x_at.ndim

    if x_ndim == 0:

        @numba_basic.numba_njit(inline="always")
        def maxandargmax(x):
            return x, 0

    else:

        axes = tuple(int(ax) for ax in axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = tuple(i for i in range(x_ndim) if i not in axes)

        reduce_max = create_multiaxis_reducer(
            scalar_maximum, -np.inf, axes, x_ndim, x_dtype
        )
        reduced_x_ndim = x_ndim - len(axes) + 1
        argmax_axis = create_axis_apply_fn(
            np.argmax, reduced_x_ndim - 1, reduced_x_ndim, np.int64
        )

        reaxis_order = keep_axes + axes
        sl1 = slice(None, len(keep_axes))
        sl2 = slice(len(keep_axes), None)

        @numba_basic.numba_njit
        def maxandargmax(x):
            max_res = reduce_max(x)

            # Not-reduced axes in front
            transposed_x = np.ascontiguousarray(np.transpose(x, reaxis_order))
            kept_shape = transposed_x.shape[sl1]
            reduced_shape = transposed_x.shape[sl2]
            reduced_size = 1
            for s in reduced_shape:
                reduced_size *= s

            # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
            # Otherwise reshape would complain citing float arg
            new_shape = kept_shape + (reduced_size,)
            reshaped_x = transposed_x.reshape(new_shape)

            max_idx_res = argmax_axis(reshaped_x)

            return max_res, max_idx_res

    return maxandargmax
