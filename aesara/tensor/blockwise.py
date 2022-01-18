from typing import Dict, List, Tuple

import numpy as np

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.tensor import get_scalar_constant_value
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.extra_ops import broadcast_shape
from aesara.tensor.shape import shape_tuple
from aesara.tensor.type import TensorType


def _update_dim_sizes(
    dim_sizes: Dict[str, int], arg: Variable, core_dims: Tuple[str, ...]
):
    """Incrementally check and update core dimension sizes for a single argument.

    From `numpy.lib.function_base`.

    Parameters
    ----------
    dim_sizes
        Sizes of existing core dimensions. Will be updated in-place.
    arg
        Argument to examine.
    core_dims
        Core dimensions for this argument.
    """
    if not core_dims:
        return

    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            f"{arg.ndim}-dimensional argument does not have enough "
            f"dimensions for all core dimensions: {core_dims}"
        )

    core_shape = shape_tuple(arg)[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim not in dim_sizes:
            dim_sizes[dim] = size
        # else:
        #     # This check can't be done (sufficiently) at compile-time
        #     if size != dim_sizes[dim]:
        #         raise ValueError(
        #             f"Inconsistent size for core dimension {dim}: {size} vs {dim_sizes[dim]}"
        #         )


def _parse_input_dimensions(
    args: Tuple[Variable, ...], input_core_dims: List[Tuple[str, ...]]
) -> Tuple[Tuple[Variable, ...], Dict[str, Variable]]:
    """Parse broadcast and core dimensions for vectorize with a signature.

    From `numpy.lib.function_base`.

    Parameters
    ----------
    args
        Tuple of input arguments to examine.
    input_core_dims
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape
        Common shape to broadcast all non-core dimensions to.
    dim_sizes
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        arg_shape = shape_tuple(arg)
        broadcast_args.append(arg_shape[:ndim])
    bcast_shape = broadcast_shape(*broadcast_args, arrays_are_shapes=True)
    return bcast_shape, dim_sizes


def _calculate_shapes(
    broadcast_shape: Tuple[Variable, ...],
    dim_sizes: Dict[str, Variable],
    list_of_core_dims: List[Tuple[str, ...]],
) -> List[Tuple[Variable, ...]]:
    """Helper for calculating broadcast shapes with core dimensions.

    From `numpy.lib.function_base`.

    """
    return [
        broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
        for core_dims in list_of_core_dims
    ]


def gufunc_sign_to_str(sign):
    in_sign = [f"({','.join(_sign)})" for _sign in sign[0]]
    out_sign = [f"({','.join(_sign)})" for _sign in sign[1]]
    return f"{','.join(in_sign)}->{','.join(out_sign)}"


class Blockwise(Op):
    __props__ = ("op", "signature")

    def __init__(self, op, signature=None):
        self.op = op
        self.signature = signature or self.op.gufunc_sig

    def make_node(self, *inputs):

        num_expected_inps = len(self.signature[0])
        if len(inputs) != num_expected_inps:
            raise ValueError(
                f"Expected {int(num_expected_inps)} inputs, got {len(inputs)}"
            )

        # TODO: Correct this
        out_dtype = inputs[0].dtype

        bcast_shape, dim_sizes = _parse_input_dimensions(inputs, self.signature[0])
        output_shapes = _calculate_shapes(bcast_shape, dim_sizes, self.signature[1])

        def safe_const_val(x):
            try:
                return get_scalar_constant_value(x)
            except NotScalarConstantError:
                return None

        outputs = [
            TensorType(out_dtype, shape=tuple(safe_const_val(s) for s in shp))()
            for shp in output_shapes
        ]
        return Apply(self, list(inputs), outputs)

    def infer_shape(self, fgraph, node, shapes):
        shape_idx = {}
        core_dims = []
        # TODO: Add broadcasting logic.
        for idx, inp_sign in enumerate(self.signature[0]):
            inp_shp = shapes[idx][-len(inp_sign) :]
            # Check length of all core dimensions are equal if any
            if core_dims:
                assert len(core_dims) == len(shapes[idx][: -len(inp_sign)])
            else:
                core_dims = shapes[idx][: -len(inp_sign)]

            for _inp_shp, _inp_sign in zip(inp_shp, inp_sign):
                shape_idx[_inp_sign] = _inp_shp

        out_shapes = []
        for idx, out_sign in enumerate(self.signature[1]):
            out_shapes.append(
                tuple(
                    list(core_dims) + [shape_idx[_out_sign] for _out_sign in out_sign]
                )
            )

        return out_shapes

    def grad(self, *args):
        raise NotImplementedError()

    def L_op(self, *args):
        raise NotImplementedError()

    def perform(self, node, inputs, outputs):
        def py_func(*inner_inputs):
            res = [[None]] * len(outputs)
            # This can be avoided by making a single dummy node
            # But will that cover all cases?
            node = self.op.make_node(*inner_inputs)
            self.op.perform(node, inner_inputs, res)

            # Numpy always expects outputs to be Numpy arrays
            # And since we have a variable number of outputs
            if len(res) == 1:
                return res[0][0]
            else:
                return tuple(_res[0] for _res in res)

        numpy_vec_func = np.vectorize(
            py_func, signature=gufunc_sign_to_str(self.signature)
        )
        res_variables = numpy_vec_func(*inputs)

        if isinstance(res_variables, tuple):
            for i, out in enumerate(outputs):
                outputs[i][0] = res_variables[i]
        else:
            outputs[0][0] = res_variables
