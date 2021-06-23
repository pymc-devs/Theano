from collections.abc import Sequence
from copy import copy

import numpy as np

import aesara
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Variable
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt_utils import optimize_graph
from aesara.misc.safe_asarray import _asarray
from aesara.tensor.basic import as_tensor_variable, constant, get_vector_length
from aesara.tensor.basic_opt import ShapeFeature, topo_constant_folding
from aesara.tensor.random.type import RandomStateType
from aesara.tensor.random.utils import normalize_size_param, params_broadcast_shapes
from aesara.tensor.type import TensorType, all_dtypes


def default_shape_from_params(
    ndim_supp, dist_params, rep_param_idx=0, param_shapes=None
):
    """Infer the dimensions for the output of a `RandomVariable`.

    This is a function that derives a random variable's support
    shape/dimensions from one of its parameters.

    XXX: It's not always possible to determine a random variable's support
    shape from its parameters, so this function has fundamentally limited
    applicability and must be replaced by custom logic in such cases.

    XXX: This function is not expected to handle `ndim_supp = 0` (i.e.
    scalars), since that is already definitively handled in the `Op` that
    calls this.

    TODO: Consider using `aesara.compile.ops.shape_i` alongside `ShapeFeature`.

    Parameters
    ----------
    ndim_supp: int
        Total number of dimensions for a single draw of the random variable
        (e.g. a multivariate normal draw is 1D, so `ndim_supp = 1`).
    dist_params: list of `aesara.graph.basic.Variable`
        The distribution parameters.
    param_shapes: list of tuple of `ScalarVariable` (optional)
        Symbolic shapes for each distribution parameter.  These will
        be used in place of distribution parameter-generated shapes.
    rep_param_idx: int (optional)
        The index of the distribution parameter to use as a reference
        In other words, a parameter in `dist_param` with a shape corresponding
        to the support's shape.
        The default is the first parameter (i.e. the value 0).

    Results
    -------
    out: a tuple representing the support shape for a distribution with the
    given `dist_params`.

    """
    if ndim_supp <= 0:
        raise ValueError("ndim_supp must be greater than 0")
    if param_shapes is not None:
        ref_param = param_shapes[rep_param_idx]
        return (ref_param[-ndim_supp],)
    else:
        ref_param = dist_params[rep_param_idx]
        if ref_param.ndim < ndim_supp:
            raise ValueError(
                (
                    "Reference parameter does not match the "
                    f"expected dimensions; {ref_param} has less than {ndim_supp} dim(s)."
                )
            )
        return ref_param.shape[-ndim_supp:]


class RandomVariable(Op):
    """An `Op` that produces a sample from a random variable.

    This is essentially `RandomFunction`, except that it removes the
    `outtype` dependency and handles shape dimension information more
    directly.

    """

    __props__ = ("name", "ndim_supp", "ndims_params", "dtype", "inplace")
    default_output = 1

    def __init__(
        self,
        name=None,
        ndim_supp=None,
        ndims_params=None,
        dtype=None,
        inplace=None,
    ):
        """Create a random variable `Op`.

        Parameters
        ----------
        name: str
            The `Op`'s display name.
        ndim_supp: int
            Total number of dimensions for a single draw of the random variable
            (e.g. a multivariate normal draw is 1D, so ``ndim_supp = 1``).
        ndims_params: list of int
            Number of dimensions for each distribution parameter when the
            parameters only specify a single drawn of the random variable
            (e.g. a multivariate normal's mean is 1D and covariance is 2D, so
            ``ndims_params = [1, 2]``).
        dtype: str (optional)
            The dtype of the sampled output.  If the value ``"floatX"`` is
            given, then ``dtype`` is set to ``aesara.config.floatX``.  If
            ``None`` (the default), the `dtype` keyword must be set when
            `RandomVariable.make_node` is called.
        inplace: boolean (optional)
            Determine whether or not the underlying rng state is updated
            in-place or not (i.e. copied).

        """
        super().__init__()

        self.name = name or getattr(self, "name")
        self.ndim_supp = (
            ndim_supp if ndim_supp is not None else getattr(self, "ndim_supp")
        )
        self.ndims_params = (
            ndims_params if ndims_params is not None else getattr(self, "ndims_params")
        )
        self.dtype = dtype or getattr(self, "dtype", None)

        self.inplace = (
            inplace if inplace is not None else getattr(self, "inplace", False)
        )

        if not isinstance(self.ndims_params, Sequence):
            raise TypeError("Parameter ndims_params must be sequence type.")

        self.ndims_params = tuple(self.ndims_params)

        if self.inplace:
            self.destroy_map = {0: [0]}

    def _shape_from_params(self, dist_params, **kwargs):
        """Determine the shape of a `RandomVariable`'s output given its parameters.

        This does *not* consider the extra dimensions added by the `size` parameter.

        Defaults to `param_supp_shape_fn`.
        """
        return default_shape_from_params(self.ndim_supp, dist_params, **kwargs)

    def rng_fn(self, rng, *args, **kwargs):
        """Sample a numeric random variate."""
        return getattr(np.random.RandomState, self.name)(rng, *args, **kwargs)

    def __str__(self):
        props_str = ", ".join((f"{getattr(self, prop)}" for prop in self.__props__[1:]))
        return f"{self.name}_rv{{{props_str}}}"

    def _infer_shape(self, size, dist_params, param_shapes=None):
        """Compute the output shape given the size and distribution parameters.

        Parameters
        ----------
        size : TensorVariable
            The size parameter specified for this `RandomVariable`.
        dist_params : list of TensorVariable
            The symbolic parameter for this `RandomVariable`'s distribution.
        param_shapes : list of tuples of TensorVariable (optional)
            The shapes of the `dist_params` as given by `ShapeFeature`'s
            via `Op.infer_shape`'s `input_shapes` argument.  This parameter's
            values are essentially more accurate versions of ``[d.shape for d
            in dist_params]``.

        Outputs
        -------
        shape : tuple of `ScalarVariable`

        """

        size_len = get_vector_length(size)

        if size_len > 0:
            if self.ndim_supp == 0:
                return size
            else:
                supp_shape = self._shape_from_params(
                    dist_params, param_shapes=param_shapes
                )
                return tuple(size) + tuple(supp_shape)

        # Broadcast the parameters
        param_shapes = params_broadcast_shapes(
            param_shapes or [p.shape for p in dist_params], self.ndims_params
        )

        def slice_ind_dims(p, ps, n):
            shape = tuple(ps)

            if n == 0:
                return (p, shape)

            ind_slice = (slice(None),) * (p.ndim - n) + (0,) * n
            ind_shape = [
                s if b is False else constant(1, "int64")
                for s, b in zip(shape[:-n], p.broadcastable[:-n])
            ]
            return (
                p[ind_slice],
                ind_shape,
            )

        # These are versions of our actual parameters with the anticipated
        # dimensions (i.e. support dimensions) removed so that only the
        # independent variate dimensions are left.
        params_ind_slice = tuple(
            slice_ind_dims(p, ps, n)
            for p, ps, n in zip(dist_params, param_shapes, self.ndims_params)
        )

        if len(params_ind_slice) == 1:
            ind_param, ind_shape = params_ind_slice[0]
            ndim_ind = len(ind_shape)
            shape_ind = ind_shape
        elif len(params_ind_slice) > 1:
            # If there are multiple parameters, the dimensions of their
            # independent variates should broadcast together.
            p_slices, p_shapes = zip(*params_ind_slice)

            shape_ind = aesara.tensor.extra_ops.broadcast_shape_iter(
                p_shapes, arrays_are_shapes=True
            )

            ndim_ind = len(shape_ind)
        else:
            ndim_ind = 0

        if self.ndim_supp == 0:
            shape_supp = tuple()
            shape_reps = tuple(size)

            if ndim_ind > 0:
                shape_reps = shape_reps[:-ndim_ind]

            ndim_reps = len(shape_reps)
        else:
            shape_supp = self._shape_from_params(
                dist_params,
                param_shapes=param_shapes,
            )

            ndim_reps = size_len
            shape_reps = size

        ndim_shape = self.ndim_supp + ndim_ind + ndim_reps

        if ndim_shape == 0:
            shape = constant([], dtype="int64")
        else:
            shape = tuple(shape_reps) + tuple(shape_ind) + tuple(shape_supp)

        # if shape is None:
        #     raise ShapeError()

        return shape

    @config.change_flags(compute_test_value="off")
    def compute_bcast(self, dist_params, size):
        """Compute the broadcast array for this distribution's `TensorType`.

        Parameters
        ----------
        dist_params: list
            Distribution parameters.
        size: int or Sequence (optional)
            Numpy-like size of the output (i.e. replications).

        """
        shape = self._infer_shape(size, dist_params)

        shape_fg = FunctionGraph(
            outputs=tuple(shape), features=[ShapeFeature()], clone=True
        )
        folded_shape = optimize_graph(
            shape_fg, custom_opt=topo_constant_folding
        ).outputs

        return [getattr(s, "data", s) == 1 for s in folded_shape]

    def infer_shape(self, fgraph, node, input_shapes):
        _, size, _, *dist_params = node.inputs
        _, _, _, *param_shapes = input_shapes

        shape = self._infer_shape(size, dist_params, param_shapes=param_shapes)

        return [None, [s for s in shape]]

    def __call__(self, *args, size=None, name=None, rng=None, dtype=None, **kwargs):
        res = super().__call__(rng, size, dtype, *args, **kwargs)

        if name is not None:
            res.name = name

        return res

    def make_node(self, rng, size, dtype, *dist_params):
        """Create a random variable node.

        XXX: Unnamed/non-keyword arguments are considered distribution
        parameters!  If you want to set `size`, `rng`, and/or `name`, use their
        keywords.

        Parameters
        ----------
        rng: RandomStateType
            Existing Aesara ``RandomState`` object to be used.  Creates a new
            one, if ``None``.
        size: int or Sequence
            NumPy-like size parameter.
        dtype: str
            The dtype of the sampled output.  If the value ``"floatX"`` is
            given, then `dtype` is set to ``aesara.config.floatX``.  This value is
            only used when ``self.dtype`` isn't set.
        dist_params: list
            Distribution parameters.

        Results
        -------
        out: Apply
            A node with inputs ``(rng, size, dtype) + dist_args`` and outputs
            ``(rng_var, out_var)``.

        """
        size = normalize_size_param(size)

        dist_params = tuple(
            as_tensor_variable(p) if not isinstance(p, Variable) else p
            for p in dist_params
        )

        if rng is None:
            rng = aesara.shared(np.random.RandomState())
        elif not isinstance(rng.type, RandomStateType):
            raise TypeError("The type of rng should be an instance of RandomStateType")

        bcast = self.compute_bcast(dist_params, size)
        dtype = self.dtype or dtype

        if dtype == "floatX":
            dtype = config.floatX
        elif dtype is None or (isinstance(dtype, str) and dtype not in all_dtypes):
            raise TypeError("dtype is unspecified")

        if isinstance(dtype, str):
            dtype_idx = constant(all_dtypes.index(dtype), dtype="int64")
        else:
            dtype_idx = constant(dtype, dtype="int64")
            dtype = all_dtypes[dtype_idx.data]

        outtype = TensorType(dtype=dtype, broadcastable=bcast)
        out_var = outtype()
        inputs = (rng, size, dtype_idx) + dist_params
        outputs = (rng.type(), out_var)

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        rng_var_out, smpl_out = outputs

        rng, size, dtype, *args = inputs

        out_var = node.outputs[1]

        # If `size == []`, that means no size is enforced, and NumPy is trusted
        # to draw the appropriate number of samples, NumPy uses `size=None` to
        # represent that.  Otherwise, NumPy expects a tuple.
        if np.size(size) == 0:
            size = None
        else:
            size = tuple(size)

        # Draw from `rng` if `self.inplace` is `True`, and from a copy of `rng`
        # otherwise.
        if not self.inplace:
            rng = copy(rng)

        rng_var_out[0] = rng

        smpl_val = self.rng_fn(rng, *(args + [size]))

        if (
            not isinstance(smpl_val, np.ndarray)
            or str(smpl_val.dtype) != out_var.type.dtype
        ):
            smpl_val = _asarray(smpl_val, dtype=out_var.type.dtype)

        smpl_out[0] = smpl_val

    def grad(self, inputs, outputs):
        return [
            aesara.gradient.grad_undefined(
                self, k, inp, "No gradient defined for random variables"
            )
            for k, inp in enumerate(inputs)
        ]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]
