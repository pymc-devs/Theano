from aesara.compile import optdb
from aesara.configdefaults import config
from aesara.graph.op import compute_test_value
from aesara.graph.opt import in2out, local_optimizer
from aesara.tensor.basic import constant, get_vector_length
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.extra_ops import broadcast_to
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.utils import broadcast_params
from aesara.tensor.shape import Shape, Shape_i
from aesara.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    as_index_variable,
    get_idx_list,
    indexed_result_shape,
)


def is_rv_used_in_graph(base_rv, node, fgraph):
    """Determine whether or not `base_rv` is used by a node other than `node` in `fgraph`.

    If a node uses `Shape` or `Shape_i` on the `base_rv`, we ignore it, because
    those `Op`s don't rely on the actual sample values of `base_rv`.

    TODO: We should apply all the shape rewrites before these rewrites, since
    that would properly remove the unnecessary dependencies on `base_rv` (when
    possible).

    """

    def _node_check(n, i):
        if n == "output":
            n = fgraph.outputs[i].owner
        return n == node or isinstance(n.op, (Shape, Shape_i))

    return not all(_node_check(n, i) for n, i in fgraph.clients.get(base_rv, ()))


@local_optimizer([RandomVariable], inplace=True)
def random_make_inplace(fgraph, node):
    op = node.op

    if isinstance(op, RandomVariable) and not op.inplace:
        name, ndim_supp, ndims_params, dtype, _ = op._props()
        new_op = type(op)(name, ndim_supp, ndims_params, dtype, True)
        return new_op.make_node(*node.inputs).outputs

    return False


optdb.register(
    "random_make_inplace",
    in2out(random_make_inplace, ignore_newtrees=True),
    99,
    "fast_run",
    "inplace",
)


@local_optimizer(tracks=None)
def local_rv_size_lift(fgraph, node):
    """Lift the ``size`` parameter in a ``RandomVariable``.

    In other words, this will broadcast the distribution parameters by adding
    the extra dimensions implied by the ``size`` parameter, and remove the
    ``size`` parameter in the process.

    For example, ``normal(0, 1, size=(1, 2))`` becomes
    ``normal([[0, 0]], [[1, 1]], size=())``.

    """

    if not isinstance(node.op, RandomVariable):
        return

    rng, size, dtype, *dist_params = node.inputs

    dist_params = broadcast_params(dist_params, node.op.ndims_params)

    if get_vector_length(size) > 0:
        dist_params = [
            broadcast_to(
                p, (tuple(size) + tuple(p.shape)) if node.op.ndim_supp > 0 else size
            )
            for p in dist_params
        ]
    else:
        return

    new_node = node.op.make_node(rng, None, dtype, *dist_params)

    if config.compute_test_value != "off":
        compute_test_value(new_node)

    return new_node.outputs


@local_optimizer([DimShuffle])
def local_dimshuffle_rv_lift(fgraph, node):
    """Lift a ``DimShuffle`` through ``RandomVariable`` inputs.

    For example, ``normal(mu, std).T == normal(mu.T, std.T)``.

    The basic idea behind this rewrite is that we need to separate the
    ``DimShuffle``-ing into distinct ``DimShuffle``s that each occur in two
    distinct sub-spaces: the (set of independent) parameters and ``size``
    (i.e. replications) sub-spaces.

    If a ``DimShuffle`` exchanges dimensions across those two sub-spaces, then we
    don't do anything.

    Otherwise, if the ``DimShuffle`` only exchanges dimensions within each of
    those sub-spaces, we can break it apart and apply the parameter-space
    ``DimShuffle`` to the distribution parameters, and then apply the
    replications-space ``DimShuffle`` to the ``size`` tuple.  The latter is a
    particularly simple rearranging of a tuple, but the former requires a
    little more work.

    TODO: Currently, multivariate support for this rewrite is disabled.

    """

    ds_op = node.op

    if not isinstance(ds_op, DimShuffle):
        return False

    base_rv = node.inputs[0]
    rv_node = base_rv.owner

    if not (
        rv_node and isinstance(rv_node.op, RandomVariable) and rv_node.op.ndim_supp == 0
    ):
        return False

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(base_rv, node, fgraph):
        return False

    rv_op = rv_node.op
    rng, size, dtype, *dist_params = rv_node.inputs

    # We need to know the dimensions that were *not* added by the `size`
    # parameter (i.e. the dimensions corresponding to independent variates with
    # different parameter values)
    num_ind_dims = None
    if len(dist_params) == 1:
        num_ind_dims = dist_params[0].ndim
    else:
        # When there is more than one distribution parameter, assume that all
        # of them will broadcast to the maximum number of dimensions
        num_ind_dims = max(d.ndim for d in dist_params)

    # If the indices in `ds_new_order` are entirely within the replication
    # indices group or the independent variates indices group, then we can apply
    # this rewrite.

    ds_new_order = ds_op.new_order
    # Create a map from old index order to new/`DimShuffled` index order
    dim_orders = [(n, d) for n, d in enumerate(ds_new_order) if isinstance(d, int)]

    # Find the index at which the replications/independents split occurs
    reps_ind_split_idx = len(dim_orders) - (num_ind_dims + rv_op.ndim_supp)

    ds_reps_new_dims = dim_orders[:reps_ind_split_idx]
    ds_ind_new_dims = dim_orders[reps_ind_split_idx:]
    ds_in_ind_space = ds_ind_new_dims and all(
        d >= reps_ind_split_idx for n, d in ds_ind_new_dims
    )

    if ds_in_ind_space or (not ds_ind_new_dims and not ds_reps_new_dims):

        # Update the `size` array to reflect the `DimShuffle`d dimensions,
        # since the trailing dimensions in `size` represent the independent
        # variates dimensions (for univariate distributions, at least)
        new_size = (
            [constant(1, dtype="int64") if o == "x" else size[o] for o in ds_new_order]
            if get_vector_length(size) > 0
            else size
        )

        # Compute the new axes parameter(s) for the `DimShuffle` that will be
        # applied to the `RandomVariable` parameters (they need to be offset)
        if ds_ind_new_dims:
            rv_params_new_order = [
                d - reps_ind_split_idx if isinstance(d, int) else d
                for d in ds_new_order[ds_ind_new_dims[0][0] :]
            ]
        else:
            # This case is reached when, for example, `ds_new_order` only
            # consists of new broadcastable dimensions (i.e. `"x"`s)
            rv_params_new_order = ds_new_order

        # Lift the `DimShuffle`s into the parameters
        # NOTE: The parameters might not be broadcasted against each other, so
        # we can only apply the parts of the `DimShuffle` that are relevant.
        new_dist_params = []
        for d in dist_params:
            if d.ndim < len(ds_ind_new_dims):
                _rv_params_new_order = [
                    o
                    for o in rv_params_new_order
                    if (isinstance(o, int) and o < d.ndim) or o == "x"
                ]
            else:
                _rv_params_new_order = rv_params_new_order

            new_dist_params.append(
                type(ds_op)(d.type.broadcastable, _rv_params_new_order)(d)
            )
        new_node = rv_op.make_node(rng, new_size, dtype, *new_dist_params)

        if config.compute_test_value != "off":
            compute_test_value(new_node)

        out = new_node.outputs[1]
        if base_rv.name:
            out.name = f"{base_rv.name}_lifted"
        return [out]

    ds_in_reps_space = ds_reps_new_dims and all(
        d < reps_ind_split_idx for n, d in ds_reps_new_dims
    )

    if ds_in_reps_space:
        # Update the `size` array to reflect the `DimShuffle`d dimensions.
        # There should be no need to `DimShuffle` now.
        new_size = [
            constant(1, dtype="int64") if o == "x" else size[o] for o in ds_new_order
        ]

        new_node = rv_op.make_node(rng, new_size, dtype, *dist_params)

        if config.compute_test_value != "off":
            compute_test_value(new_node)

        out = new_node.outputs[1]
        if base_rv.name:
            out.name = f"{base_rv.name}_lifted"
        return [out]

    return False


@local_optimizer([Subtensor, AdvancedSubtensor1, AdvancedSubtensor])
def local_subtensor_rv_lift(fgraph, node):
    """Lift a ``*Subtensor`` through ``RandomVariable`` inputs.

    In a fashion similar to ``local_dimshuffle_rv_lift``, the indexed dimensions
    need to be separated into distinct replication-space and (independent)
    parameter-space ``*Subtensor``s.

    The replication-space ``*Subtensor`` can be used to determine a
    sub/super-set of the replication-space and, thus, a "smaller"/"larger"
    ``size`` tuple.  The parameter-space ``*Subtensor`` is simply lifted and
    applied to the distribution parameters.

    Consider the following example graph:
    ``normal(mu, std, size=(d1, d2, d3))[idx1, idx2, idx3]``.  The
    ``*Subtensor`` ``Op`` requests indices ``idx1``, ``idx2``, and ``idx3``,
    which correspond to all three ``size`` dimensions.  Now, depending on the
    broadcasted dimensions of ``mu`` and ``std``, this ``*Subtensor`` ``Op``
    could be reducing the ``size`` parameter and/or sub-setting the independent
    ``mu`` and ``std`` parameters.  Only once the dimensions are properly
    separated into the two replication/parameter subspaces can we determine how
    the ``*Subtensor`` indices are distributed.
    For instance, ``normal(mu, std, size=(d1, d2, d3))[idx1, idx2, idx3]``
    could become
    ``normal(mu[idx1], std[idx2], size=np.shape(idx1) + np.shape(idx2) + np.shape(idx3))``
    if ``mu.shape == std.shape == ()``

    ``normal`` is a rather simple case, because it's univariate.  Multivariate
    cases require a mapping between the parameter space and the image of the
    random variable.  This may not always be possible, but for many common
    distributions it is.  For example, the dimensions of the multivariate
    normal's image can be mapped directly to each dimension of its parameters.
    We use these mappings to change a graph like ``multivariate_normal(mu, Sigma)[idx1]``
    into ``multivariate_normal(mu[idx1], Sigma[idx1, idx1])``.

    """

    st_op = node.op

    if not isinstance(st_op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)):
        return False

    base_rv = node.inputs[0]

    rv_node = base_rv.owner
    if not (rv_node and isinstance(rv_node.op, RandomVariable)):
        return False

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(base_rv, node, fgraph):
        return False

    rv_op = rv_node.op
    rng, size, dtype, *dist_params = rv_node.inputs

    # TODO: Remove this once the multi-dimensional changes described below are
    # in place.
    if rv_op.ndim_supp > 0:
        return False

    rv_op = base_rv.owner.op
    rng, size, dtype, *dist_params = base_rv.owner.inputs

    idx_list = getattr(st_op, "idx_list", None)
    if idx_list:
        cdata = get_idx_list(node.inputs, idx_list)
    else:
        cdata = node.inputs[1:]

    st_indices, st_is_bool = zip(
        *tuple(
            (as_index_variable(i), getattr(i, "dtype", None) == "bool") for i in cdata
        )
    )

    # We need to separate dimensions into replications and independents
    num_ind_dims = None
    if len(dist_params) == 1:
        num_ind_dims = dist_params[0].ndim
    else:
        # When there is more than one distribution parameter, assume that all
        # of them will broadcast to the maximum number of dimensions
        num_ind_dims = max(d.ndim for d in dist_params)

    reps_ind_split_idx = base_rv.ndim - (num_ind_dims + rv_op.ndim_supp)

    if len(st_indices) > reps_ind_split_idx:
        # These are the indices that need to be applied to the parameters
        ind_indices = tuple(st_indices[reps_ind_split_idx:])

        # We need to broadcast the parameters before applying the `*Subtensor*`
        # with these indices, because the indices could be referencing broadcast
        # dimensions that don't exist (yet)
        bcast_dist_params = broadcast_params(dist_params, rv_op.ndims_params)

        # TODO: For multidimensional distributions, we need a map that tells us
        # which dimensions of the parameters need to be indexed.
        #
        # For example, `multivariate_normal` would have the following:
        # `RandomVariable.param_to_image_dims = ((0,), (0, 1))`
        #
        # I.e. the first parameter's (i.e. mean's) first dimension maps directly to
        # the dimension of the RV's image, and its second parameter's
        # (i.e. covariance's) first and second dimensions map directly to the
        # dimension of the RV's image.

        args_lifted = tuple(p[ind_indices] for p in bcast_dist_params)
    else:
        # In this case, no indexing is applied to the parameters; only the
        # `size` parameter is affected.
        args_lifted = dist_params

    # TODO: Could use `ShapeFeature` info.  We would need to be sure that
    # `node` isn't in the results, though.
    # if hasattr(fgraph, "shape_feature"):
    #     output_shape = fgraph.shape_feature.shape_of(node.outputs[0])
    # else:
    output_shape = indexed_result_shape(base_rv.shape, st_indices)

    size_lifted = (
        output_shape if rv_op.ndim_supp == 0 else output_shape[: -rv_op.ndim_supp]
    )

    # Boolean indices can actually change the `size` value (compared to just
    # *which* dimensions of `size` are used).
    if any(st_is_bool):
        size_lifted = tuple(
            aet_sum(idx) if is_bool else s
            for s, is_bool, idx in zip(
                size_lifted, st_is_bool, st_indices[: (reps_ind_split_idx + 1)]
            )
        )

    new_node = rv_op.make_node(rng, size_lifted, dtype, *args_lifted)
    _, new_rv = new_node.outputs

    # Calling `Op.make_node` directly circumvents test value computations, so
    # we need to compute the test values manually
    if config.compute_test_value != "off":
        compute_test_value(new_node)

    return [new_rv]
