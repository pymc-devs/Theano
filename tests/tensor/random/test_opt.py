import numpy as np
import pytest

import aesara.tensor as aet
from aesara import config, shared
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.opt import EquilibriumOptimizer
from aesara.graph.optdb import OptimizationQuery
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.basic import (
    dirichlet,
    multinomial,
    multivariate_normal,
    normal,
    poisson,
    uniform,
)
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.type import iscalar, vector


no_mode = Mode("py", OptimizationQuery(include=[], exclude=[]))


def apply_local_opt_to_rv(opt, op_fn, dist_op, dist_params, size, rng):
    dist_params_aet = []
    for p in dist_params:
        p_aet = aet.as_tensor(p).type()
        p_aet.tag.test_value = p
        dist_params_aet.append(p_aet)

    size_aet = []
    for s in size:
        s_aet = iscalar()
        s_aet.tag.test_value = s
        size_aet.append(s_aet)

    dist_st = op_fn(dist_op(*dist_params_aet, size=size_aet, rng=rng))

    f_inputs = [
        p for p in dist_params_aet + size_aet if not isinstance(p, (slice, Constant))
    ]

    mode = Mode("py", EquilibriumOptimizer([opt], max_use_ratio=100))

    f_opt = function(
        f_inputs,
        dist_st,
        mode=mode,
    )

    (new_out,) = f_opt.maker.fgraph.outputs

    return new_out, f_inputs, dist_st, f_opt


def test_inplace_optimization():

    out = normal(0, 1)
    out.owner.inputs[0].default_update = out.owner.outputs[0]

    assert out.owner.op.inplace is False

    f = function(
        [],
        out,
        mode="FAST_RUN",
    )

    (new_out, new_rng) = f.maker.fgraph.outputs
    assert new_out.type == out.type
    assert isinstance(new_out.owner.op, type(out.owner.op))
    assert new_out.owner.op.inplace is True
    assert all(
        np.array_equal(a.data, b.data)
        for a, b in zip(new_out.owner.inputs[1:], out.owner.inputs[1:])
    )


@config.change_flags(compute_test_value="raise")
@pytest.mark.parametrize(
    "dist_op, dist_params, size",
    [
        (
            normal,
            [
                np.array(1.0, dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [],
        ),
        (
            normal,
            [
                np.array([0.0, 1.0], dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [],
        ),
        (
            normal,
            [
                np.array([0.0, 1.0], dtype=config.floatX),
                np.array(5.0, dtype=config.floatX),
            ],
            [3, 2],
        ),
        (
            multivariate_normal,
            [
                np.array([[0], [10], [100]], dtype=config.floatX),
                np.diag(np.array([1e-6], dtype=config.floatX)),
            ],
            [2, 3, 3],
        ),
        (
            dirichlet,
            [np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX)],
            [2, 3, 3],
        ),
        (
            multinomial,
            [
                np.array([10, 20], dtype="int64"),
                np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX),
            ],
            [3, 2],
        ),
    ],
)
def test_local_rv_size_lift(dist_op, dist_params, size):
    rng = shared(np.random.RandomState(1233532), borrow=False)

    new_out, f_inputs, dist_st, f_opt = apply_local_opt_to_rv(
        local_rv_size_lift,
        lambda rv: rv,
        dist_op,
        dist_params,
        size,
        rng,
    )

    assert aet.get_vector_length(new_out.owner.inputs[1]) == 0


@pytest.mark.parametrize(
    "ds_order, lifted, dist_op, dist_params, size, rtol",
    [
        (
            ("x",),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            ("x", "x", "x"),
            True,
            normal,
            (
                np.array(-10.0, dtype=np.float64),
                np.array(1e-6, dtype=np.float64),
            ),
            (),
            1e-7,
        ),
        (
            (1, 0, 2),
            True,
            normal,
            (
                np.arange(2 * 2 * 2).reshape((2, 2, 2)).astype(config.floatX),
                np.array(1e-6).astype(config.floatX),
            ),
            (),
            1e-3,
        ),
        (
            (0, 1, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (0, 2, 1),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (1, 0, 2),
            True,
            normal,
            (np.array(0).astype(config.floatX), np.array(1e-6).astype(config.floatX)),
            (2, 1, 2),
            1e-3,
        ),
        (
            (0, 2, 1),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, 2, 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, "x", 2, "x", 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 0, 2, 1, "x"),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        (
            ("x", 1, 0, 2, "x"),
            False,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
            1e-3,
        ),
        # Only one distribution parameter
        (
            (0, 2, 1),
            True,
            poisson,
            (np.array([[10, 50], [100, 150]], dtype=config.floatX),),
            (3, 2, 2),
            1,
        ),
        # A multi-dimensional case
        (
            (0, 2, 1),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (3, 2),
            1e-3,
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_DimShuffle_lift(ds_order, lifted, dist_op, dist_params, size, rtol):

    rng = shared(np.random.RandomState(1233532), borrow=False)

    new_out, f_inputs, dist_st, f_opt = apply_local_opt_to_rv(
        local_dimshuffle_rv_lift,
        lambda rv: rv.dimshuffle(ds_order),
        dist_op,
        dist_params,
        size,
        rng,
    )

    if lifted:
        assert new_out.owner.op == dist_op
        assert all(
            isinstance(i.owner.op, DimShuffle)
            for i in new_out.owner.inputs[3:]
            if i.owner
        )
    else:
        assert isinstance(new_out.owner.op, DimShuffle)
        return

    f_base = function(
        f_inputs,
        dist_st,
        mode=no_mode,
    )

    arg_values = [p.get_test_value() for p in f_inputs]
    res_base = f_base(*arg_values)
    res_opt = f_opt(*arg_values)

    np.testing.assert_allclose(res_base, res_opt, rtol=rtol)


@pytest.mark.parametrize(
    "indices, lifted, dist_op, dist_params, size",
    [
        (
            # `size`-less advanced boolean indexing
            (np.r_[True, False, False, True],),
            True,
            uniform,
            (
                (0.1 - 1e-5) * np.arange(4).astype(dtype=config.floatX),
                0.1 * np.arange(4).astype(dtype=config.floatX),
            ),
            (),
        ),
        (
            # `size`-only advanced boolean indexing
            (np.r_[True, False, False, True],),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (4,),
        ),
        (
            # `size`-only slice
            (slice(4, -6, -1),),
            True,
            uniform,
            (
                np.array(0.9 - 1e-5, dtype=config.floatX),
                np.array(0.9, dtype=config.floatX),
            ),
            (5, 2),
        ),
        (
            (slice(1, None), [0, 2]),
            True,
            normal,
            (
                np.array([1, 10, 100], dtype=config.floatX),
                np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX),
            ),
            (4, 3),
        ),
        (
            (np.array([1]), 0),
            True,
            normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ),
            (3, 2, 2),
        ),
        # A multi-dimensional case
        (
            (np.array([1]), 0),
            False,
            multivariate_normal,
            (
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.eye(2).astype(config.floatX) * 1e-6,
            ),
            (),
        ),
        # Only one distribution parameter
        (
            (0,),
            True,
            poisson,
            (np.array([[1, 2], [3, 4]], dtype=config.floatX),),
            (3, 2, 2),
        ),
    ],
)
@config.change_flags(compute_test_value_opt="raise", compute_test_value="raise")
def test_Subtensor_lift(indices, lifted, dist_op, dist_params, size):
    from aesara.tensor.subtensor import as_index_constant

    rng = shared(np.random.RandomState(1233532), borrow=False)

    indices_aet = ()
    for i in indices:
        i_aet = as_index_constant(i)
        if not isinstance(i_aet, slice):
            i_aet.tag.test_value = i
        indices_aet += (i_aet,)

    new_out, f_inputs, dist_st, f_opt = apply_local_opt_to_rv(
        local_subtensor_rv_lift,
        lambda rv: rv[indices_aet],
        dist_op,
        dist_params,
        size,
        rng,
    )

    if lifted:
        assert isinstance(new_out.owner.op, RandomVariable)
        assert all(
            isinstance(i.owner.op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor))
            for i in new_out.owner.inputs[3:]
            if i.owner
        )
    else:
        assert isinstance(
            new_out.owner.op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)
        )
        return

    f_base = function(
        f_inputs,
        dist_st,
        mode=no_mode,
    )

    arg_values = [p.get_test_value() for p in f_inputs]
    res_base = f_base(*arg_values)
    res_opt = f_opt(*arg_values)

    np.testing.assert_allclose(res_base, res_opt, rtol=1e-3)


def test_Subtensor_lift_restrictions():
    rng = shared(np.random.RandomState(1233532), borrow=False)

    std = vector("std")
    std.tag.test_value = np.array([1e-5, 2e-5, 3e-5], dtype=config.floatX)
    x = normal(aet.arange(2), aet.ones(2), rng=rng)
    y = x[1]
    # The non-`Subtensor` client depends on the RNG state, so we can't perform
    # the lift
    z = x - y

    fg = FunctionGraph([rng], [z], clone=False)
    _ = EquilibriumOptimizer([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    subtensor_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert subtensor_node == y.owner
    assert isinstance(subtensor_node.op, Subtensor)
    assert subtensor_node.inputs[0].owner.op == normal

    # The non-`Subtensor` client doesn't depend on the RNG state, so we can
    # perform the lift
    z = aet.ones(x.shape) - x[1]

    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumOptimizer([local_subtensor_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner.inputs[0].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, Subtensor)
    assert isinstance(rv_node.inputs[-2].owner.op, Subtensor)


def test_Dimshuffle_lift_restrictions():
    rng = shared(np.random.RandomState(1233532), borrow=False)

    x = normal(aet.arange(2).reshape((2,)), 100, size=(2, 2, 2), rng=rng)
    y = x.dimshuffle(1, 0, 2)
    # The non-`Dimshuffle` client depends on the RNG state, so we can't
    # perform the lift
    z = x - y

    fg = FunctionGraph([rng], [z], clone=False)
    _ = EquilibriumOptimizer([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    dimshuffle_node = fg.outputs[0].owner.inputs[1].owner
    assert dimshuffle_node == y.owner
    assert isinstance(dimshuffle_node.op, DimShuffle)
    assert dimshuffle_node.inputs[0].owner.op == normal

    # The non-`Dimshuffle` client doesn't depend on the RNG state, so we can
    # perform the lift
    z = aet.ones(x.shape) - y

    fg = FunctionGraph([rng], [z], clone=False)
    EquilibriumOptimizer([local_dimshuffle_rv_lift], max_use_ratio=100).apply(fg)

    rv_node = fg.outputs[0].owner.inputs[1].owner
    assert rv_node.op == normal
    assert isinstance(rv_node.inputs[-1].owner.op, DimShuffle)
    assert isinstance(rv_node.inputs[-2].owner.op, DimShuffle)
