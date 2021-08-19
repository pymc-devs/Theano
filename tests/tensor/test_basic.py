import itertools
import warnings
from copy import copy, deepcopy
from functools import partial
from tempfile import mkstemp

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import aesara
import aesara.scalar as aes
import aesara.tensor.basic as aet
import aesara.tensor.math as tm
from aesara import compile, config, function, shared
from aesara.assert_op import Assert
from aesara.compile.io import In, Out
from aesara.compile.mode import get_default_mode
from aesara.compile.ops import DeepCopyOp
from aesara.gradient import grad, hessian
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.misc.safe_asarray import _asarray
from aesara.scalar import autocast_float, autocast_float_as
from aesara.tensor.basic import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ARange,
    Choose,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    PermuteRowElements,
    Rebroadcast,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    Tile,
    Tri,
    addbroadcast,
    alloc,
    arange,
    as_tensor_variable,
    atleast_Nd,
    cast,
    choose,
    constant,
    default,
    diag,
    expand_dims,
    extract_constant,
    eye,
    fill,
    flatnonzero,
    flatten,
    full_like,
    get_scalar_constant_value,
    get_vector_length,
    horizontal_stack,
    inverse_permutation,
    join,
    make_vector,
    mgrid,
    nonzero,
    nonzero_values,
    ogrid,
    ones_like,
    patternbroadcast,
    permute_row_elements,
    roll,
    scalar_from_tensor,
    second,
    stack,
    stacklists,
    swapaxes,
    switch,
    tensor_copy,
    tensor_from_scalar,
    tile,
    tri,
    tril,
    triu,
    unbroadcast,
    vertical_stack,
    zeros_like,
)
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.exceptions import EmptyConstantError, NotScalarConstantError
from aesara.tensor.math import dense_dot, eq
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.shape import Reshape, Shape, Shape_i, shape_padright
from aesara.tensor.type import (
    TensorType,
    bvector,
    col,
    dmatrix,
    dscalar,
    dscalars,
    dtensor3,
    dtensor4,
    dvector,
    fmatrix,
    fscalar,
    fscalars,
    fvector,
    imatrix,
    int_dtypes,
    iscalar,
    iscalars,
    itensor3,
    ivector,
    lscalar,
    lvector,
    matrices,
    matrix,
    row,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor4,
    vector,
    vectors,
    wvector,
)
from aesara.tensor.var import TensorConstant
from aesara.utils import PYTHON_INT_BITWIDTH
from tests import unittest_tools as utt
from tests.tensor.utils import (
    ALL_DTYPES,
    COMPLEX_DTYPES,
    REAL_DTYPES,
    _good_broadcast_unary_normal,
    _grad_broadcast_unary_normal,
    eval_outputs,
    get_numeric_types,
    inplace_func,
    integers,
    integers_ranged,
    makeBroadcastTester,
    makeTester,
    multi_dtype_cast_checks,
    multi_dtype_checks,
    random,
    random_of_dtype,
)


if config.mode == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
else:
    mode_opt = get_default_mode()

TestSwitchBroadcast = makeBroadcastTester(
    op=switch,
    expected=np.where,
    good=dict(
        all_true=(np.asarray(1, dtype=config.floatX), random(4, 5), random(4, 5)),
        false_true=(np.asarray(0, dtype=config.floatX), random(4, 5), random(4, 5)),
        mixed=(integers_ranged(0, 1, (4, 5)), random(4, 5), random(4, 5)),
    ),
    bad_build=dict(all_true=(np.asarray(1, dtype=config.floatX), random(4, 5))),
    bad_runtime=dict(
        all_true=(np.asarray(1, dtype=config.floatX), random(3, 5), random(4, 5)),
        false_true=(np.asarray(0, dtype=config.floatX), random(4, 6), random(4, 5)),
    ),
    # We suppose that cond+eps do not switch branch in switch.grad()
    # So we can't call verify_grad with cond 0.
    grad=dict(
        all_true=(np.asarray(1, dtype=config.floatX), random(4, 5), random(4, 5)),
        # false_true=(np.asarray(0, dtype=config.floatX),
        #             random(4, 5), random(4, 5)),
        # mixed=(integers_ranged(0, 1, (4, 5)).astype(config.floatX),
        #        random(4, 5), random(4, 5))
    ),
)


def _numpy_second(x, y):
    return np.broadcast_arrays(x, y)[1]


TestSecondBroadcast = makeTester(
    name="SecondBroadcastTester",
    op=second,
    expected=_numpy_second,
    good=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (5,)),
            multi_dtype_checks((2, 3, 2), (3, 2)),
            multi_dtype_checks((2, 3, 2), (2,)),
        )
    ),
    # I can't think of any way to make this fail at build time
    # Just some simple smoke tests
    bad_runtime=dict(
        fail1=(random(5, 4), random(5)),
        fail2=(random(3, 2, 3), random(6, 9)),
        fail3=(integers(6, 2, 9), random(3, 2)),
    ),
)

# We exclude local_fill_to_alloc because it optimizes the "second" node
# away from the graph.
TestSecondSameRank = makeTester(
    name="SecondSameRankTester",
    op=second,
    expected=_numpy_second,
    good=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (4, 5)),
            multi_dtype_checks((1, 2), (3, 2)),
            multi_dtype_checks((3, 2), (1, 2)),
        )
    ),
    # These sizes are not broadcastable to one another
    # and SHOULD raise an error, but currently don't.
    bad_runtime=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (5, 4)),
            multi_dtype_checks((1, 5), (5, 4)),
        )
    ),
    mode=get_default_mode().excluding("local_fill_to_alloc", "local_useless_fill"),
)

# Alloc
TestAllocBroadcast = makeBroadcastTester(
    name="AllocTester",
    op=alloc,
    expected=(lambda x, *shp: np.zeros(shp, dtype=x.dtype) + x),
    good=dict(
        correct01=(random(), np.int32(7)),
        correct01_bcast=(random(1), np.int32(7)),
        correct02=(random(), np.int32(4), np.int32(7)),
        correct12=(random(7), np.int32(4), np.int32(7)),
        correct13=(random(7), np.int32(2), np.int32(4), np.int32(7)),
        correct23=(random(4, 7), np.int32(2), np.int32(4), np.int32(7)),
        correctb1=(random(1, 7), np.int32(4), np.int32(7)),
        correctb2=(random(1, 7), np.int32(2), np.int32(4), np.int32(7)),
        correctb3=(random(7, 1), np.int32(7), np.int32(4)),
        correctb4=(random(7, 1), np.int32(2), np.int32(7), np.int32(4)),
    ),
    bad_runtime=dict(
        bad_shape12=(random(7), np.int32(7), np.int32(5)),
    ),
    bad_build=dict(
        vec=(random(1), [np.int32(2)]),
        too_big32=(random(6, 2, 4), np.int32(6), np.int32(2)),
        too_big32b=(random(6, 2, 4), np.int32(6), np.int32(4)),
        too_big32c=(random(6, 2, 4), np.int32(2), np.int32(4)),
        too_big32d=(random(6, 2, 4), np.int32(2), np.int32(6)),
        too_big32e=(random(6, 2, 4), np.int32(4), np.int32(6)),
        too_big32f=(random(6, 2, 4), np.int32(4), np.int32(2)),
    ),
)

# Since not all inputs of Alloc are differentiable, we need different testers
s1, s2, s3 = integers_ranged(1, 13, (3,))
# alloc a scalar into a vector
TestAlloc01GradBroadcast = makeBroadcastTester(
    name="Alloc01GradTester",
    op=(lambda x: alloc(x, s1)),
    expected=(lambda x: np.zeros((s1,), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(),),
        x2=(random(),),
        x3=(random(),),
    ),
)

# alloc a vector into a tensor3
TestAlloc13GradBroadcast = makeBroadcastTester(
    name="Alloc13GradTester",
    op=(lambda x: alloc(x, s1, s2, s3)),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s3),),
        x2=(random(s3),),
        x3=(random(s3),),
    ),
)

# unbroadcast a row to a matrix
TestAllocb1GradBroadcast = makeBroadcastTester(
    name="Allocb1GradTester",
    op=lambda x: alloc(x, s1, s2),
    expected=(lambda x: np.zeros((s1, s2), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(1, s2),),
        x2=(random(1, s2),),
        x3=(random(1, s2),),
    ),
)

# unbroadcast a row to a tensor3
TestAllocb2GradBroadcast = makeBroadcastTester(
    name="Allocb2GradTester",
    op=lambda x: alloc(x, s1, s2, s3),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(1, s3),),
        x2=(random(1, s3),),
        x3=(random(1, s3),),
    ),
)

# unbroadcast a col to a matrix
TestAllocb3GradBroadcast = makeBroadcastTester(
    name="Allocb3GradTester",
    op=lambda x: alloc(x, s1, s2),
    expected=(lambda x: np.zeros((s1, s2), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s1, 1),),
        x2=(random(s1, 1),),
        x3=(random(s1, 1),),
    ),
)

# unbroadcast a col to a tensor3
TestAllocb4GradBroadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x, s1, s2, s3),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s2, 1),),
        x2=(random(s2, 1),),
        x3=(random(s2, 1),),
    ),
)


# Partial un broadcast of a dimshuffled input
TestAllocDimshuffleGradBroadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x.dimshuffle("x", "x", 0), 1, s2, s3),
    expected=(lambda x: np.zeros((1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s3),),
        x2=(random(s3),),
        x3=(random(s3),),
    ),
)
TestAllocDimshuffleGrad2Broadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x.dimshuffle("x", 0), 1, s2, s3),
    expected=(lambda x: np.zeros((1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s3),),
        x2=(random(s3),),
        x3=(random(s3),),
    ),
)

TestZerosLikeBroadcast = makeBroadcastTester(
    op=zeros_like,
    expected=np.zeros_like,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    name="ZerosLike",
)

TestOnesLikeBroadcast = makeBroadcastTester(
    op=ones_like,
    expected=np.ones_like,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    name="OnesLike",
)


class ApplyDefaultTestOp(Op):
    def __init__(self, id):
        self.default_output = id

    def make_node(self, x):
        x = aet.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


def test_constant():
    int8_vector_type = TensorType(dtype="int8", broadcastable=(False,))

    # Make sure we return a `TensorConstant` unchanged
    x = TensorConstant(int8_vector_type, [1, 2])
    y = constant(x)
    assert y is x

    # Make sure we can add and remove broadcastable dimensions
    int8_scalar_type = TensorType(dtype="int8", broadcastable=())
    x_data = np.array(2, dtype="int8")

    x = TensorConstant(int8_scalar_type, x_data)
    y = constant(x, ndim=1)
    assert y.ndim == 1
    assert np.array_equal(y.data, np.expand_dims(x_data, 0))

    y = constant(x, ndim=2)
    assert y.ndim == 2
    assert np.array_equal(y.data, np.expand_dims(x_data, (0, 1)))

    z = constant(y, ndim=0)
    assert y.ndim == 2 and z.ndim == 0
    assert np.array_equal(z.data, x_data)


class TestAsTensorVariable:
    """
    Unit test for ensuring that as_tensor_variable handles Apply objects
    correctly and removes leading broadcastable dimensions when possible.
    """

    def setup_method(self):
        self.x = scalar("x")

    def test_tensor_from_scalar(self):
        y = as_tensor_variable(aes.int8())
        assert isinstance(y.owner.op, TensorFromScalar)

    def test_multi_outputs(self):
        good_apply_var = ApplyDefaultTestOp(0).make_node(self.x)
        as_tensor_variable(good_apply_var)

        bad_apply_var = ApplyDefaultTestOp(-1).make_node(self.x)
        with pytest.raises(ValueError):
            _ = as_tensor_variable(bad_apply_var)

        bad_apply_var = ApplyDefaultTestOp(2).make_node(self.x)
        with pytest.raises(ValueError):
            _ = as_tensor_variable(bad_apply_var)

    def test_list(self):
        # Make sure our exception handling during `Sequence` processing doesn't
        # mask exceptions caused by unrelated logic (e.g.  computing test
        # values)
        with config.change_flags(compute_test_value="raise"), pytest.raises(ValueError):
            a = lscalar("a")
            y = (a, a, 1)
            _ = as_tensor_variable(y)

        bad_apply_var = ApplyDefaultTestOp([0, 1]).make_node(self.x)
        with pytest.raises(ValueError):
            as_tensor_variable(bad_apply_var)

    def test_strip_leading_broadcastable(self):
        x = TensorType(config.floatX, (True, False))("x")
        x = as_tensor_variable(x, ndim=1)
        assert x.ndim == 1

        x = matrix("x", dtype=config.floatX)
        with pytest.raises(ValueError):
            as_tensor_variable(x, ndim=1)

    def test_bool(self):
        # We should not allow `as_tensor_variable` to accept `True` or `False`,
        # but it should up-cast an `ndarray` of `bool` to uint8
        with pytest.raises(TypeError):
            as_tensor_variable(True)

        ten = as_tensor_variable(np.array([True, False, False, True, True]))
        assert ten.type.dtype == "bool"

    def test_dtype(self):
        res = as_tensor_variable([])
        assert res.type.dtype == config.floatX

        res = as_tensor_variable([], dtype="int64")
        assert res.type.dtype == "int64"

        res = as_tensor_variable(np.array([1], dtype="int32"), dtype="int64")
        assert res.type.dtype == "int64"

        res = as_tensor_variable(np.array([1.0], dtype=config.floatX), dtype="int64")
        # TODO: This cross-type conversion probably shouldn't be the default.
        assert res.type.dtype == "int64"

        x = as_tensor_variable(np.array([1.0, 2.0], dtype="float64"))
        # This shouldn't convert the dtype, because it's already a `Variable`
        # with a set dtype
        res = as_tensor_variable(x, dtype="int64")
        assert res.type.dtype == "float64"

    def test_memmap(self):
        inp = np.random.random((4, 3))
        _, fname = mkstemp()
        new_inp = np.memmap(fname, dtype=inp.dtype, mode="w+", shape=inp.shape)
        new_inp[...] = inp
        res = as_tensor_variable(new_inp)
        assert isinstance(res, TensorConstant)
        assert res.data is new_inp

    @pytest.mark.parametrize(
        "dtype",
        [
            "float16",
            "float32",
            "float64",
        ],
    )
    def test_empty_dtype(self, dtype):
        with config.change_flags(floatX=dtype):
            assert as_tensor_variable(()).dtype == dtype
            assert as_tensor_variable([]).dtype == dtype

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            ([1, 2], [1, 2]),
            ([aet.as_tensor(1), aet.as_tensor(2)], [1, 2]),
            ([aes.constant(1), aes.constant(2)], [1, 2]),
        ],
    )
    def test_constant_consistency(self, x, y):
        a = as_tensor_variable(x)
        assert isinstance(a, TensorConstant)
        assert np.array_equal(a.data, y)

    def test_constant_identity(self):
        # Values that are already `TensorType`s shouldn't be recreated by
        # `as_tensor_variable`
        x_scalar = TensorConstant(TensorType(dtype="int8", broadcastable=()), 2)
        a_scalar = as_tensor_variable(x_scalar)
        assert x_scalar is a_scalar

        x_vector = TensorConstant(
            TensorType(dtype="int8", broadcastable=(False,)),
            np.array([1, 2], dtype="int8"),
        )
        a_vector = as_tensor_variable(x_vector)
        assert x_vector is a_vector

    def test_make_vector(self):
        a = iscalar()
        x = aet.tile(a, (1, 1, 1))
        y = (constant(1, dtype="int64"), x.shape[2])
        res = aet.as_tensor(y, ndim=1)
        assert isinstance(res.owner.op, MakeVector)
        assert tuple(res.owner.inputs) == y

        y = (1, x.shape[2])
        res = aet.as_tensor(y)
        assert isinstance(res.owner.op, MakeVector)

    def test_multi_out(self):
        class TestOp(Op):
            def make_node(self, a, b):
                return Apply(self, [a, b], [a, b])

        with pytest.raises(TypeError):
            aet.as_tensor(TestOp(matrix(), matrix()))


class TestAlloc:
    dtype = config.floatX
    mode = mode_opt
    shared = staticmethod(aesara.shared)
    allocs = [Alloc()] * 3

    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())

    def test_alloc_constant_folding(self):
        test_params = np.asarray(self.rng.standard_normal((50 * 60)), self.dtype)

        some_vector = vector("some_vector", dtype=self.dtype)
        some_matrix = some_vector.reshape((60, 50))
        variables = self.shared(np.ones((50,), dtype=self.dtype))
        idx = constant(np.arange(50))

        for alloc_, (subtensor, n_alloc) in zip(
            self.allocs,
            [
                # IncSubtensor1
                (some_matrix[:60], 2),
                # AdvancedIncSubtensor1
                (some_matrix[arange(60)], 2),
                # AdvancedIncSubtensor
                (some_matrix[idx, idx], 1),
            ],
        ):
            derp = aet_sum(dense_dot(subtensor, variables))

            fobj = aesara.function([some_vector], derp, mode=self.mode)
            grad_derp = aesara.grad(derp, some_vector)
            fgrad = aesara.function([some_vector], grad_derp, mode=self.mode)

            topo_obj = fobj.maker.fgraph.toposort()
            assert np.sum([isinstance(node.op, type(alloc_)) for node in topo_obj]) == 0

            topo_grad = fgrad.maker.fgraph.toposort()
            assert (
                np.sum([isinstance(node.op, type(alloc_)) for node in topo_grad])
                == n_alloc
            ), (alloc_, subtensor, n_alloc, topo_grad)
            fobj(test_params)
            fgrad(test_params)

    def test_alloc_output(self):
        val = constant(self.rng.standard_normal((1, 1)), dtype=self.dtype)
        for alloc_ in self.allocs:
            # The output is the result of the alloc operation,
            # we do not want it to be constant-folded
            out = alloc_(val, 50, 60)

            f = aesara.function([], out, mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert np.sum([isinstance(node.op, type(alloc_)) for node in topo]) == 1
            assert not isinstance(topo[0].op, DeepCopyOp)

    def test_ones(self):
        for shp in [[], 1, [1], [1, 2], [1, 2, 3], np.r_[1, 2, 3]]:
            ones = aesara.function([], [aet.ones(shp)], mode=self.mode)
            assert np.allclose(ones(), np.ones(shp))

        # scalar doesn't have to be provided as input
        x = scalar()
        shp = []
        ones_scalar = aesara.function([], [aet.ones(x.shape)], mode=self.mode)
        assert np.allclose(ones_scalar(), np.ones(shp))

        for (typ, shp) in [(vector, [3]), (matrix, [3, 4])]:
            x = typ()
            ones_tensor = aesara.function([x], [aet.ones(x.shape)], mode=self.mode)
            inp = np.zeros(shp, dtype=config.floatX)
            assert np.allclose(ones_tensor(inp), np.ones(shp))

    def test_zeros(self):
        for shp in [[], 1, [1], [1, 2], [1, 2, 3], np.r_[1, 2, 3]]:
            zeros = aesara.function([], [aet.zeros(shp)], mode=self.mode)
            assert np.allclose(zeros(), np.zeros(shp))

        # scalar doesn't have to be provided as input
        x = scalar()
        shp = []
        zeros_scalar = aesara.function([], [aet.zeros(x.shape)], mode=self.mode)
        assert np.allclose(zeros_scalar(), np.zeros(shp))

        for (typ, shp) in [(vector, [3]), (matrix, [3, 4])]:
            x = typ()
            zeros_tensor = aesara.function([x], [aet.zeros(x.shape)], mode=self.mode)
            inp = np.zeros(shp, dtype=config.floatX)
            assert np.allclose(zeros_tensor(inp), np.zeros(shp))

    def test_full(self):
        full_at = aet.full((2, 3), 3, dtype="int64")
        res = aesara.function([], full_at, mode=self.mode)()
        assert np.array_equal(res, np.full((2, 3), 3, dtype="int64"))


# This is slow for the ('int8', 3) version.
def test_eye():
    def check(dtype, N, M_=None, k=0):
        # Aesara does not accept None as a tensor.
        # So we must use a real value.
        M = M_
        # Currently DebugMode does not support None as inputs even if this is
        # allowed.
        if M is None and config.mode in ["DebugMode", "DEBUG_MODE"]:
            M = N
        N_symb = iscalar()
        M_symb = iscalar()
        k_symb = iscalar()
        f = function([N_symb, M_symb, k_symb], eye(N_symb, M_symb, k_symb, dtype=dtype))
        result = f(N, M, k)
        assert np.allclose(result, np.eye(N, M_, k, dtype=dtype))
        assert result.dtype == np.dtype(dtype)

    for dtype in ALL_DTYPES:
        check(dtype, 3)
        # M != N, k = 0
        check(dtype, 3, 5)
        check(dtype, 5, 3)
        # N == M, k != 0
        check(dtype, 3, 3, 1)
        check(dtype, 3, 3, -1)
        # N < M, k != 0
        check(dtype, 3, 5, 1)
        check(dtype, 3, 5, -1)
        # N > M, k != 0
        check(dtype, 5, 3, 1)
        check(dtype, 5, 3, -1)


class TestTriangle:
    def test_tri(self):
        def check(dtype, N, M_=None, k=0):
            # Aesara does not accept None as a tensor.
            # So we must use a real value.
            M = M_
            # Currently DebugMode does not support None as inputs even if this is
            # allowed.
            if M is None and config.mode in ["DebugMode", "DEBUG_MODE"]:
                M = N
            N_symb = iscalar()
            M_symb = iscalar()
            k_symb = iscalar()
            f = function(
                [N_symb, M_symb, k_symb], tri(N_symb, M_symb, k_symb, dtype=dtype)
            )
            result = f(N, M, k)
            assert np.allclose(result, np.tri(N, M_, k, dtype=dtype))
            assert result.dtype == np.dtype(dtype)

        for dtype in ALL_DTYPES:
            check(dtype, 3)
            # M != N, k = 0
            check(dtype, 3, 5)
            check(dtype, 5, 3)
            # N == M, k != 0
            check(dtype, 3, 3, 1)
            check(dtype, 3, 3, -1)
            # N < M, k != 0
            check(dtype, 3, 5, 1)
            check(dtype, 3, 5, -1)
            # N > M, k != 0
            check(dtype, 5, 3, 1)
            check(dtype, 5, 3, -1)

    def test_tril_triu(self):
        def check_l(m, k=0):
            m_symb = matrix(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], tril(m_symb, k_symb))
            result = f(m, k)
            assert np.allclose(result, np.tril(m, k))
            assert result.dtype == np.dtype(dtype)

        def check_u(m, k=0):
            m_symb = matrix(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], triu(m_symb, k_symb))
            result = f(m, k)
            assert np.allclose(result, np.triu(m, k))
            assert result.dtype == np.dtype(dtype)

        for dtype in ALL_DTYPES:
            m = random_of_dtype((10, 10), dtype)
            check_l(m, 0)
            check_l(m, 1)
            check_l(m, -1)

            check_u(m, 0)
            check_u(m, 1)
            check_u(m, -1)

            m = random_of_dtype((10, 5), dtype)
            check_l(m, 0)
            check_l(m, 1)
            check_l(m, -1)

            check_u(m, 0)
            check_u(m, 1)
            check_u(m, -1)


class TestNonzero:
    @config.change_flags(compute_test_value="raise")
    def test_nonzero(self):
        def check(m):
            m_symb = tensor(dtype=m.dtype, broadcastable=(False,) * m.ndim)
            m_symb.tag.test_value = m

            res_tuple_aet = nonzero(m_symb, return_matrix=False)
            res_matrix_aet = nonzero(m_symb, return_matrix=True)

            res_tuple = tuple(r.tag.test_value for r in res_tuple_aet)
            res_matrix = res_matrix_aet.tag.test_value

            assert np.allclose(res_matrix, np.vstack(np.nonzero(m)))

            for i, j in zip(res_tuple, np.nonzero(m)):
                assert np.allclose(i, j)

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)

    @config.change_flags(compute_test_value="raise")
    def test_flatnonzero(self):
        def check(m):
            m_symb = tensor(dtype=m.dtype, broadcastable=(False,) * m.ndim)
            m_symb.tag.test_value = m

            res_aet = flatnonzero(m_symb)

            result = res_aet.tag.test_value
            assert np.allclose(result, np.flatnonzero(m))

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)

    @config.change_flags(compute_test_value="raise")
    def test_nonzero_values(self):
        def check(m):
            m_symb = tensor(dtype=m.dtype, broadcastable=(False,) * m.ndim)
            m_symb.tag.test_value = m

            res_aet = nonzero_values(m_symb)

            result = res_aet.tag.test_value
            assert np.allclose(result, m[np.nonzero(m)], equal_nan=True)

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)


def test_identity():
    def check(dtype):
        obj = random_of_dtype((2,), dtype)
        sym = vector(dtype=dtype)
        f = function([sym], tensor_copy(sym))
        assert np.all(obj == f(obj))
        assert obj.dtype == f(obj).dtype
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        if config.mode != "FAST_COMPILE":
            assert isinstance(topo[0].op, DeepCopyOp)

    for dtype in ALL_DTYPES:
        check(dtype)


class TestCast:
    def test_can_use_numpy_types(self):
        x = vector(dtype=np.int32)
        y = cast(x, np.int64)
        f = function([x], y)
        assert f(np.array([1, 2], dtype=np.int32)).dtype == np.int64

    def test_good_between_real_types(self):
        good = itertools.chain(
            multi_dtype_cast_checks((2,), dtypes=REAL_DTYPES),
            # Casts from foo to foo
            [
                (
                    f"{random_of_dtype((2,), dtype)}_{dtype}",
                    (random_of_dtype((2,), dtype), dtype),
                )
                for dtype in ALL_DTYPES
            ],
        )
        for testname, (obj, dtype) in good:
            inp = vector(dtype=obj.dtype)
            out = cast(inp, dtype=dtype)
            f = function([inp], out)
            assert f(obj).dtype == np.dtype(dtype)

            # Test astype too
            out2 = inp.astype(dtype=dtype)
            assert out2.type == out.type

    def test_cast_from_real_to_complex(self):
        for real_dtype in REAL_DTYPES:
            for complex_dtype in COMPLEX_DTYPES:
                inp = vector(dtype=real_dtype)
                out = cast(inp, dtype=complex_dtype)
                f = function([inp], out)
                obj = random_of_dtype((2,), real_dtype)
                assert f(obj).dtype == np.dtype(complex_dtype)

    def test_cast_from_complex_to_real_raises_error(self):
        for real_dtype in REAL_DTYPES:
            for complex_dtype in COMPLEX_DTYPES:
                inp = vector(dtype=real_dtype)
                with pytest.raises(TypeError):
                    tensor(cast(inp, dtype=complex_dtype))


# TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary every time you want to verify
#      gradient numerically


def test_nan_inf_constant_signature():
    # Test that the signature of a constant tensor containing NaN and Inf
    # values is correct.
    test_constants = [
        [np.nan, np.inf, 0, 1],
        [np.nan, np.inf, -np.inf, 1],
        [0, np.inf, -np.inf, 1],
        [0, 3, -np.inf, 1],
        [0, 3, np.inf, 1],
        [np.nan, 3, 4, 1],
        [0, 3, 4, 1],
        np.nan,
        np.inf,
        -np.inf,
        0,
        1,
    ]
    n = len(test_constants)
    # We verify that signatures of two rows i, j in the matrix above are
    # equal if and only if i == j.
    for i in range(n):
        for j in range(n):
            x = constant(test_constants[i])
            y = constant(test_constants[j])
            assert (x.signature() == y.signature()) == (i == j)

    # Also test that nan !=0 and nan != nan.
    x = scalar()
    mode = get_default_mode()
    if isinstance(mode, aesara.compile.debugmode.DebugMode):
        # Disable the check preventing usage of NaN / Inf values.
        # We first do a copy of the mode to avoid side effects on other tests.
        mode = copy(mode)
        mode.check_isfinite = False
    f = aesara.function([x], eq(x, np.nan), mode=mode)

    assert f(0) == 0
    assert f(np.nan) == 0


def test_basic_allclose():
    # This was raised by a user in https://github.com/Theano/Theano/issues/2975
    assert tm._allclose(-0.311023883434, -0.311022856884)


def test_get_vector_length():
    x = aesara.shared(np.zeros((2, 3, 4, 5)))
    assert len(list(x.shape)) == 4
    assert len(list(x.shape[2:4])) == 2
    assert len(list(x.shape[2:])) == 2
    assert len(list(x.shape[1:4])) == 3
    assert len(list(x.shape[2:2])) == 0
    assert len(list(x.shape[1:5])) == 3
    assert len(list(x.shape[1:10])) == 3
    # Test step
    assert len(list(x.shape[1:10:2])) == 2
    # Test neg start
    assert len(list(x.shape[-1:4])) == 1
    assert len(list(x.shape[-6:4])) == 4
    # test neg stop
    assert len(list(x.shape[1:-2])) == 1
    assert len(list(x.shape[1:-1])) == 2

    z = join(0, as_tensor_variable(1, ndim=1), as_tensor_variable(x.shape[0], ndim=1))
    assert isinstance(z.owner.op, Join)
    assert get_vector_length(z) == 2

    z = join(
        0, as_tensor_variable([1, 2], ndim=1), as_tensor_variable(x.shape[0], ndim=1)
    )
    assert isinstance(z.owner.op, Join)
    assert get_vector_length(z) == 3

    empty_tuple = as_tensor_variable(())
    assert 0 == get_vector_length(empty_tuple)

    x = lscalar("x")
    y = dscalar("y")

    triple = as_tensor_variable((x, y, 9.0))
    assert 3 == get_vector_length(triple)

    triple = cast(as_tensor_variable((x, y, 9.0)), "int64")
    assert 3 == get_vector_length(triple)

    a, b, c = triple
    mode = aesara.compile.get_default_mode().excluding("constant_folding")
    f = function([x, y], [b, c, a], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert [True for node in topo if isinstance(node.op, MakeVector)]

    assert np.allclose(f(4, 5), [5, 9, 4])


class TestJoinAndSplit:
    # Split is tested by each verify_grad method.
    def setup_method(self):
        Join.debug = False

        self.mode = aesara.compile.get_default_mode().excluding("constant_folding")
        self.join_op = Join()
        self.split_op_class = Split
        self.make_vector_op = MakeVector()
        self.floatX = config.floatX
        self.hide_error = config.mode not in [
            "DebugMode",
            "DEBUG_MODE",
            "FAST_COMPILE",
        ]
        self.shared = shared

    def eval_outputs_and_check_join(self, outputs):
        f = aesara.function([], outputs, self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]
        variables = f()
        if isinstance(variables, (tuple, list)) and len(variables) == 1:
            return variables[0]
        return variables

    def eval_outputs_and_check_vector(self, outputs, make_vector_op=None):
        if make_vector_op is None:
            make_vector_op = self.make_vector_op
        f = aesara.function([], outputs, self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(make_vector_op))]
        variables = f()
        if isinstance(variables, (tuple, list)) and len(variables) == 1:
            return variables[0]
        return variables

    def test_join_scalar(self):
        a = as_tensor_variable(1)
        b = as_tensor_variable(2)
        with pytest.raises(TypeError):
            join(0, a, b)

    def test_stack_mixed_type_constants(self):
        # tested only on cpu as gpu support only float32
        a = as_tensor_variable(1)
        b = as_tensor_variable(2.0)
        c = aesara.shared(np.asarray(3.0, dtype=self.floatX))
        s = stack([a, b, c])
        want = np.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s], MakeVector())
        assert (out == want).all()

    def test_stack_scalar(self):
        a = self.shared(np.asarray(1.0, dtype=self.floatX))
        b = as_tensor_variable(2.0)
        c = as_tensor_variable(3.0)
        s = stack([a, b, c])

        want = np.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s])
        assert (out == want).all()

    def test_stack_scalar_make_vector(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # not Join. Test that the floatX dtype stay floatX, not downcasted
        # to int64
        a = scalar("a", dtype=self.floatX)
        b = scalar("b", dtype=self.floatX)
        s = stack([a, b, a, b])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        # print val
        assert np.all(val == [1, 2, 1, 2])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == self.floatX

    def test_stack_scalar_make_vector_dtype(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # event when the scalar don't have the same dtype.
        a = iscalar("a")
        b = lscalar("b")
        s = stack([a, b, a, b])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        assert np.all(val == [1, 2, 1, 2])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == "int64"

    def test_stack_scalar_make_vector_constant(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # event when the scalar are simple int type.
        a = iscalar("a")
        b = lscalar("b")
        # test when the constant is the first element.
        # The first element is used in a special way
        s = stack([10, a, b, np.int8(3)])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        assert np.all(val == [10, 1, 2, 3])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == "int64"

    def test_stack_new_interface(self):
        # Test the new numpy-like interface: stack(tensors, axis=0).

        # Testing against old interface
        warnings.simplefilter("always", DeprecationWarning)
        a = imatrix("a")
        b = imatrix("b")
        s1 = stack(a, b)
        s2 = stack([a, b])
        f = function([a, b], [s1, s2], mode=self.mode)
        v1, v2 = f([[1, 2]], [[3, 4]])
        assert v1.shape == v2.shape
        assert np.all(v1 == v2)
        # Testing axis parameter
        s3 = stack([a, b], 1)
        f = function([a, b], s3, mode=self.mode)
        v3 = f([[1, 2]], [[3, 4]])
        v4 = np.array([[[1, 2], [3, 4]]])
        assert v3.shape == v4.shape
        assert np.all(v3 == v4)
        # Testing negative axis
        v1 = [[1, 2, 3], [4, 5, 6]]
        v2 = [[7, 8, 9], [10, 11, 12]]
        s = stack([a, b], axis=-1)
        f = function([a, b], s, mode=self.mode)
        v = np.zeros((2, 3, 2))
        v[:, :, 0] = v1
        v[:, :, 1] = v2
        out = f(v1, v2)
        assert v.shape == out.shape
        assert np.all(v == out)
        s = stack([a, b], axis=-2)
        f = function([a, b], s, mode=self.mode)
        v = np.zeros((2, 2, 3))
        v[:, 0, :] = v1
        v[:, 1, :] = v2
        out = f(v1, v2)
        assert v.shape == out.shape
        assert np.all(v == out)
        # Testing out-of-bounds axis
        with pytest.raises(IndexError):
            stack([a, b], 4)
        with pytest.raises(IndexError):
            stack([a, b], -4)
        # Testing depreciation warning
        with warnings.catch_warnings(record=True) as w:
            s = stack(a, b)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
        with warnings.catch_warnings(record=True) as w:
            s = stack([a, b])
            s = stack([a, b], 1)
            s = stack([a, b], axis=1)
            s = stack(tensors=[a, b])
            s = stack(tensors=[a, b], axis=1)
            assert not w

    def test_stack_hessian(self):
        # Test the gradient of stack when used in hessian, see gh-1589
        a = dvector("a")
        b = dvector("b")
        A = stack([a, b])
        B = A.T.dot(A)
        Ha, Hb = hessian(B.sum(), [a, b])

        # Try some values
        a_v = np.random.random((4))
        b_v = np.random.random((4))
        f = aesara.function([a, b], [Ha, Hb])
        Ha_v, Hb_v = f(a_v, b_v)
        # The Hessian is always a matrix full of 2
        assert Ha_v.shape == (4, 4)
        assert Hb_v.shape == (4, 4)
        assert np.allclose(Ha_v, 2.0)
        assert np.allclose(Hb_v, 2.0)

    def test_stack_hessian2(self):
        # Test the hessian macro when the gradient itself does not depend
        # on the input (but the cost does)
        a = dvector("a")
        b = dvector("b")
        A = stack([a, b])
        Ha, Hb = hessian(A.sum(), [a, b])

        # Try some values
        a_v = np.random.random((4))
        b_v = np.random.random((4))
        f = aesara.function([a, b], [Ha, Hb])
        Ha_v, Hb_v = f(a_v, b_v)
        # The Hessian is always a matrix full of 0
        assert Ha_v.shape == (4, 4)
        assert Hb_v.shape == (4, 4)
        assert np.allclose(Ha_v, 0.0)
        assert np.allclose(Hb_v, 0.0)

    def test_join_concatenate_one_element(self):
        # Fast test of concatenate as this is an alias for join.
        # also test that we remove the Join op if there is only 1 input
        m = fmatrix()
        c = aet.concatenate([m])
        f = aesara.function(
            inputs=[m], outputs=[c], mode=self.mode.including("local_join_1")
        )
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

    def test_join_vector(self):
        a = self.shared(np.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(np.array([7, 8, 9], dtype=self.floatX))

        s = join(0, a, b)
        want = np.array([1, 2, 3, 7, 8, 9])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_roll(self):

        for get_shift in [lambda a: a, lambda x: aesara.shared(x)]:
            # Test simple 1D example
            a = self.shared(np.array([1, 2, 3, 4, 5, 6], dtype=self.floatX))
            b = roll(a, get_shift(2))
            want = np.array([5, 6, 1, 2, 3, 4])
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test simple 1D example with explicit 0 axis
            b = roll(a, get_shift(-1), 0)
            want = np.array([2, 3, 4, 5, 6, 1])
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test 2D example - ensure that behavior matches np.roll behavior
            a = self.shared(np.arange(21).reshape((3, 7)).astype(self.floatX))
            b = roll(a, get_shift(-2), 1)

            want = np.roll(a.get_value(borrow=True), -2, 1)
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test example when axis < 0 - ensure that behavior matches np.roll behavior
            a = self.shared(np.arange(24).reshape((3, 2, 4)).astype(self.floatX))
            b = roll(a, get_shift(-2), -2)

            want = np.roll(a.get_value(borrow=True), -2, -2)
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0
            want = np.roll(a.get_value(borrow=True), -2, 0)
            b = roll(a, get_shift(-2), 0)
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test rolling on default axis with ndim > 1
            want = np.roll(a.get_value(borrow=True), 2)
            b = roll(a, get_shift(2))
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0 with a positive shift that is
            # larger than axis size
            want = np.roll(a.get_value(borrow=True), 4, 0)
            b = roll(a, get_shift(4), 0)
            out = aesara.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0 with a negative shift that is
            # larger than axis size
            want = np.roll(a.get_value(borrow=True), -4, 0)
            b = roll(a, get_shift(-4), 0)
            out = aesara.function([], b)()

            assert (out == want).all()

    def test_stack_vector(self):
        a = self.shared(np.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(np.array([7, 8, 9], dtype=self.floatX))

        s = stack([a, b])
        want = np.array([[1, 2, 3], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix0(self):
        a = self.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(np.array([[7, 8, 9]], dtype=self.floatX))
        s = join(0, a, b)

        want = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix1(self):
        av = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        bv = np.array([[0.7], [0.8]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[0.1, 0.2, 0.3, 0.7], [0.4, 0.5, 0.6, 0.8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv], mode=self.mode)

    def test_join_matrix_dtypes(self):
        if "float32" in self.shared.__name__:
            pytest.skip(
                "The shared variable constructor"
                " need to support other dtype then float32"
            )
        # Test mixed dtype. There was a bug that caused crash in the past.
        av = np.array([[1, 2, 3], [4, 5, 6]], dtype="int8")
        bv = np.array([[7], [8]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        grad(s.sum(), b)
        grad(s.sum(), a)
        utt.verify_grad(lambda b: join(1, a, b), [bv], eps=1.0e-2, mode=self.mode)

    def test_join_matrix_ints(self):
        if "float32" in self.shared.__name__:
            pytest.skip(
                "The shared variable constructor"
                " need to support other dtype then float32"
            )
        # Test mixed dtype. There was a bug that caused crash in the past.
        av = np.array([[1, 2, 3], [4, 5, 6]], dtype="int8")
        bv = np.array([[7], [8]], dtype="int32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        assert (np.asarray(grad(s.sum(), b).eval()) == 0).all()
        assert (np.asarray(grad(s.sum(), a).eval()) == 0).all()

    def test_join_matrix1_using_vertical_stack(self):
        a = self.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(np.array([[7, 8, 9]], dtype=self.floatX))
        c = as_tensor_variable(np.array([[9, 8, 7]], dtype=self.floatX))
        s = vertical_stack(a, b, c)

        want = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix1_using_horizontal_stack(self):
        av = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        bv = np.array([[0.7], [0.8]], dtype="float32")
        cv = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        c = as_tensor_variable(cv)
        s = horizontal_stack(a, b, c)
        want = np.array(
            [[0.1, 0.2, 0.3, 0.7, 0.3, 0.2, 0.1], [0.4, 0.5, 0.6, 0.8, 0.6, 0.5, 0.4]],
            dtype="float32",
        )
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv], mode=self.mode)

    def test_join_matrixV(self):
        # variable join axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        got = f(0)
        assert np.allclose(got, want)

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )
        got = f(1)
        assert np.allclose(got, want)

        utt.verify_grad(lambda a, b: join(0, a, b), [v, 2 * v], mode=self.mode)
        utt.verify_grad(lambda a, b: join(1, a, b), [v, 2 * v], mode=self.mode)

    def test_join_matrixV_negative_axis(self):
        # variable join negative axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )

        got = f(-1)
        assert np.allclose(got, want)

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        got = f(-2)
        assert np.allclose(got, want)

        with pytest.raises(IndexError):
            f(-3)

    def test_join_matrixC_negative_axis(self):
        # constant join negative axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)

        s = join(-1, a, b)
        f = aesara.function([], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )

        got = f()
        assert np.allclose(got, want)

        s = join(-2, a, b)
        f = aesara.function([], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        got = f()
        assert np.allclose(got, want)

        with pytest.raises(IndexError):
            join(-3, a, b)

        utt.verify_grad(lambda a, b: join(-1, a, b), [v, 2 * v], mode=self.mode)

    def test_broadcastable_flag_assignment_mixed_otheraxes(self):
        # Test that the broadcastable flags for the output of
        # a join operation on non-join axes are True if one or
        # more inputs is broadcastable on that dimension.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 4, 1)).astype(self.floatX)
        b_val = rng.random((1, 3, 1)).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(False, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op(1, a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        # Opt can remplace the int by an Aesara constant
        c = self.join_op(constant(1), a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        # In case futur opt insert other useless stuff
        c = self.join_op(cast(constant(1), dtype="int32"), a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(1, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )

        # Should raise an error if dimension 0 does not match
        a.set_value(rng.random((2, 4, 1)).astype(self.floatX))
        with pytest.raises(ValueError):
            f()

    def test_broadcastable_flag_assignment_mixed_thisaxes(self):
        # Test that the broadcastable flag of the join axis
        # is False when some inputs are broadcastable on that
        # dimension.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((2, 4, 1)).astype(self.floatX)
        b_val = rng.random((1, 4, 1)).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(False, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op(0, a, b)
        assert not c.type.broadcastable[0]

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )
        # Should raise an error if b_val.shape[0] is not 1
        # We can't set the value|
        with pytest.raises(TypeError):
            b.set_value(rng.random((3, 4, 1)).astype(self.floatX))
        a = TensorType(dtype=self.floatX, broadcastable=[0, 0, 1])()
        b = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1])()
        c = self.join_op(0, a, b)
        f = function([a, b], c, mode=self.mode)
        bad_b_val = rng.random((3, 4, 1)).astype(self.floatX)
        with pytest.raises(TypeError):
            f(a_val, bad_b_val)

    def test_broadcastable_flags_all_broadcastable_on_joinaxis(self):
        # Test that joining together several inputs which are all
        # broadcastable on the join dimension results in the output
        # being non-broadcastable on the join dimension.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 4, 1)).astype(self.floatX)
        b_val = rng.random((1, 4, 1)).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(True, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op(0, a, b)
        assert not c.type.broadcastable[0]

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )

    def test_broadcastable_single_input_broadcastable_dimension(self):
        # Test that all broadcastable flags are preserved by a
        # single-input join.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 4, 1)).astype(self.floatX)
        a = self.shared(a_val, broadcastable=(True, False, True))
        b = self.join_op(0, a)
        assert b.type.broadcastable[0]
        assert b.type.broadcastable[2]
        assert not b.type.broadcastable[1]

        f = function([], b, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert not [
                True for node in topo if isinstance(node.op, type(self.join_op))
            ]

        f()
        utt.verify_grad((lambda a: join(0, a)), [a_val], rng=rng, mode=self.mode)
        # Should raise an error if length of dimension 0 is not 1
        with pytest.raises(TypeError):
            a.set_value(rng.random((2, 4, 1)).astype(self.floatX))
        # with pytest.raises(TypeError):
        #    f(bad_a_val)

    def test_broadcastable_flags_many_dims_and_inputs(self):
        # Test that the right broadcastable flags get set for a join
        # with many inputs and many input dimensions.
        a = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 0, 0, 0])()
        b = TensorType(dtype=self.floatX, broadcastable=[1, 1, 1, 0, 0, 0])()
        c = TensorType(dtype=self.floatX, broadcastable=[1, 0, 0, 0, 0, 0])()
        d = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 1, 0, 1])()
        e = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 0, 0, 1])()
        f = self.join_op(0, a, b, c, d, e)
        fb = f.type.broadcastable
        assert not fb[0] and fb[1] and fb[2] and fb[3] and not fb[4] and fb[5]
        g = self.join_op(1, a, b, c, d, e)
        gb = g.type.broadcastable
        assert gb[0] and not gb[1] and gb[2] and gb[3] and not gb[4] and gb[5]
        h = self.join_op(4, a, b, c, d, e)
        hb = h.type.broadcastable
        assert hb[0] and hb[1] and hb[2] and hb[3] and not hb[4] and hb[5]

        f = function([a, b, c, d, e], f, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        b_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        c_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        d_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        e_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        f(a_val, b_val, c_val, d_val, e_val)
        utt.verify_grad(
            (lambda a, b, c, d, e: join(0, a, b, c, d, e)),
            [a_val, b_val, c_val, d_val, e_val],
            rng=rng,
            mode=self.mode,
        )
        # Should raise an error if length of dimension 0 is not 1
        bad_val = rng.random((2, 1, 1, 1, 2, 1)).astype(self.floatX)
        with pytest.raises(TypeError):
            f(bad_val, b_val, c_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, bad_val, c_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, bad_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, c_val, bad_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, c_val, d_val, bad_val)
        # Should raise an error if any dimension other than 4 has length != 1
        bad_a_val = rng.random((1, 2, 1, 1, 2, 1)).astype(self.floatX)
        bad_b_val = rng.random((1, 1, 1, 1, 2, 2)).astype(self.floatX)
        bad_c_val = rng.random((1, 1, 2, 1, 2, 1)).astype(self.floatX)
        bad_d_val = rng.random((1, 2, 1, 1, 2, 1)).astype(self.floatX)
        bad_e_val = rng.random((1, 1, 1, 2, 2, 1)).astype(self.floatX)
        with pytest.raises(ValueError):
            f(bad_a_val, b_val, c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, bad_b_val, c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, bad_c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, c_val, bad_d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, c_val, d_val, bad_e_val)

    def test_infer_shape_join(self):
        def get_mat(s1, s2):
            return np.asarray(np.random.uniform(size=(s1, s2)), dtype=self.floatX)

        x1 = self.shared(get_mat(3, 4))
        x2 = self.shared(get_mat(2, 4))
        x3 = self.shared(get_mat(1, 4))

        # Test dim 0
        z = self.join_op(0, x1, x2, x3)
        f = aesara.function([], z.shape, mode=self.mode)
        topo = f.maker.fgraph.toposort()

        out = f()
        assert (out == [6, 4]).all()

        if config.mode != "FAST_COMPILE":
            for node in f.maker.fgraph.toposort():
                assert not isinstance(node.op, type(self.join_op))

        # Test dim 1
        x1.set_value(get_mat(3, 4))
        x2.set_value(get_mat(3, 4))
        x3.set_value(get_mat(3, 5))
        z = self.join_op(1, x1, x2, x3)
        f = aesara.function([], z.shape, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        out = f()
        assert (out == [3, 13]).all()

        if config.mode != "FAST_COMPILE":
            for node in topo:
                assert not isinstance(node.op, type(self.join_op))

        with config.change_flags(compute_test_value="off"):
            # Test hide error
            x1.set_value(get_mat(3, 4))
            x2.set_value(get_mat(3, 4))
            x3.set_value(get_mat(2, 5))
            if not self.hide_error:
                with pytest.raises(ValueError):
                    f()
            else:
                f()

    def test_rebroadcast(self):
        # Regression test for a crash that used to happen when rebroadcasting.
        x = TensorType(self.floatX, [False, False, True])()
        u = TensorType(self.floatX, [False, False, True])()
        # This line used to crash.
        aet.concatenate([x, -u], axis=2)

    def test_concatenate_same(self):
        # Test that we can concatenate the same tensor multiple time.

        # In the past it was broken on the GPU.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        T_shared = self.shared(rng.random((3, 4)).astype(self.floatX))
        Tout = aet.concatenate([T_shared, T_shared])
        f = function([], Tout, mode=self.mode)
        out = f()
        if config.mode != "FAST_COMPILE":
            assert [
                True
                for node in f.maker.fgraph.toposort()
                if isinstance(node.op, type(self.join_op))
            ]
        assert np.allclose(
            out, np.concatenate([T_shared.get_value(), T_shared.get_value()])
        )

    def test_mixed_ndim_error(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        v = self.shared(rng.random((4)).astype(self.floatX))
        m = self.shared(rng.random((4, 4)).astype(self.floatX))
        with pytest.raises(TypeError):
            self.join_op(0, v, m)

    def test_split_0elem(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        m = self.shared(rng.random((4, 6)).astype(self.floatX))
        o = self.split_op_class(2)(m, 0, [4, 0])
        f = function([], o, mode=self.mode)
        assert any(
            [
                isinstance(node.op, self.split_op_class)
                for node in f.maker.fgraph.toposort()
            ]
        )
        o1, o2 = f()
        assert np.allclose(o1, m.get_value(borrow=True))
        assert np.allclose(o2, m.get_value(borrow=True)[4:])

    @config.change_flags(compute_test_value="off")
    def test_split_neg(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        m = self.shared(rng.random((4, 6)).astype(self.floatX))
        o = self.split_op_class(2)(m, 0, [5, -1])
        f = function([], o, mode=self.mode)
        assert any(
            [
                isinstance(node.op, self.split_op_class)
                for node in f.maker.fgraph.toposort()
            ]
        )
        with pytest.raises(ValueError):
            f()


def test_join_inplace():
    # Test join to work inplace.
    #
    # This function tests the case when several elements are passed to the
    # join function but all except one of them are empty. In this case join
    # should work inplace and the output should be the view of the non-empty
    # element.
    s = lscalar()
    x = vector("x")
    z = aet.zeros((s,))

    join = Join(view=0)
    c = join(0, x, z, z)

    f = aesara.function([In(x, borrow=True), s], Out(c, borrow=True))

    data = np.array([3, 4, 5], dtype=config.floatX)
    print(f(data, 0))

    if config.mode not in ["DebugMode", "DEBUG_MODE"]:
        assert f(data, 0) is data
    assert np.allclose(f(data, 0), [3, 4, 5])


def test_join_oneInput():
    # Test join when only 1 input is given.
    #
    # This functions tests the case when concatenate is called
    # on an array of tensors but the array has only one element.
    # In this case, we would like to avoid the computational
    # overhead of concatenation of one element.
    x_0 = fmatrix()
    x_1 = fmatrix()
    x_2 = fvector()
    join_0 = aet.concatenate([x_0], axis=1)
    join_1 = aet.concatenate([x_0, x_1, shape_padright(x_2)], axis=1)

    assert join_0 is x_0
    assert join_1 is not x_0


def test_TensorFromScalar():
    s = aes.constant(56)
    t = tensor_from_scalar(s)
    assert t.owner.op is tensor_from_scalar
    assert t.type.broadcastable == (), t.type.broadcastable
    assert t.type.ndim == 0, t.type.ndim
    assert t.type.dtype == s.type.dtype

    v = eval_outputs([t])

    assert v == 56, v
    assert isinstance(v, np.ndarray)
    assert v.shape == (), v.shape

    g = grad(t, s)
    assert eval_outputs([g]) == 0.0


def test_ScalarFromTensor():
    tc = constant(56)  # aes.constant(56)
    ss = scalar_from_tensor(tc)
    assert ss.owner.op is scalar_from_tensor
    assert ss.type.dtype == tc.type.dtype

    v = eval_outputs([ss])

    assert v == 56
    assert v.shape == ()

    if config.cast_policy == "custom":
        assert isinstance(v, np.int8)
    elif config.cast_policy in ("numpy", "numpy+floatX"):
        assert isinstance(v, str(np.asarray(56).dtype))
    else:
        raise NotImplementedError(config.cast_policy)

    aes = lscalar()
    ss = scalar_from_tensor(aes)
    ss.owner.op.grad([aes], [ss])
    fff = function([aes], ss)
    v = fff(np.asarray(5))
    assert v == 5
    assert isinstance(v, np.int64)
    assert v.shape == ()


class TestOpCache:
    def test_basic(self):
        # trigger bug in ticket #162
        v = matrix()
        v.name = "v"
        gv = fill(v / v, 1.0) / v - (fill(v / v, 1.0) * v) / (v * v)
        fn_py = inplace_func([v], gv)
        fn_c_or_py = inplace_func([v], gv)

        a = random(5, 2).astype(config.floatX)
        assert np.all(fn_py(a) == fn_c_or_py(a))


def test_dimshuffle():
    # The goal of the operation made by `b` is to ensure the second dimension
    # of the column matrix is broadcastable.
    a = dmatrix()
    b = a.reshape((a.shape[0],)).dimshuffle(0, "x")
    f = function([a], b)
    assert (f(np.zeros((3, 1))) + np.ones(2) == np.ones((3, 2))).all()


def test_flatten_ndim_default():
    a = dmatrix()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = _asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    c_val = _asarray([0, 1, 2, 3, 4, 5], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    utt.verify_grad(flatten, [a_val])


def test_flatten_scalar():
    a = dscalar()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = _asarray(3.0, dtype="float64")
    c_val = _asarray([3.0], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    # utt.verify_grad(flatten, [a_val]) #TODO: fix verify_grd to work on scalars


def test_flatten_ndim1():
    a = dmatrix()
    c = flatten(a, 1)
    f = inplace_func([a], c)
    a_val = _asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    c_val = _asarray([0, 1, 2, 3, 4, 5], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    utt.verify_grad(flatten, [a_val])


def test_flatten_ndim2():
    a = dmatrix()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = _asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    assert np.all(f(a_val) == a_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == a_val)

    flatten_2 = partial(flatten, ndim=2)
    utt.verify_grad(flatten_2, [a_val])


def test_flatten_ndim2_of_3():
    a = TensorType("float64", (False, False, False))()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = _asarray([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="float64")
    c_val = _asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    flatten_2 = partial(flatten, ndim=2)
    utt.verify_grad(flatten_2, [a_val])


def test_flatten_broadcastable():
    # Ensure that the broadcastable pattern of the output is coherent with
    # that of the input

    inp = TensorType("float64", (False, False, False, False))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, False)

    inp = TensorType("float64", (False, False, False, True))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, False)

    inp = TensorType("float64", (False, True, False, True))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, False)

    inp = TensorType("float64", (False, True, True, True))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, True)

    inp = TensorType("float64", (True, False, True, True))()
    out = flatten(inp, ndim=3)
    assert out.broadcastable == (True, False, True)


def test_flatten_ndim_invalid():
    a = dmatrix()
    with pytest.raises(ValueError):
        flatten(a, 3)
    with pytest.raises(ValueError):
        flatten(a, 0)


def test_is_flat():
    # tests is_flat method for constant and symbolic variables,
    # as well as reshaped constant and symbolic variables on the
    # given `ndim`

    # Constant variable
    assert aet.is_flat(aet.as_tensor_variable(np.zeros(10)))
    assert aet.is_flat(aet.as_tensor_variable(np.zeros((10, 10, 10))), ndim=3)
    assert not aet.is_flat(aet.as_tensor_variable(np.zeros((10, 10, 10))))

    # Symbolic variable
    assert aet.is_flat(vector())
    assert aet.is_flat(tensor3(), ndim=3)
    assert not aet.is_flat(tensor3())

    # Reshape with constant shape
    X = tensor4()
    assert aet.is_flat(X.reshape((-1,)))
    assert aet.is_flat(X.reshape((10, 10, -1)), ndim=3)
    assert not aet.is_flat(X.reshape((10, 10, -1)))

    # Reshape with symbolic shape
    X = tensor4()
    assert aet.is_flat(X.reshape((iscalar(),)))
    assert aet.is_flat(X.reshape((iscalar(),) * 3), ndim=3)
    assert not aet.is_flat(X.reshape((iscalar(),) * 3))


def test_tile():
    def run_tile(x, x_, reps, use_symbolic_reps):
        if use_symbolic_reps:
            rep_symbols = [iscalar() for _ in range(len(reps))]
            f = function([x] + rep_symbols, tile(x, rep_symbols))
            return f(*([x_] + list(reps)))
        else:
            f = function([x], tile(x, reps))
            return f(x_)

    rng = np.random.default_rng(utt.fetch_seed())

    for use_symbolic_reps in [False, True]:
        # Test the one-dimensional case.
        x = vector()
        x_ = rng.standard_normal(5).astype(config.floatX)
        assert np.all(run_tile(x, x_, (2,), use_symbolic_reps) == np.tile(x_, (2,)))

        # Test the two-dimensional case.
        x = matrix()
        x_ = rng.standard_normal((2, 4)).astype(config.floatX)
        assert np.all(run_tile(x, x_, (2, 3), use_symbolic_reps) == np.tile(x_, (2, 3)))

        # Test the three-dimensional case.
        x = tensor3()
        x_ = rng.standard_normal((2, 4, 3)).astype(config.floatX)
        assert np.all(
            run_tile(x, x_, (2, 3, 4), use_symbolic_reps) == np.tile(x_, (2, 3, 4))
        )

        # Test the four-dimensional case.
        x = tensor4()
        x_ = rng.standard_normal((2, 4, 3, 5)).astype(config.floatX)
        assert np.all(
            run_tile(x, x_, (2, 3, 4, 6), use_symbolic_reps)
            == np.tile(x_, (2, 3, 4, 6))
        )

    # Test when reps is integer, scalar or vector.
    # Test 1,2,3,4-dimensional cases.
    # Test input x has the shape [2], [2, 4], [2, 4, 3], [2, 4, 3, 5].
    test_shape = [2, 4, 3, 5]
    k = 0
    for xtype in [vector(), matrix(), tensor3(), tensor4()]:
        x = xtype
        k = k + 1
        x_ = rng.standard_normal(test_shape[0:k]).astype(config.floatX)

        # integer:
        reps_ = 2
        f = function([x], tile(x, reps_))
        assert np.all(f(x_) == np.tile(x_, reps_))

        # scalar:
        reps = iscalar()
        reps_ = 2
        f = function([x, reps], tile(x, reps))
        assert np.all(f(x_, reps_) == np.tile(x_, reps_))

        # vector:
        reps = ivector()
        reps_ = [2] if k == 1 or k == 2 else [2, 3]
        ndim_ = k
        f = function([x, reps], tile(x, reps, ndim_))
        assert np.all(f(x_, reps_) == np.tile(x_, reps_))

        # list of integers:
        reps_ = [2, 3, 4]
        f = function([x], tile(x, reps_))
        assert np.all(f(x_) == np.tile(x_, reps_))

        # list of integers and scalars:
        d = iscalar()
        reps = [2, d, 4]
        f = function([x, d], tile(x, reps))
        reps_ = [2, 3, 4]
        assert np.all(f(x_, 3) == np.tile(x_, reps_))

        # reps is list, len(reps) > x.ndim, 3 cases below:
        r = [2, 3, 4, 5, 6]
        reps_ = r[: k + 1]  # len(reps_) = x.ndim+1
        # (1) ndim = None.
        f = function([x], tile(x, reps_))
        assert np.all(f(x_) == np.tile(x_, reps_))
        # (2) ndim = len(reps).
        ndim_ = len(reps_)
        f = function([x], tile(x, reps_, ndim_))
        assert np.all(f(x_) == np.tile(x_, reps_))
        # (3) ndim > len(reps)
        ndim_ = len(reps_) + 1
        f = function([x], tile(x, reps_, ndim_))
        assert np.all(f(x_) == np.tile(x_, [1] + reps_))

        # reps is list, ndim > x.ndim > len(reps):
        r = [2, 3, 4, 5]
        if k > 1:
            ndim_ = k + 1
            reps_ = r[: k - 1]
            f = function([x], tile(x, reps_, ndim_))
            assert np.all(f(x_) == np.tile(x_, [1, 1] + reps_))

        # error raising test: ndim not specified when reps is vector
        reps = ivector()
        with pytest.raises(ValueError):
            tile(x, reps)

        # error raising test: not a integer
        for reps in [2.5, fscalar(), fvector()]:
            with pytest.raises(ValueError):
                tile(x, reps)

        # error raising test: the dimension of reps exceeds 1
        reps = imatrix()
        with pytest.raises(ValueError):
            tile(x, reps)

        # error raising test: ndim is not None, ndim < x.ndim
        # 3 cases below (reps is list/scalar/vector):
        for reps in [[2, 3, 4], iscalar(), ivector()]:
            if k > 1:
                ndim = k - 1
                with pytest.raises(ValueError):
                    tile(x, reps, ndim)

        # error raising test: reps is list, len(reps) > ndim
        r = [2, 3, 4, 5, 6]
        reps = r[: k + 1]
        ndim = k
        with pytest.raises(ValueError):
            tile(x, reps, ndim)

        # error raising test:
        # reps is vector and len(reps_value) > ndim,
        # reps_value is the real value when executing the function.
        reps = ivector()
        r = [2, 3, 4, 5, 6, 7]
        reps_ = r[: k + 2]
        ndim_ = k + 1
        f = function([x, reps], tile(x, reps, ndim_))
        with pytest.raises(AssertionError):
            f(x_, reps_)


def test_tile_grad():
    def grad_tile(x, reps, np_x):
        y = tile(x, reps)
        z = y.sum()
        g = aesara.function([x], grad(z, x))
        grad_res = g(np_x)
        # The gradient should be the product of the tiling dimensions
        # (since the gradients are additive through the tiling operation)
        assert np.all(grad_res == np.prod(reps))

    rng = np.random.default_rng(utt.fetch_seed())

    # test vector
    grad_tile(vector("x"), [3], rng.standard_normal((5)).astype(config.floatX))
    # test matrix
    grad_tile(matrix("x"), [3, 4], rng.standard_normal((2, 3)).astype(config.floatX))
    # test tensor3
    grad_tile(
        tensor3("x"), [3, 4, 5], rng.standard_normal((2, 4, 3)).astype(config.floatX)
    )
    # test tensor4
    grad_tile(
        tensor4("x"),
        [3, 4, 5, 6],
        rng.standard_normal((2, 4, 3, 5)).astype(config.floatX),
    )


class TestARange:
    def test_Op_integers(self):
        # Test behaviour of ARange Op on integer inputs
        start, stop, step = iscalars("start", "stop", "step")
        out = ARange(start.type.dtype)(start, stop, step)
        f = function([start, stop, step], out)

        assert np.all(f(0, 5, 1) == np.arange(0, 5, 1))
        assert np.all(f(2, 11, 4) == np.arange(2, 11, 4))
        assert np.all(f(-5, 1, 1) == np.arange(-5, 1, 1))
        assert np.all(f(10, 2, -2) == np.arange(10, 2, -2))
        assert np.all(f(10, 2, 2) == np.arange(10, 2, 2))
        assert np.all(f(0, 0, 1) == np.arange(0, 0, 1))

    def test_grads(self):
        def f(start, stop, step):
            return ARange(start.type.dtype)(start, stop, step)

        rng = np.random.default_rng(utt.fetch_seed())
        # Due to the random projection, we should not use the exact
        # point that change the shape of the output.
        for start, stop, step in [(0, 4.9, 1), (5.1, 0, -0.5), (1, 5.1, 0.5)]:
            utt.verify_grad(
                f,
                [
                    np.asarray(start).astype(config.floatX),
                    np.asarray(stop).astype(config.floatX),
                    np.asarray(step).astype(config.floatX),
                ],
                rng=rng,
            )

    def test_integers(self):
        # Test arange constructor, on integer outputs
        start, stop, step = iscalars("start", "stop", "step")
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            numpy_dtype = np.arange(np.array(1, dtype="int32")).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(0, 5, 1) == np.arange(0, 5, 1))
        assert np.all(f(2, 11, 4) == np.arange(2, 11, 4))
        assert np.all(f(-5, 1, 1) == np.arange(-5, 1, 1))
        assert np.all(f(10, 2, -2) == np.arange(10, 2, -2))
        assert np.all(f(10, 2, 2) == np.arange(10, 2, 2))
        assert np.all(f(0, 0, 1) == np.arange(0, 0, 1))

    def test_float32(self):
        # Test arange constructor, on float32 outputs
        start, stop, step = fscalars("start", "stop", "step")
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        if config.cast_policy == "custom":
            assert out.dtype == start.type.dtype
        elif config.cast_policy == "numpy":
            numpy_dtype = np.arange(
                np.array(0, dtype=start.dtype),
                np.array(1, dtype=stop.dtype),
                np.array(1, dtype=step.dtype),
            ).dtype
            assert out.dtype == numpy_dtype
        elif config.cast_policy == "numpy+floatX":
            assert out.dtype == config.floatX
        else:
            raise NotImplementedError(config.cast_policy)
        arg_vals = [(0, 5, 1), (2, 11, 4), (-5, 1.1, 1.2), (1.3, 2, -2.1), (10, 2, 2)]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = np.asarray(arg_v, dtype=start.type.dtype)
            f_val = f(start_v_, stop_v_, step_v_)
            if config.cast_policy == "custom":
                expected_val = np.arange(
                    start_v, stop_v, step_v, dtype=start.type.dtype
                )
            elif config.cast_policy in ("numpy", "numpy+floatX"):
                expected_val = np.arange(start_v_, stop_v_, step_v_, dtype=out.dtype)
            else:
                raise NotImplementedError(config.cast_policy)
            assert np.all(f_val == expected_val)

    def test_float64(self):
        # Test arange constructor, on float64 outputs
        start, stop, step = dscalars("start", "stop", "step")
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        arg_vals = [(0, 5, 1), (2, 11, 4), (-5, 1.1, 1.2), (1.3, 2, -2.1), (10, 2, 2)]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = np.asarray(arg_v, dtype=start.type.dtype)
            f_val = f(start_v_, stop_v_, step_v_)
            if config.cast_policy == "custom":
                expected_val = np.arange(
                    start_v, stop_v, step_v, dtype=start.type.dtype
                )
            elif config.cast_policy in ("numpy", "numpy+floatX"):
                expected_val = np.arange(start_v_, stop_v_, step_v_)
            else:
                raise NotImplementedError(config.cast_policy)
            assert np.all(f_val == expected_val)

    def test_default_step(self):
        # Test that arange constructor uses the correct default step
        start, stop = iscalars("start", "stop")
        out = arange(start, stop)
        f = function([start, stop], out)

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            assert out.dtype == np.arange(np.int32(0), np.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(0, 5) == np.arange(0, 5))
        assert np.all(f(-5, 1) == np.arange(-5, 1))
        assert np.all(f(0, 0) == np.arange(0, 0))

        dstart, dstop = dscalars("start", "stop")
        dout = arange(dstart, dstop)
        df = function([dstart, dstop], dout)

        assert dout.dtype == dstart.type.dtype
        # print df(0.2, 5.3)
        # print np.arange(0.2, 5.3)
        assert np.all(df(0.2, 5.3) == np.arange(0.2, 5.3))
        assert np.all(df(0.8, 5.3) == np.arange(0.8, 5.3))
        assert np.all(df(-0.7, 5.3) == np.arange(-0.7, 5.3))

    def test_default_start(self):
        # Test that arange constructor uses the correct default start
        stop = iscalar("stop")
        out = arange(stop)
        f = function([stop], out)

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            assert out.dtype == np.arange(np.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(8) == np.arange(8))
        assert np.all(f(-2) == np.arange(-2))

        fstop = fscalar("stop")
        fout = arange(fstop)
        ff = function([fstop], fout)

        if config.cast_policy == "custom":
            assert fout.dtype == fstop.type.dtype
        elif config.cast_policy == "numpy":
            assert fout.dtype == np.arange(np.float32(1)).dtype
        elif config.cast_policy == "numpy+floatX":
            if config.floatX == "float32":
                assert fout.dtype == "float32"
            else:
                assert fout.dtype == np.arange(np.float32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)

        fstop_values = [0.2, -0.7, 8.5]
        for fstop_v in fstop_values:
            fstop_v32 = np.float32(fstop_v)
            assert np.all(ff(fstop_v32) == np.arange(fstop_v))

    def test_upcast(self):
        # Test that arange computes output type adequately
        if config.cast_policy == "custom":
            assert arange(iscalar()).dtype == "int64"
            assert arange(fscalar()).dtype == fscalar().dtype
            assert arange(dscalar()).dtype == dscalar().dtype

            # int32 + float32 -> float64
            assert arange(iscalar(), fscalar()).dtype == dscalar().dtype
            assert arange(iscalar(), dscalar()).dtype == dscalar().dtype
            assert arange(fscalar(), dscalar()).dtype == dscalar().dtype

            assert arange(iscalar(), fscalar(), dscalar()).dtype == dscalar().dtype
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            for dtype in get_numeric_types():
                # Test with a single argument.
                arange_dtype = arange(scalar(dtype=str(dtype))).dtype
                numpy_dtype = np.arange(np.array(1, dtype=dtype)).dtype
                if (
                    dtype != "float64"
                    and numpy_dtype == "float64"
                    and config.cast_policy == "numpy+floatX"
                    and config.floatX == "float32"
                ):
                    # We want a float32 arange.
                    assert arange_dtype == "float32"
                else:
                    # Follow numpy.
                    assert arange_dtype == numpy_dtype

                # Test with two arguments.
                for stop_dtype in get_numeric_types():
                    arange_dtype = arange(
                        start=scalar(dtype=str(dtype)),
                        stop=scalar(dtype=str(stop_dtype)),
                    ).dtype
                    numpy_dtype = np.arange(
                        start=np.array(0, dtype=dtype),
                        stop=np.array(1, dtype=stop_dtype),
                    ).dtype
                    if (
                        dtype != "float64"
                        and stop_dtype != "float64"
                        and numpy_dtype == "float64"
                        and config.cast_policy == "numpy+floatX"
                        and config.floatX == "float32"
                    ):
                        # We want a float32 arange.
                        assert arange_dtype == "float32"
                    else:
                        # Follow numpy.
                        assert arange_dtype == numpy_dtype

                    # Test with three arguments.
                    for step_dtype in get_numeric_types():
                        arange_dtype = arange(
                            start=scalar(dtype=str(dtype)),
                            stop=scalar(dtype=str(stop_dtype)),
                            step=scalar(dtype=str(step_dtype)),
                        ).dtype
                        numpy_dtype = np.arange(
                            start=np.array(0, dtype=dtype),
                            stop=np.array(1, dtype=stop_dtype),
                            step=np.array(1, dtype=step_dtype),
                        ).dtype
                        if (
                            dtype != "float64"
                            and stop_dtype != "float64"
                            and step_dtype != "float64"
                            and numpy_dtype == "float64"
                            and config.cast_policy == "numpy+floatX"
                            and config.floatX == "float32"
                        ):
                            # We want a float32 arange.
                            assert arange_dtype == "float32"
                        else:
                            # Follow numpy.
                            assert arange_dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

    def test_dtype_cache(self):
        # Checks that the same Op is returned on repeated calls to arange
        # using the same dtype, but not for different dtypes.

        start, stop, step = iscalars("start", "stop", "step")
        out1 = arange(start, stop, step)
        out2 = arange(start, stop, step, dtype=out1.dtype)
        out3 = arange(start, stop, 2.0, dtype=out1.dtype)
        out4 = arange(start, stop, 2.0)

        assert out1.owner.op is out2.owner.op
        assert out2.owner.op is out3.owner.op
        assert out3.owner.op is not out4.owner.op

    def test_infer_shape(self):
        start, stop, step = iscalars("start", "stop", "step")
        out = arange(start, stop, step)
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        mode = compile.mode.get_mode(mode).excluding("fusion")
        f = function([start, stop, step], out.shape, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 9

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            numpy_dtype = np.arange(
                np.array(0, dtype=start.dtype),
                np.array(1, dtype=stop.dtype),
                np.array(1, dtype=step.dtype),
            ).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

        assert np.all(f(0, 5, 1) == len(np.arange(0, 5, 1)))
        assert np.all(f(2, 11, 4) == len(np.arange(2, 11, 4)))
        assert np.all(f(-5, 1, 1) == len(np.arange(-5, 1, 1)))
        assert np.all(f(10, 2, -2) == len(np.arange(10, 2, -2)))
        assert np.all(f(10, 2, 2) == len(np.arange(10, 2, 2)))
        assert np.all(f(0, 0, 1) == len(np.arange(0, 0, 1)))

        out = arange(start, stop, 1)
        f = function([start, stop], out.shape, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 5
        # 4 [Elemwise{sub,no_inplace}(stop, start), Elemwise{Cast{int64}}(Elemwise{sub,no_inplace}.0), Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{Cast{int64}}.0, 0), MakeVector(Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)].0)]
        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            assert out.dtype == np.arange(np.int32(0), np.int32(1), np.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(0, 5) == len(np.arange(0, 5)))
        assert np.all(f(2, 11) == len(np.arange(2, 11)))
        assert np.all(f(-5, 1) == len(np.arange(-5, 1)))
        assert np.all(f(10, 2) == len(np.arange(10, 2)))
        assert np.all(f(10, 2) == len(np.arange(10, 2)))
        assert np.all(f(0, 0) == len(np.arange(0, 0)))
        assert np.all(f(-64, 64) == len(np.arange(-64, 64)))
        assert arange(-64, 64).shape.eval() == [128]
        assert arange(-64, 64, 2).shape.eval() == [64]

        out = arange(0, stop, 1)
        f = function([stop], out.shape, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 2
        # [Elemwise{Cast{int64}}(stop), MakeVector(Elemwise{Cast{int64}}.0)]

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            numpy_dtype = np.arange(0, np.array(1, dtype=stop.dtype), 1).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

        assert np.all(f(5) == len(np.arange(0, 5)))
        assert np.all(f(11) == len(np.arange(0, 11)))
        assert np.all(f(1) == len(np.arange(0, 1)))
        assert np.all(f(2) == len(np.arange(0, 2)))
        assert np.all(f(2) == len(np.arange(0, 2)))
        assert np.all(f(0) == len(np.arange(0, 0)))


class TestNdGrid:
    def setup_method(self):
        pass

    def test_mgrid_numpy_equiv(self):
        nmgrid = (
            [np.mgrid[0:1:0.1]],
            np.mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            np.mgrid[0:2:1, 1:10:1, 10:100:10],
        )
        tmgrid = (
            [mgrid[0:1:0.1]],
            mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            mgrid[0:2:1, 1:10:1, 10:100:10],
        )
        for n, t in zip(nmgrid, tmgrid):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg.eval())

    def test_ogrid_numpy_equiv(self):
        nogrid = (
            [np.ogrid[0:1:0.1]],
            np.ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            np.ogrid[0:2:1, 1:10:1, 10:100:10],
        )
        togrid = (
            [ogrid[0:1:0.1]],
            ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            ogrid[0:2:1, 1:10:1, 10:100:10],
        )
        for n, t in zip(nogrid, togrid):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg.eval())

    def test_mgrid_aesara_variable_numpy_equiv(self):
        nfmgrid = np.mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0]
        nimgrid = np.mgrid[0:2:1, 1:10:1, 10:100:10]
        i, j, k = dscalars("i", "j", "k")
        l, m, n = iscalars("l", "m", "n")
        tfmgrid = mgrid[i:1:0.1, 1:j:1.0, 10:100:k]
        timgrid = mgrid[l:2:1, 1:m:1, 10:100:n]
        ff = aesara.function([i, j, k], tfmgrid)
        fi = aesara.function([l, m, n], timgrid)
        for n, t in zip((nfmgrid, nimgrid), (ff(0, 10, 10.0), fi(0, 10, 10))):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg)

    def test_ogrid_aesara_variable_numpy_equiv(self):
        nfogrid = np.ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0]
        niogrid = np.ogrid[0:2:1, 1:10:1, 10:100:10]
        i, j, k = dscalars("i", "j", "k")
        l, m, n = iscalars("l", "m", "n")
        tfogrid = ogrid[i:1:0.1, 1:j:1.0, 10:100:k]
        tiogrid = ogrid[l:2:1, 1:m:1, 10:100:n]
        ff = aesara.function([i, j, k], tfogrid)
        fi = aesara.function([l, m, n], tiogrid)
        for n, t in zip((nfogrid, niogrid), (ff(0, 10, 10.0), fi(0, 10, 10))):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg)


class TestInversePermutation:
    def test_dim1(self):
        # Test the inversion of one permutation (int vector)
        p = ivector()
        inv = inverse_permutation(p)
        assert inv.dtype == p.dtype
        f_inverse = function([p], inv)

        # Generate a random permutation
        rng = np.random.default_rng(utt.fetch_seed())
        p_val = rng.permutation(10).astype("int32")
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation
        assert np.all(f_inverse(inv_val) == p_val)
        # Check that permutation(inverse) == inverse(permutation) = identity
        assert np.all(p_val[inv_val] == np.arange(10))
        assert np.all(inv_val[p_val] == np.arange(10))

    def test_dim2(self):
        # Test the inversion of several permutations at a time
        # Each row of p is a different permutation to inverse
        p = imatrix()
        inv = inverse_permutation(p)
        f_inverse = function([p], inv)

        rng = np.random.default_rng(utt.fetch_seed())
        # Generate 10 random permutations
        p_val = np.asarray([rng.permutation(10) for i in range(7)], dtype="int32")
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation list
        assert np.all(f_inverse(inv_val) == p_val)
        # Check that, for each permutation,
        # permutation(inverse) == inverse(permutation) = identity
        for p_row, i_row in zip(p_val, inv_val):
            assert np.all(p_row[i_row] == np.arange(10))
            assert np.all(i_row[p_row] == np.arange(10))


class TestPermuteRowElements:
    def test_1_1(self):
        # Test PermuteRowElements(vector, vector)
        input = dvector()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(5,))
        p_val = rng.permutation(5).astype("int32")
        out_val = permute(input_val, p_val)

        # Should be equivalent to advanced indexing
        out_bis = input_val[p_val]
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_2_1(self):
        # Test broadcasting in PermuteRowElements(matrix, vector)
        input = matrix()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
        p_val = rng.permutation(5).astype("int32")
        out_val = permute(input_val, p_val)

        # The same permutation should be applied to every row of the input matrix.
        out_bis = np.asarray([r[p_val] for r in input_val])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_2_2(self):
        # Test PermuteRowElements(matrix, matrix)
        input = matrix()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the corresponding
        # row of input
        out_bis = np.asarray([i_row[p_row] for i_row, p_row in zip(input_val, p_val)])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_1_2(self):
        # Test PermuteRowElements(vector, matrix)
        # Different permutations will be applied to the same input vector
        input = vector()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(5,)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the input vector
        out_bis = np.asarray([input_val[p_row] for p_row in p_val])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_3b_2(self):
        # Test permute_row_elements on a more complex broadcasting pattern:
        # input.type.broadcastable = (False, True, False),
        # p.type.broadcastable = (False, False).

        input = TensorType("floatX", (False, True, False))()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(4, 1, 5)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to each row
        # of the input tensor
        out_bis = np.asarray(
            [[in_mat[0, p_row] for p_row in p_val] for in_mat in input_val]
        )
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])


def test_stack():
    sx, sy = dscalar(), dscalar()

    rval = inplace_func([sx, sy], stack([sx, sy]))(-4.0, -2.0)
    assert type(rval) == np.ndarray
    assert [-4, -2] == list(rval)


@pytest.mark.skipif(
    isinstance(get_default_mode(), aesara.compile.debugmode.DebugMode),
    reason="This test fails in DEBUG_MODE, but the generated code is OK. "
    "It is actually a problem of DEBUG_MODE, see #626.",
)
def test_default():
    x, y = scalars("xy")
    z = default(x, y)
    f = function([x, y], z)
    assert f(1, 2) == 1
    assert f(None, 2) == 2
    assert f(1, None) == 1


@pytest.mark.skipif(
    isinstance(get_default_mode(), aesara.compile.debugmode.DebugMode),
    reason="This test fails in DEBUG_MODE, but the generated code is OK. "
    "It is actually a problem of DEBUG_MODE, see #626.",
)
def test_default_state():
    x, y = scalars("xy")
    # print config.floatX
    # print x.type
    # print y.type
    z = default(x, 3.8)
    new_x = y + z
    f = function([y, compile.In(x, update=new_x, value=12.0)], new_x)
    assert f(3) == 15
    f["x"] = None
    assert np.allclose(f(1), 4.8)
    assert np.allclose(f(np.asarray(2.2, dtype=config.floatX)), 7)


def test_autocast():
    # Call test functions for all possible values of `config.cast_policy`.
    for autocast_cfg in (
        "custom",
        # 'numpy', # Commented out until it is implemented properly.
        "numpy+floatX",
    ):
        with config.change_flags(cast_policy=autocast_cfg):
            eval("_test_autocast_" + autocast_cfg.replace("+", "_"))()


def _test_autocast_custom():
    # Called from `test_autocast`.
    assert config.cast_policy == "custom"
    orig_autocast = autocast_float.dtypes

    # Test that autocast_float_as sets the autocast dtype correctly
    with autocast_float_as("float32"):
        assert autocast_float.dtypes == ("float32",)
    assert autocast_float.dtypes == orig_autocast

    with autocast_float_as("float64"):
        assert autocast_float.dtypes == ("float64",)
    assert autocast_float.dtypes == orig_autocast

    # Test that we can set it back to something, and nest it
    with autocast_float_as("float32"):
        assert autocast_float.dtypes == ("float32",)
        with autocast_float_as("float64"):
            assert autocast_float.dtypes == ("float64",)
        assert autocast_float.dtypes == ("float32",)
    assert autocast_float.dtypes == orig_autocast

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float32"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == "float32"
        assert (fvector() + _asarray(1.1, dtype="float64")).dtype == "float64"
        assert (fvector() + _asarray(1.1, dtype="float32")).dtype == "float32"

        assert (dvector() + 1).dtype == "float64"
        assert (fvector() + 1).dtype == "float32"

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float64"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.0).dtype == "float64"
        assert (fvector() + _asarray(1.1, dtype="float64")).dtype == "float64"
        assert (fvector() + _asarray(1.1, dtype="float32")).dtype == "float32"

        assert (dvector() + 1).dtype == "float64"
        assert (fvector() + 1).dtype == "float32"

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float32", "float64"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == config.floatX
        assert (fvector() + 1.0).dtype == "float32"
        assert (dvector() + np.float32(1.1)).dtype == "float64"
        assert (dvector() + np.float64(1.1)).dtype == "float64"
        assert (dvector() + np.float(1.1)).dtype == "float64"
        assert (fvector() + np.float32(1.1)).dtype == "float32"
        assert (fvector() + np.float64(1.1)).dtype == "float64"
        assert (fvector() + np.float(1.1)).dtype == config.floatX
        assert (lvector() + np.int64(1)).dtype == "int64"
        assert (lvector() + np.int32(1)).dtype == "int64"
        assert (lvector() + np.int16(1)).dtype == "int64"
        assert (lvector() + np.int8(1)).dtype == "int64"
        assert (ivector() + np.int8(1)).dtype == "int32"
        assert (wvector() + np.int8(1)).dtype == "int16"
        assert (bvector() + np.int8(1)).dtype == "int8"
        with autocast_float_as("float64"):
            assert (fvector() + 1.0).dtype == "float64"


def _test_autocast_numpy():
    # Called from `test_autocast`.
    assert config.cast_policy == "numpy"
    # Go through some typical scalar values.

    def ok(z):
        assert constant(z).dtype == np.asarray(z).dtype

    for x in (
        [2 ** i for i in range(63)] + [0, 0, 1, 2 ** 63 - 1] + [0.0, 1.0, 1.1, 1.5]
    ):
        n_x = np.asarray(x)
        # Make sure the data type is the same as the one found by numpy.
        ok(x)
        ok(-x)
        ok(x - 1)
        ok(-x + 1)
        ok(n_x)


def _test_autocast_numpy_floatX():
    # Called from `test_autocast`.
    assert config.cast_policy == "numpy+floatX"

    def ok(z, floatX):
        if isinstance(z, float) and floatX == "float32" and not hasattr(z, "dtype"):
            # Special case where we use 'float32' instead of 'float64'.
            assert constant(z).dtype == "float32"
        else:
            assert constant(z).dtype == np.asarray(z).dtype

    # Test with various values of `config.floatX`.
    for floatX in ("float32", "float64"):
        # Go through some typical scalar values.
        # We only consider 'int' and 'long' Python values that can fit
        # into int64, as that is the maximal integer type that Aesara
        # supports, and that is the maximal type in Python indexing.
        for x in (
            [2 ** i - 1 for i in range(64)]
            + [0, 0, 1, 2 ** 63 - 1]
            + [0.0, 1.0, 1.1, 1.5]
        ):
            with config.change_flags(floatX=floatX):
                ok(x, floatX)
                ok(-x, floatX)
                ok(x - 1, floatX)
                ok(-x + 1, floatX)
                ok(np.asarray(x), floatX)
                ok(np.float64(x), floatX)


class TestLongTensor:
    def test_fit_int64(self):
        bitwidth = PYTHON_INT_BITWIDTH
        for exponent in range(bitwidth):
            val = 2 ** exponent - 1
            scalar_ct = constant(val)

            assert scalar_ct.dtype in int_dtypes, (
                exponent,
                val,
                scalar_ct.dtype,
            )
            assert scalar_ct.value == val

            vector_ct = constant([val, val])
            # On Python 2, np.array() on a "long" returns int64,
            # but on Python 3, all integers are long, and np.asarray
            # will not force the upcasting, and return the native int width.
            if bitwidth == 32:
                assert vector_ct.dtype == "int32"
            else:
                assert vector_ct.dtype == "int64"
            assert np.all(vector_ct.value == val)

            matrix_ct = constant([[val, val]])
            # On Python 2, np.array() on a "long" returns int64,
            # but on Python 3, all integers are long, and np.asarray
            # will not force the upcasting, and return the native int width.
            if bitwidth == 32:
                assert matrix_ct.dtype == "int32"
            else:
                assert matrix_ct.dtype == "int64"
            assert np.all(matrix_ct.value == val)

    def test_too_big(self):
        val = 2 ** 64
        # This fail for all NumPy version.
        with pytest.raises(Exception):
            constant(val)
        with pytest.raises(Exception):
            constant()[val, val]
        with pytest.raises(Exception):
            constant()[[val, val]]


class TestBroadcast:
    def test_broadcast_bigdim(self):
        def f():
            x = matrix()
            addbroadcast(x, 2)

        with pytest.raises(ValueError):
            f()

    def test_unbroadcast_addbroadcast(self):
        # test that the unbroadcast fct don't insert not needed broadcast
        # and fuse consecutive Rebroadcast op

        x = matrix()
        assert unbroadcast(x, 0) is x
        assert unbroadcast(x, 1) is x
        assert unbroadcast(x, 1, 0) is x
        assert unbroadcast(x, 0, 1) is x

        assert addbroadcast(x, 0) is not x
        assert addbroadcast(x, 1) is not x
        assert addbroadcast(x, 1, 0).owner.inputs[0] is x

        assert unbroadcast(addbroadcast(x, 0), 0) is x
        assert addbroadcast(unbroadcast(x, 0), 0) is not x
        x = row()
        assert unbroadcast(x, 0) is not x
        assert unbroadcast(x, 1) is x
        assert unbroadcast(x, 1, 0) is not x
        assert unbroadcast(x, 0, 1) is not x

        assert addbroadcast(x, 0) is x
        assert addbroadcast(x, 1).owner.inputs[0] is x
        assert addbroadcast(x, 1, 0).owner.inputs[0] is x
        assert addbroadcast(x, 0, 1).owner.inputs[0] is x

        assert unbroadcast(addbroadcast(x, 1), 1) is x
        assert addbroadcast(unbroadcast(x, 1), 1) is not x

        # The first broadcast is remove the broadcast, so the second
        # should not make one
        assert unbroadcast(unbroadcast(x, 0), 0).owner.inputs[0] is x

        # Test that consecutive Rebroadcast op are fused
        x = TensorType(dtype="float64", broadcastable=(True, True))()
        assert unbroadcast(unbroadcast(x, 1), 0).owner.inputs[0] is x
        assert addbroadcast(unbroadcast(x, 1), 0).owner.inputs[0] is x
        assert addbroadcast(unbroadcast(x, 0), 0) is x

    def test_patternbroadcast(self):
        # Test that patternbroadcast with an empty broadcasting pattern works
        x = scalar("x")
        m = matrix("m")
        s = patternbroadcast(m, x.broadcastable)
        assert s is m
        x2 = patternbroadcast(x, x.broadcastable)
        assert x2 is x

    def test_infer_shape(self):
        x = matrix()
        y = addbroadcast(x, 0)
        f = aesara.function([x], y.shape)
        assert (f(np.zeros((1, 5), dtype=config.floatX)) == [1, 5]).all()
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert len(topo) == 2
            assert isinstance(topo[0].op, Shape_i)
            assert isinstance(topo[1].op, MakeVector)

        x = matrix()
        y = unbroadcast(x, 0)
        f = aesara.function([x], y.shape)
        assert (f(np.zeros((2, 5), dtype=config.floatX)) == [2, 5]).all()
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert len(topo) == 3
            assert isinstance(topo[0].op, Shape_i)
            assert isinstance(topo[1].op, Shape_i)
            assert isinstance(topo[2].op, MakeVector)

        x = row()
        y = unbroadcast(x, 0)
        f = aesara.function([x], y.shape)
        assert (f(np.zeros((1, 5), dtype=config.floatX)) == [1, 5]).all()
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert len(topo) == 2
            assert isinstance(topo[0].op, Shape_i)
            assert isinstance(topo[1].op, MakeVector)


class TestRebroadcast(utt.InferShapeTester):
    def test_rebroadcast(self):
        rng = np.random.default_rng(3453)
        # Rebroadcast
        adtens4 = dtensor4()
        adict = [(0, False), (1, True), (2, False), (3, True)]
        adtens4_val = rng.random((2, 1, 3, 1)).astype(config.floatX)
        self._compile_and_check(
            [adtens4],
            [Rebroadcast(*adict)(adtens4)],
            [adtens4_val],
            Rebroadcast,
            warn=False,
        )

        adtens4_bro = TensorType("float64", (True, True, True, False))()
        bdict = [(0, True), (1, False), (2, False), (3, False)]
        adtens4_bro_val = rng.random((1, 1, 1, 3)).astype(config.floatX)
        self._compile_and_check(
            [adtens4_bro],
            [Rebroadcast(*bdict)(adtens4_bro)],
            [adtens4_bro_val],
            Rebroadcast,
        )


def test_len():
    for shape_ in [(5,), (3, 4), (7, 4, 6)]:
        x = tensor(dtype="floatX", broadcastable=(False,) * len(shape_))
        with pytest.raises(TypeError):
            len(x)


def test_unalign():
    if config.floatX == "float64":
        dtype = "b1,f8"
    else:
        dtype = "b1,f4"

    a = np.empty(10000, dtype=dtype)["f1"]
    b = np.empty(10000, dtype=dtype)["f1"]
    assert not a.flags.aligned
    assert not b.flags.aligned
    a[:] = random(len(a))
    b[:] = random(len(b))
    # out_numpy = 2 * a + 3 * b

    av, bv = vectors("ab")
    f = aesara.function([av, bv], 2 * av + 3 * bv)
    f.maker.fgraph.toposort()

    with pytest.raises(TypeError):
        f(a, b)

    a = np.empty((), dtype=dtype)["f1"]
    b = np.empty((), dtype=dtype)["f1"]
    assert not a.flags.aligned
    assert not b.flags.aligned
    # out_numpy = 2 * a + 3 * b

    av, bv = scalars("ab")
    f = aesara.function([av, bv], 2 * av + 3 * bv)
    f.maker.fgraph.toposort()
    with pytest.raises(TypeError):
        f(a, b)


def test_dimshuffle_duplicate():
    x = vector()
    with pytest.raises(ValueError, match="may not appear twice"):
        DimShuffle((False,), (0, 0))(x)


class TestGetScalarConstantValue:
    def test_basic(self):
        a = aet.stack([1, 2, 3])
        assert get_scalar_constant_value(a[0]) == 1
        assert get_scalar_constant_value(a[1]) == 2
        assert get_scalar_constant_value(a[2]) == 3

        b = iscalar()
        a = aet.stack([b, 2, 3])
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(a[0])
        assert get_scalar_constant_value(a[1]) == 2
        assert get_scalar_constant_value(a[2]) == 3

        # For now get_scalar_constant_value goes through only MakeVector and Join of
        # scalars.
        v = ivector()
        a = aet.stack([v, [2], [3]])
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(a[0])
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(a[1])
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(a[2])

        # Test the case SubTensor(Shape(v)) when the dimensions
        # is broadcastable.
        v = row()
        assert get_scalar_constant_value(v.shape[0]) == 1

        res = aet.get_scalar_constant_value(aet.as_tensor([10, 20]).shape[0])
        assert isinstance(res, np.ndarray)
        assert 2 == res

        res = aet.get_scalar_constant_value(
            9 + aet.as_tensor([1.0]).shape[0],
            elemwise=True,
            only_process_constants=False,
            max_recur=9,
        )
        assert isinstance(res, np.ndarray)
        assert 10 == res

    def test_subtensor_of_constant(self):
        c = constant(random(5))
        for i in range(c.value.shape[0]):
            assert get_scalar_constant_value(c[i]) == c.value[i]
        c = constant(random(5, 5))
        for i in range(c.value.shape[0]):
            for j in range(c.value.shape[1]):
                assert get_scalar_constant_value(c[i, j]) == c.value[i, j]

    def test_numpy_array(self):
        # Regression test for crash when called on a numpy array.
        assert get_scalar_constant_value(np.array(3)) == 3
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(np.array([0, 1]))
        with pytest.raises(EmptyConstantError):
            get_scalar_constant_value(np.array([]))

    def test_make_vector(self):
        mv = make_vector(1, 2, 3)
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(mv)
        assert get_scalar_constant_value(mv[0]) == 1
        assert get_scalar_constant_value(mv[1]) == 2
        assert get_scalar_constant_value(mv[2]) == 3
        assert get_scalar_constant_value(mv[np.int32(0)]) == 1
        assert get_scalar_constant_value(mv[np.int64(1)]) == 2
        assert get_scalar_constant_value(mv[np.uint(2)]) == 3
        t = aes.Scalar("int64")
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(mv[t()])

    def test_shape_i(self):
        c = constant(np.random.random((3, 4)))
        s = Shape_i(0)(c)
        assert get_scalar_constant_value(s) == 3
        s = Shape_i(1)(c)
        assert get_scalar_constant_value(s) == 4
        d = aesara.shared(np.random.standard_normal((1, 1)), broadcastable=(True, True))
        f = ScalarFromTensor()(Shape_i(0)(d))
        assert get_scalar_constant_value(f) == 1

    def test_elemwise(self):
        # We test only for a few elemwise, the list of all supported
        # elemwise are in the fct.
        c = constant(np.random.random())
        s = c + 1
        assert np.allclose(get_scalar_constant_value(s), c.data + 1)
        s = c - 1
        assert np.allclose(get_scalar_constant_value(s), c.data - 1)
        s = c * 1.2
        assert np.allclose(get_scalar_constant_value(s), c.data * 1.2)
        s = c < 0.5
        assert np.allclose(get_scalar_constant_value(s), int(c.data < 0.5))
        s = aet.second(c, 0.4)
        assert np.allclose(get_scalar_constant_value(s), 0.4)

    def test_assert(self):
        # Make sure we still get the constant value if it is wrapped in
        # an Assert.
        c = constant(2)
        x = scalar()

        # condition is always True
        a = Assert()(c, c > 1)
        assert get_scalar_constant_value(a) == 2

        with config.change_flags(compute_test_value="off"):
            # condition is always False
            a = Assert()(c, c > 2)
            with pytest.raises(NotScalarConstantError):
                get_scalar_constant_value(a)

        # condition is not constant
        a = Assert()(c, c > x)
        with pytest.raises(NotScalarConstantError):
            get_scalar_constant_value(a)

    def test_second(self):
        # Second should apply when the value is constant but not the shape
        c = constant(np.random.random())
        shp = vector()
        s = aet.second(shp, c)
        assert get_scalar_constant_value(s) == c.data

    def test_copy(self):
        # Make sure we do not return the internal storage of a constant,
        # so we cannot change the value of a constant by mistake.
        c = constant(3)
        d = extract_constant(c)
        d += 1
        e = extract_constant(c)
        assert e == 3, (c, d, e)


def test_complex_mod_failure():
    # Make sure % fails on complex numbers.
    x = vector(dtype="complex64")
    with pytest.raises(aes.ComplexError):
        x % 5


class TestSize:
    # Ensure the `size` attribute of tensors behaves as in numpy.
    def test_matrix(self):
        x = matrix()
        y = np.zeros((5, 7), dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_vector(self):
        x = vector()
        y = np.zeros(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_scalar(self):
        x = scalar()
        y = np.array(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_shared(self):
        # NB: we also test higher order tensors at the same time.
        y = np.zeros((1, 2, 3, 4), dtype=config.floatX)
        x = aesara.shared(y)
        assert y.size == function([], x.size)()


class TestDiag:
    """
    Test that linalg.diag has the same behavior as numpy.diag.
    numpy.diag has two behaviors:
    (1) when given a vector, it returns a matrix with that vector as the
    diagonal.
    (2) when given a matrix, returns a vector which is the diagonal of the
    matrix.

    (1) and (2) are tested by test_alloc_diag and test_extract_diag
    respectively.

    test_diag test makes sure that linalg.diag instantiates
    the right op based on the dimension of the input.
    """

    def setup_method(self):
        self.mode = None
        self.shared = shared
        self.floatX = config.floatX
        self.type = TensorType

    def test_diag(self):
        rng = np.random.default_rng(utt.fetch_seed())

        # test vector input
        x = vector()
        g = diag(x)
        assert isinstance(g.owner.op, AllocDiag)
        f = aesara.function([x], g)
        for shp in [5, 0, 1]:
            m = rng.random(shp).astype(self.floatX)
            v = np.diag(m)
            r = f(m)
            # The right matrix is created
            assert (r == v).all()

        # Test matrix input
        xx = self.shared(rng.random((3, 5)))
        g = diag(xx)
        assert isinstance(g.owner.op, ExtractDiag)
        f = aesara.function([], g)
        for shp in [(5, 3), (3, 5), (5, 1), (1, 5), (5, 0), (0, 5), (1, 0), (0, 1)]:
            m = rng.random(shp).astype(self.floatX)
            xx.set_value(m)
            v = np.diag(m)
            r = f()
            # The right matrix is created
            assert (r == v).all()

        # Test scalar input
        xx = scalar()
        with pytest.raises(ValueError):
            diag(xx)

    def test_infer_shape(self):
        rng = np.random.default_rng(utt.fetch_seed())

        x = vector()
        g = diag(x)
        f = aesara.function([x], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert np.sum([isinstance(node.op, AllocDiag) for node in topo]) == 0
        for shp in [5, 0, 1]:
            m = rng.random(shp).astype(self.floatX)
            assert (f(m) == np.diag(m).shape).all()

        x = matrix()
        g = diag(x)
        f = aesara.function([x], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert np.sum([isinstance(node.op, ExtractDiag) for node in topo]) == 0
        for shp in [(5, 3), (3, 5), (5, 1), (1, 5), (5, 0), (0, 5), (1, 0), (0, 1)]:
            m = rng.random(shp).astype(self.floatX)
            assert (f(m) == np.diag(m).shape).all()

    def test_diag_grad(self):
        rng = np.random.default_rng(utt.fetch_seed())
        x = rng.random((5))
        utt.verify_grad(diag, [x], rng=rng)
        x = rng.random((5, 3))
        utt.verify_grad(diag, [x], rng=rng)


class TestAllocDiag:
    def setup_method(self):
        self.alloc_diag = AllocDiag
        self.mode = aesara.compile.mode.get_default_mode()

    def _generator(self):
        dims = 4
        shape = (5,) * dims
        xv = np.random.standard_normal(shape).astype(config.floatX)
        for d in range(1, dims + 1):
            # Create a TensorType of the same dimensions as
            # as the data we want to test.
            x = TensorType(dtype=config.floatX, broadcastable=(False,) * d)("x")

            # Make a slice of the test data that has the
            # dimensions we need by doing xv[0,...,0]
            # For example, for an array of shape (5,), we
            # need to do xv[0, 0, 0, 0].
            test_val = xv[((0,) * (dims - d))]
            yield x, test_val

    def test_alloc_diag_values(self):
        for x, test_val in self._generator():
            for offset, axis1, axis2 in [
                (0, 0, 1),
                (0, 1, 2),
                (1, 0, 1),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
                (-1, 0, 1),
                (-2, 0, 1),
                (-1, 1, 2),
            ]:
                # Test AllocDiag values
                if np.maximum(axis1, axis2) > len(test_val.shape):
                    continue
                adiag_op = self.alloc_diag(offset=offset, axis1=axis1, axis2=axis2)
                f = aesara.function([x], adiag_op(x))
                # AllocDiag and extract the diagonal again
                # to check
                diag_arr = f(test_val)
                rediag = np.diagonal(diag_arr, offset=offset, axis1=axis1, axis2=axis2)
                assert np.all(rediag == test_val)

                # Test infer_shape
                f_shape = aesara.function([x], adiag_op(x).shape, mode="FAST_RUN")

                aesara.printing.debugprint(f_shape.maker.fgraph.outputs[0])
                output_shape = f_shape(test_val)
                assert not any(
                    isinstance(node.op, self.alloc_diag)
                    for node in f_shape.maker.fgraph.toposort()
                )
                rediag_shape = np.diagonal(
                    np.ones(output_shape), offset=offset, axis1=axis1, axis2=axis2
                ).shape
                assert np.all(rediag_shape == test_val.shape)

                diag_x = adiag_op(x)
                sum_diag_x = aet_sum(diag_x)
                grad_x = aesara.grad(sum_diag_x, x)
                grad_diag_x = aesara.grad(sum_diag_x, diag_x)
                f_grad_x = aesara.function([x], grad_x, mode=self.mode)
                f_grad_diag_x = aesara.function([x], grad_diag_x, mode=self.mode)
                grad_input = f_grad_x(test_val)
                grad_diag_input = f_grad_diag_x(test_val)
                true_grad_input = np.diagonal(
                    grad_diag_input, offset=offset, axis1=axis1, axis2=axis2
                )

                assert np.all(true_grad_input == grad_input)


class TestNumpyAssumptions:
    # Verify that some assumptions Aesara makes on Numpy's behavior still hold.
    def test_ndarray_copy(self):
        # A copy or deepcopy of the ndarray type should not create a new object.
        #
        # This is because Aesara makes some comparisons of the form:
        #     if type(x) is np.ndarray
        assert copy(np.ndarray) is np.ndarray
        assert deepcopy(np.ndarray) is np.ndarray

    def test_dtype_equality(self):
        # Ensure dtype string comparisons are consistent.
        #
        # Aesara often uses string representations of dtypes (e.g. 'float32'). We
        # need to make sure that comparing the string representations is the same
        # as comparing the dtype objects themselves.
        dtypes = get_numeric_types(with_complex=True)
        # Perform all pairwise comparisons of dtypes, making sure comparing
        # their string representation yields the same result.
        for dtype1_idx, dtype1 in enumerate(dtypes):
            for dtype2 in dtypes[dtype1_idx + 1 :]:
                assert (dtype1 == dtype2) == (str(dtype1) == str(dtype2))


def test_transpose():
    x1 = dvector("x1")
    x2 = dmatrix("x2")
    x3 = dtensor3("x3")

    x1v = np.arange(24)
    x2v = np.arange(24).reshape(2, 12)
    x3v = np.arange(24).reshape(2, 3, 4)

    f = aesara.function(
        [x1, x2, x3],
        [
            aet.transpose(x1),
            aet.transpose(x2),
            aet.transpose(x3),
            x1.transpose(),
            x2.transpose(),
            x3.transpose(),
            x2.transpose(0, 1),
            x3.transpose((0, 2, 1)),
            aet.transpose(x2, [0, 1]),
            aet.transpose(x3, [0, 2, 1]),
        ],
    )

    t1, t2, t3, t1b, t2b, t3b, t2c, t3c, t2d, t3d = f(x1v, x2v, x3v)
    assert t1.shape == np.transpose(x1v).shape
    assert t2.shape == np.transpose(x2v).shape
    assert t3.shape == np.transpose(x3v).shape
    assert np.all(t1 == np.transpose(x1v))
    assert np.all(t2 == np.transpose(x2v))
    assert np.all(t3 == np.transpose(x3v))
    assert np.all(t1b == x1v.transpose())
    assert np.all(t2b == x2v.transpose())
    assert np.all(t3b == x3v.transpose())
    assert t2c.shape == (2, 12)
    assert t3c.shape == (2, 4, 3)
    assert np.all(t2c == x2v.transpose([0, 1]))
    assert np.all(t3c == x3v.transpose([0, 2, 1]))
    assert t2d.shape == (2, 12)
    assert t3d.shape == (2, 4, 3)
    assert np.all(t2d == np.transpose(x2v, [0, 1]))
    assert np.all(t3d == np.transpose(x3v, [0, 2, 1]))

    # Check that we create a name.
    assert aet.transpose(x1).name == "x1.T"
    assert aet.transpose(x2).name == "x2.T"
    assert aet.transpose(x3).name == "x3.T"
    assert aet.transpose(dmatrix()).name is None


def test_stacklists():
    a, b, c, d = map(scalar, "abcd")
    X = stacklists([[a, b], [c, d]])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (2, 2)
    assert np.allclose(f(1, 2, 3, 4), np.asarray([[1, 2], [3, 4]]))

    X = stacklists([a, b, c, d])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (4,)
    assert np.allclose(f(1, 2, 3, 4), np.asarray([[1, 2, 3, 4]]))

    X = stacklists([[[a], [b]], [[c], [d]]])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (2, 2, 1)

    a, b, c, d = [matrix(x) for x in "abcd"]
    X = stacklists([[a, b], [c, d]])
    f = function([a, b, c, d], X)
    x = np.ones((4, 4), "float32")
    assert f(x, x, x, x).shape == (2, 2, 4, 4)


class TestInferShape(utt.InferShapeTester):
    def test_Flatten(self):
        atens3 = tensor3()
        atens3_val = random(4, 5, 3)
        for ndim in (3, 2, 1):
            self._compile_and_check(
                [atens3],
                [flatten(atens3, ndim)],
                [atens3_val],
                Reshape,
                excluding=["local_useless_reshape"],
            )

        amat = matrix()
        amat_val = random(4, 5)
        for ndim in (2, 1):
            self._compile_and_check(
                [amat],
                [flatten(amat, ndim)],
                [amat_val],
                Reshape,
                excluding=["local_useless_reshape"],
            )

        avec = vector()
        avec_val = random(4)
        ndim = 1
        self._compile_and_check(
            [avec],
            [flatten(avec, ndim)],
            [avec_val],
            Reshape,
            excluding=["local_useless_reshape"],
        )

    def test_Eye(self):
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [4, 4, 0], Eye
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [4, 5, 0], Eye
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [3, 5, 0], Eye
        )

    def test_Tri(self):
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [4, 4, 0], Tri
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [4, 5, 0], Tri
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [3, 5, 0], Tri
        )

    def test_ExtractDiag(self):
        atens3 = tensor3()
        atens3_val = random(4, 5, 3)
        atens3_diag = ExtractDiag()(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(-1)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 0, 2)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 1, 2)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 2, 0)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)

    def test_AllocDiag(self):
        advec = dvector()
        advec_val = random(4)
        self._compile_and_check([advec], [AllocDiag()(advec)], [advec_val], AllocDiag)

        # Shape
        # 'opt.Makevector' precludes optimizer from disentangling
        # elements of shape
        adtens = tensor3()
        adtens_val = random(4, 5, 3)
        self._compile_and_check(
            [adtens], [Shape()(adtens)], [adtens_val], (MakeVector, Shape)
        )

    def test_Split(self):
        aiscal = iscalar()
        aivec = ivector()
        adtens = tensor3()
        adtens_val = random(4, 10, 3)
        aivec_val = [2, 5, 3]
        for aiscal_val in [1, -2]:
            self._compile_and_check(
                [adtens, aiscal, aivec],
                [Split(3)(adtens, aiscal, aivec)[0]],
                [adtens_val, aiscal_val, aivec_val],
                (Split),
            )

    def test_Join(self):
        aiscal = iscalar()
        cdmat = dmatrix()
        admat_val = random(1, 3)
        bdmat_val = random(2, 3)
        cdmat_val = random(4, 3)
        admat = dmatrix()
        bdmat = dmatrix()
        for aiscal_val in [0, -2]:
            self._compile_and_check(
                [aiscal, admat, bdmat, cdmat],
                [Join()(aiscal, admat, bdmat, cdmat)],
                [aiscal_val, admat_val, bdmat_val, cdmat_val],
                Join,
            )

        admat_val = random(4, 1)
        bdmat_val = random(4, 3)
        cdmat_val = random(4, 2)
        for aiscal_val in [-1, 1]:
            self._compile_and_check(
                [aiscal, admat, bdmat, cdmat],
                [Join()(aiscal, admat, bdmat, cdmat)],
                [aiscal_val, admat_val, bdmat_val, cdmat_val],
                Join,
            )

    def test_PermuteRowElements(self):
        admat = dmatrix()
        advec = dvector()
        aivec = ivector()

        abool = True
        rng = np.random.default_rng(utt.fetch_seed())
        advec_val = random(5)
        aivec_val = rng.permutation(5).astype("int32")
        self._compile_and_check(
            [advec, aivec],
            [PermuteRowElements()(advec, aivec, abool)],
            [advec_val, aivec_val],
            PermuteRowElements,
        )

        admat_val = random(3, 5)
        self._compile_and_check(
            [admat, aivec],
            [PermuteRowElements()(admat, aivec, abool)],
            [admat_val, aivec_val],
            PermuteRowElements,
        )

        adtens3 = dtensor3()
        adtens3_val = random(3, 2, 5)
        self._compile_and_check(
            [adtens3, aivec],
            [PermuteRowElements()(adtens3, aivec, abool)],
            [adtens3_val, aivec_val],
            PermuteRowElements,
        )

        aimat = imatrix()
        perma = rng.permutation(5).astype("int32")
        permb = rng.permutation(5).astype("int32")
        permc = rng.permutation(5).astype("int32")
        aimat_val = np.vstack((perma, permb, permc))
        admat_val = random(3, 5)
        self._compile_and_check(
            [admat, aimat],
            [PermuteRowElements()(admat, aimat, abool)],
            [admat_val, aimat_val],
            PermuteRowElements,
        )

        aitens3 = itensor3()
        perma = rng.permutation(5).astype("int32")
        permb = rng.permutation(5).astype("int32")
        permc = rng.permutation(5).astype("int32")
        bimat_val = np.vstack((perma, permb, permc))
        aitens3_val = np.empty((2, 3, 5), "int32")
        aitens3_val[0, ::, ::] = aimat_val
        aitens3_val[1, ::, ::] = bimat_val
        self._compile_and_check(
            [admat, aitens3],
            [PermuteRowElements()(admat, aitens3, abool)],
            [admat_val, aitens3_val],
            PermuteRowElements,
        )

    def test_ScalarFromTensor(self):
        aiscal = iscalar()
        self._compile_and_check(
            [aiscal],
            [TensorFromScalar()(ScalarFromTensor()(aiscal))],
            [45],
            ScalarFromTensor,
            excluding=["local_tensor_scalar_tensor"],
        )

    def test_TensorFromScalar(self):
        aiscal = aes.float64()

        self._compile_and_check(
            [aiscal], [TensorFromScalar()(aiscal)], [4.0], TensorFromScalar
        )

    def test_Alloc(self):
        integers = np.random.default_rng(utt.fetch_seed()).integers
        adscal = dscalar()
        aiscal = lscalar()
        biscal = lscalar()
        ciscal = lscalar()
        discal = lscalar()
        adscal_val = random()
        aiscal_val = integers(3, 6, size=())
        biscal_val = integers(3, 6, size=())
        ciscal_val = integers(3, 6, size=())
        discal_val = integers(3, 6, size=())
        self._compile_and_check(
            [adscal, aiscal, biscal, ciscal, discal],
            [Alloc()(adscal, aiscal, biscal, ciscal, discal)],
            [adscal_val, aiscal_val, biscal_val, ciscal_val, discal_val],
            Alloc,
        )

    def test_ARange(self):
        aiscal = lscalar()
        biscal = lscalar()
        ciscal = lscalar()

        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [0, 5, 1],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [2, 11, 4],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [-5, 1, 1],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [10, 2, -2],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [10, 2, 2],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [0, 0, 1],
            ARange,
        )

    def test_Tile(self):
        # Tile op is deprecated so the tile function doesn't use it
        # anymore, we'll test here the op directly
        advec = dvector()
        advec_val = random(5)
        aivec_val = [3]
        ndim = 1
        self._compile_and_check(
            [advec], [Tile(ndim)(advec, aivec_val)], [advec_val], Tile
        )

        admat = dmatrix()
        admat_val = random(2, 4)
        aivec_val = [2, 3]
        ndim = 2
        self._compile_and_check(
            [admat], [Tile(ndim)(admat, aivec_val)], [admat_val], Tile
        )

        adtens4 = dtensor4()
        adtens4_val = random(2, 4, 3, 5)
        aivec_val = [2, 3, 1, 4]
        ndim = 4
        self._compile_and_check(
            [adtens4], [Tile(ndim)(adtens4, aivec_val)], [adtens4_val], Tile
        )


class TestTensorInstanceMethods:
    def setup_method(self):
        self.vars = matrices("X", "Y")
        self.vals = [m.astype(config.floatX) for m in [random(2, 2), random(2, 2)]]

    def test_repeat(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.repeat(2).eval({X: x}), x.repeat(2))

    def test_trace(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.trace().eval({X: x}), x.trace())

    def test_ravel(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.ravel().eval({X: x}), x.ravel())

    def test_diagonal(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.diagonal().eval({X: x}), x.diagonal())
        assert_array_equal(X.diagonal(1).eval({X: x}), x.diagonal(1))
        assert_array_equal(X.diagonal(-1).eval({X: x}), x.diagonal(-1))
        for offset, axis1, axis2 in [(1, 0, 1), (-1, 0, 1), (0, 1, 0), (-2, 1, 0)]:
            assert_array_equal(
                X.diagonal(offset, axis1, axis2).eval({X: x}),
                x.diagonal(offset, axis1, axis2),
            )

    def test_take(self):
        X, _ = self.vars
        x, _ = self.vals
        indices = [1, 0, 3]
        assert_array_equal(X.take(indices).eval({X: x}), x.take(indices))
        indices = [1, 0, 1]
        assert_array_equal(X.take(indices, 1).eval({X: x}), x.take(indices, 1))
        indices = np.array([-10, 5, 12], dtype="int32")
        assert_array_equal(
            X.take(indices, 1, mode="wrap").eval({X: x}),
            x.take(indices, 1, mode="wrap"),
        )
        assert_array_equal(
            X.take(indices, -1, mode="wrap").eval({X: x}),
            x.take(indices, -1, mode="wrap"),
        )
        assert_array_equal(
            X.take(indices, 1, mode="clip").eval({X: x}),
            x.take(indices, 1, mode="clip"),
        )
        assert_array_equal(
            X.take(indices, -1, mode="clip").eval({X: x}),
            x.take(indices, -1, mode="clip"),
        )
        # Test error handling
        with pytest.raises(IndexError):
            X.take(indices).eval({X: x})
        with pytest.raises(IndexError):
            (2 * X.take(indices)).eval({X: x})
        with pytest.raises(TypeError):
            X.take([0.0])
        indices = [[1, 0, 1], [0, 1, 1]]
        assert_array_equal(X.take(indices, 1).eval({X: x}), x.take(indices, 1))
        # Test equivalent advanced indexing
        assert_array_equal(X[:, indices].eval({X: x}), x[:, indices])


class TestSwapaxes:
    def test_no_dimensional_input(self):
        with pytest.raises(IndexError):
            swapaxes(2, 0, 1)

    def test_unidimensional_input(self):
        with pytest.raises(IndexError):
            swapaxes([2, 1], 0, 1)

    def test_not_enough_dimension(self):
        with pytest.raises(IndexError):
            swapaxes([[2, 1], [3, 4]], 3, 4)

    def test_doubleswap(self):
        y = matrix()
        n = swapaxes(y, 0, 1)
        f = function([y], n)
        testMatrix = [[2, 1], [3, 4]]
        assert np.array_equal(testMatrix, f(f(testMatrix)))

    def test_interface(self):
        x = matrix()
        x.swapaxes(0, 1)

    def test_numpy_compare(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix("A", dtype=config.floatX)
        Q = swapaxes(A, 0, 1)
        fn = function([A], [Q])
        a = rng.random((4, 4)).astype(config.floatX)

        n_s = np.swapaxes(a, 0, 1)
        t_s = fn(a)
        assert np.allclose(n_s, t_s)


class TestChoose(utt.InferShapeTester):
    op = staticmethod(choose)
    op_class = Choose
    modes = ["raise", "wrap", "clip"]
    rng = np.random.default_rng(utt.fetch_seed())

    def test_numpy_compare(self):

        a = vector(dtype="int32")
        b = matrix(dtype="float32")

        A = self.rng.integers(0, 4, 4).astype("int32")
        B = np.asarray(np.random.random((4, 4)), dtype="float32")

        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_method(self):
        a = vector(dtype="int32")
        b = matrix(dtype="float32")

        A = self.rng.integers(0, 4, 4).astype("int32")
        B = np.asarray(np.random.random((4, 4)), dtype="float32")

        for m in self.modes:
            f = function([a, b], a.choose(b, mode=m))
            t_c = f(A, B)
            n_c = A.choose(B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_broadcasted(self):
        a = scalar(dtype="int32")
        b = matrix(dtype="float32")

        # Test when a is broadcastable
        A = 3
        B = np.asarray(np.random.random((4, 4)), dtype="float32")

        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

        # Test when the result should be broadcastable
        b = col(dtype="float32")
        B = np.asarray(np.random.random((4, 1)), dtype="float32")
        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            assert choose(a, b, mode=m).broadcastable[0]
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_dtype_error(self):
        a = scalar(dtype="float32")
        b = matrix(dtype="float32")

        with pytest.raises(TypeError):
            choose(a, b)

    @pytest.mark.parametrize(
        "test_input",
        [
            (
                tensor3(dtype="int32"),
                tensor3(dtype="float32"),
                tensor3(dtype="float32"),
                rng.integers(0, 2, (2, 1, 1)).astype("int32"),
                np.asarray(np.random.random((1, 6, 1)), dtype="float32"),
                np.asarray(np.random.random((1, 1, 5)), dtype="float32"),
            ),
            (
                vector(dtype="int32"),
                scalar(),
                scalar(),
                [0, 1, 1, 0],
                0.1,
                0.2,
            ),
        ],
    )
    def test_numpy_compare_tuple(self, test_input):
        """Test with list and tuples of scalars and 3d tensors."""
        a, b, c, A, B, C = test_input
        for m in self.modes:
            for ls in [list, tuple]:
                f = function([a, b, c], choose(a, ls([b, c]), mode=m))
                t_c = f(A, B, C)
                n_c = np.choose(A, ls([B, C]), mode=m)
                assert np.allclose(t_c, n_c)

    def test_infer_shape(self):
        for shp1, shp2 in [
            ((5, 4), (7, 4)),
            ((1, 4), (7, 4)),
            ((5, 1), (7, 4)),
            ((5, 4), (1, 4)),
            ((5, 4), (7, 1)),
            ((5, 4), (4,)),
            ((1, 4), (4,)),
            ((5, 1), (4,)),
            ((5, 4), (1,)),
            ((4,), (5, 4)),
            ((1,), (5, 4)),
            ((4,), (1, 4)),
            ((4,), (3, 1)),
            ((4,), (4,)),
            ((1,), (4,)),
            ((4,), (1,)),
            ((1,), (1,)),
        ]:
            a = tensor(dtype="int32", broadcastable=[n == 1 for n in shp1])
            c = tensor(dtype="float32", broadcastable=[n == 1 for n in shp2])
            A = np.asarray(np.random.random(shp1) * shp2[0], dtype="int32")
            C = np.asarray(np.random.random(shp2) * shp2[0], dtype="float32")
            self._compile_and_check(
                [a, c],  # aesara.function inputs
                [self.op(a, c)],  # aesara.function outputs
                # Always use not square matrix!
                # inputs data
                [A, C],
                # Op that should be removed from the graph.
                self.op_class,
            )

    @pytest.mark.skip(reason="Not implemented")
    def test_infer_shape_tuple(self):

        a = tensor3(dtype="int32")
        b = tensor3(dtype="int32")
        c = tensor3(dtype="int32")

        A = np.asarray([1, 0], dtype="int32").reshape((2, 1, 1))
        B = np.asarray(np.random.random((1, 4, 1)), dtype="int32")
        C = np.asarray(np.random.random((1, 1, 7)), dtype="int32")

        f = function([a, b, c], choose(a, (b, c)))
        shape = (2, 4, 7)
        assert np.allclose(f(A, B, C).shape, shape)

        self._compile_and_check(
            [a, b, c],  # aesara.function inputs
            [self.op(a, (b, c))],  # aesara.function outputs
            # Always use not square matrix!
            # inputs data
            [A, B, C],
            # Op that should be removed from the graph.
            self.op_class,
        )


def test_allocempty():
    # Test that we allocated correctly
    f = aesara.function([], AllocEmpty("float32")(2, 3))
    assert len(f.maker.fgraph.apply_nodes) == 1
    out = f()

    assert out.shape == (2, 3)
    assert out.dtype == "float32"

    empty_at = aet.empty((2, 3), dtype=None)
    res = aesara.function([], empty_at)()
    assert res.shape == (2, 3)

    empty_at = aet.empty((2, 3), dtype="int64")
    res = aesara.function([], empty_at)()
    assert res.shape == (2, 3)
    assert res.dtype == "int64"


def test_atleast_Nd():
    ary1 = dscalar()
    res_ary1 = atleast_Nd(ary1, n=1)
    assert res_ary1.ndim == 1

    for n in range(1, 3):
        ary1, ary2 = dscalar(), dvector()
        res_ary1, res_ary2 = atleast_Nd(ary1, ary2, n=n)

        assert res_ary1.ndim == n
        if n == ary2.ndim:
            assert ary2 is res_ary2
        else:
            assert res_ary2.ndim == n

        ary1_val = np.array(1.0, dtype=np.float64)
        ary2_val = np.array([1.0, 2.0], dtype=np.float64)
        res_ary1_val, res_ary2_val = aesara.function(
            [ary1, ary2], [res_ary1, res_ary2]
        )(ary1_val, ary2_val)

        np_fn = np.atleast_1d if n == 1 else np.atleast_2d
        assert np.array_equal(res_ary1_val, np_fn(ary1_val))
        assert np.array_equal(res_ary2_val, np_fn(ary2_val))


def test_expand_dims():
    x_at = dscalar()
    res_at = expand_dims(x_at, 0)
    x_val = np.array(1.0, dtype=np.float64)
    exp_res = np.expand_dims(x_val, 0)
    res_val = aesara.function([x_at], res_at)(x_val)
    assert np.array_equal(exp_res, res_val)

    x_at = dscalar()
    res_at = expand_dims(x_at, (0, 1))
    x_val = np.array(1.0, dtype=np.float64)
    exp_res = np.expand_dims(x_val, (0, 1))
    res_val = aesara.function([x_at], res_at)(x_val)
    assert np.array_equal(exp_res, res_val)

    x_at = dmatrix()
    res_at = expand_dims(x_at, (2, 1))
    x_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    exp_res = np.expand_dims(x_val, (2, 1))
    res_val = aesara.function([x_at], res_at)(x_val)
    assert np.array_equal(exp_res, res_val)


class TestTakeAlongAxis:
    @pytest.mark.parametrize(
        ["shape", "axis", "samples"],
        (
            ((1,), None, 1),
            ((1,), -1, 10),
            ((3, 2, 1), -1, 1),
            ((3, 2, 1), 0, 10),
            ((3, 2, 1), -1, 10),
        ),
        ids=str,
    )
    def test_take_along_axis(self, shape, axis, samples):
        rng = np.random.default_rng()
        arr = rng.normal(size=shape).astype(config.floatX)
        indices_size = list(shape)
        indices_size[axis or 0] = samples
        indices = rng.integers(low=0, high=shape[axis or 0], size=indices_size)

        arr_in = aet.tensor(config.floatX, [s == 1 for s in arr.shape])
        indices_in = aet.tensor(np.int64, [s == 1 for s in indices.shape])

        out = aet.take_along_axis(arr_in, indices_in, axis)

        func = aesara.function([arr_in, indices_in], out)

        assert np.allclose(
            np.take_along_axis(arr, indices, axis=axis), func(arr, indices)
        )

    def test_ndim_dtype_failures(self):
        arr = aet.tensor(config.floatX, [False] * 2)
        indices = aet.tensor(np.int64, [False] * 3)
        with pytest.raises(ValueError):
            aet.take_along_axis(arr, indices)

        indices = aet.tensor(np.float64, [False] * 2)
        with pytest.raises(IndexError):
            aet.take_along_axis(arr, indices)


@pytest.mark.parametrize(
    "inp, shape",
    [(scalar, ()), (fvector, 3), (fmatrix, (3, 4))],
)
def test_full_like(inp, shape):
    fill_value = 5
    dtype = "float32"

    x = inp("x")
    y = full_like(x, fill_value, dtype=dtype)

    np.testing.assert_array_equal(
        y.eval({x: np.zeros(shape, dtype=dtype)}),
        np.full(shape, fill_value, dtype=dtype),
    )
