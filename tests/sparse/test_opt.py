import pytest


sp = pytest.importorskip("scipy", minversion="0.7.0")

import numpy as np

import aesara
from tests import unittest_tools as utt
from tests.sparse.test_basic import random_lil
from aesara import config, sparse, tensor


def test_local_csm_properties_csm():
    data = tensor.vector()
    indices, indptr, shape = (tensor.ivector(), tensor.ivector(), tensor.ivector())
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_csm_properties_csm")
    for CS, cast in [
        (sparse.CSC, sp.sparse.csc_matrix),
        (sparse.CSR, sp.sparse.csr_matrix),
    ]:
        f = aesara.function(
            [data, indices, indptr, shape],
            sparse.csm_properties(CS(data, indices, indptr, shape)),
            mode=mode,
        )
        assert not any(
            isinstance(node.op, (sparse.CSM, sparse.CSMProperties))
            for node in f.maker.fgraph.toposort()
        )
        v = cast(random_lil((10, 40), config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


@pytest.mark.skip(reason="Opt disabled as it don't support unsorted indices")
@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_csm_grad_c():
    data = tensor.vector()
    indices, indptr, shape = (tensor.ivector(), tensor.ivector(), tensor.ivector())
    mode = aesara.compile.mode.get_default_mode()

    if aesara.config.mode == "FAST_COMPILE":
        mode = aesara.compile.Mode(linker="c|py", optimizer="fast_compile")

    mode = mode.including("specialize", "local_csm_grad_c")
    for CS, cast in [
        (sparse.CSC, sp.sparse.csc_matrix),
        (sparse.CSR, sp.sparse.csr_matrix),
    ]:
        cost = tensor.sum(sparse.DenseFromSparse()(CS(data, indices, indptr, shape)))
        f = aesara.function(
            [data, indices, indptr, shape], tensor.grad(cost, data), mode=mode
        )
        assert not any(
            isinstance(node.op, sparse.CSMGrad) for node in f.maker.fgraph.toposort()
        )
        v = cast(random_lil((10, 40), config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_mul_s_d():
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_d")

    for sp_format in sparse.sparse_formats:
        inputs = [getattr(aesara.sparse, sp_format + "_matrix")(), tensor.matrix()]

        f = aesara.function(inputs, sparse.mul_s_d(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.MulSD) for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_mul_s_v():
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_v")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [getattr(aesara.sparse, sp_format + "_matrix")(), tensor.vector()]

        f = aesara.function(inputs, sparse.mul_s_v(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.MulSV) for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_structured_add_s_v():
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_structured_add_s_v")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [getattr(aesara.sparse, sp_format + "_matrix")(), tensor.vector()]

        f = aesara.function(inputs, sparse.structured_add_s_v(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.StructuredAddSV)
            for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_sampling_dot_csr():
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_sampling_dot_csr")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [
            tensor.matrix(),
            tensor.matrix(),
            getattr(aesara.sparse, sp_format + "_matrix")(),
        ]

        f = aesara.function(inputs, sparse.sampling_dot(*inputs), mode=mode)

        if aesara.config.blas.ldflags:
            assert not any(
                isinstance(node.op, sparse.SamplingDot)
                for node in f.maker.fgraph.toposort()
            )
        else:
            # SamplingDotCSR's C implementation needs blas, so it should not
            # be inserted
            assert not any(
                isinstance(node.op, sparse.opt.SamplingDotCSR)
                for node in f.maker.fgraph.toposort()
            )


def test_local_dense_from_sparse_sparse_from_dense():
    mode = aesara.compile.mode.get_default_mode()
    mode = mode.including("local_dense_from_sparse_sparse_from_dense")

    m = aesara.tensor.matrix()
    for op in [aesara.sparse.csr_from_dense, aesara.sparse.csc_from_dense]:
        s = op(m)
        o = aesara.sparse.dense_from_sparse(s)
        f = aesara.function([m], o, mode=mode)
        # We should just have a deep copy.
        assert len(f.maker.fgraph.apply_nodes) == 1
        f([[1, 2], [3, 4]])


def test_sd_csc():

    A = sp.sparse.rand(4, 5, density=0.60, format="csc", dtype=np.float32)
    b = np.random.rand(5, 2).astype(np.float32)
    target = A * b

    a_val = aesara.tensor.as_tensor_variable(A.data)
    a_ind = aesara.tensor.as_tensor_variable(A.indices)
    a_ptr = aesara.tensor.as_tensor_variable(A.indptr)
    nrows = aesara.tensor.as_tensor_variable(np.int32(A.shape[0]))
    b = aesara.tensor.as_tensor_variable(b)

    res = aesara.sparse.opt.sd_csc(a_val, a_ind, a_ptr, nrows, b).eval()

    utt.assert_allclose(res, target)
