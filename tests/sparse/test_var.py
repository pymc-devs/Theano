from contextlib import ExitStack

import numpy as np
import pytest
from scipy.sparse.csr import csr_matrix

import aesara
import aesara.sparse as sparse
import aesara.tensor as at
from aesara.sparse.type import SparseType
from aesara.tensor.type import DenseTensorType


class TestSparseVariable:
    @pytest.mark.parametrize(
        "method, exp_type, cm",
        [
            ("__abs__", DenseTensorType, None),
            ("__neg__", SparseType, ExitStack()),
            ("__ceil__", DenseTensorType, None),
            ("__floor__", DenseTensorType, None),
            ("__trunc__", DenseTensorType, None),
            ("transpose", DenseTensorType, None),
            ("any", DenseTensorType, None),
            ("all", DenseTensorType, None),
            ("flatten", DenseTensorType, None),
            ("ravel", DenseTensorType, None),
            ("arccos", DenseTensorType, None),
            ("arcsin", DenseTensorType, None),
            ("arctan", DenseTensorType, None),
            ("arccosh", DenseTensorType, None),
            ("arcsinh", DenseTensorType, None),
            ("arctanh", DenseTensorType, None),
            ("ceil", DenseTensorType, None),
            ("cos", DenseTensorType, None),
            ("cosh", DenseTensorType, None),
            ("deg2rad", DenseTensorType, None),
            ("exp", DenseTensorType, None),
            ("exp2", DenseTensorType, None),
            ("expm1", DenseTensorType, None),
            ("floor", DenseTensorType, None),
            ("log", DenseTensorType, None),
            ("log10", DenseTensorType, None),
            ("log1p", DenseTensorType, None),
            ("log2", DenseTensorType, None),
            ("rad2deg", DenseTensorType, None),
            ("sin", DenseTensorType, None),
            ("sinh", DenseTensorType, None),
            ("sqrt", DenseTensorType, None),
            ("tan", DenseTensorType, None),
            ("tanh", DenseTensorType, None),
            ("copy", DenseTensorType, None),
            ("sum", DenseTensorType, ExitStack()),
            ("prod", DenseTensorType, None),
            ("mean", DenseTensorType, None),
            ("var", DenseTensorType, None),
            ("std", DenseTensorType, None),
            ("min", DenseTensorType, None),
            ("max", DenseTensorType, None),
            ("argmin", DenseTensorType, None),
            ("argmax", DenseTensorType, None),
            ("nonzero", DenseTensorType, ExitStack()),
            ("nonzero_values", DenseTensorType, None),
            ("argsort", DenseTensorType, ExitStack()),
            ("conj", DenseTensorType, None),
            ("round", DenseTensorType, None),
            ("trace", DenseTensorType, None),
            ("zeros_like", SparseType, ExitStack()),
            ("ones_like", DenseTensorType, ExitStack()),
            ("cumsum", DenseTensorType, None),
            ("cumprod", DenseTensorType, None),
            ("ptp", DenseTensorType, None),
            ("squeeze", DenseTensorType, None),
            ("diagonal", DenseTensorType, None),
        ],
    )
    def test_unary(self, method, exp_type, cm):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        method_to_call = getattr(x, method)

        if cm is None:
            cm = pytest.warns(UserWarning, match=".*converted to dense.*")

        if exp_type == SparseType:
            exp_res_type = csr_matrix
        else:
            exp_res_type = np.ndarray

        with cm:
            z = method_to_call()

        if not isinstance(z, tuple):
            z_outs = (z,)
        else:
            z_outs = z

        assert all(isinstance(out.type, exp_type) for out in z_outs)

        f = aesara.function([x], z, on_unused_input="ignore")

        res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])

        if not isinstance(res, list):
            res_outs = [res]
        else:
            res_outs = res

        assert all(isinstance(out, exp_res_type) for out in res_outs)

    @pytest.mark.parametrize(
        "method, exp_type",
        [
            ("__lt__", SparseType),
            ("__le__", SparseType),
            ("__gt__", SparseType),
            ("__ge__", SparseType),
            ("__and__", DenseTensorType),
            ("__or__", DenseTensorType),
            ("__xor__", DenseTensorType),
            ("__add__", SparseType),
            ("__sub__", SparseType),
            ("__mul__", SparseType),
            ("__pow__", DenseTensorType),
            ("__mod__", DenseTensorType),
            ("__divmod__", DenseTensorType),
            ("__truediv__", DenseTensorType),
            ("__floordiv__", DenseTensorType),
        ],
    )
    def test_binary(self, method, exp_type):
        x = at.lmatrix("x")
        y = at.lmatrix("y")
        x = sparse.csr_from_dense(x)
        y = sparse.csr_from_dense(y)

        method_to_call = getattr(x, method)

        if exp_type == SparseType:
            exp_res_type = csr_matrix
            cm = ExitStack()
        else:
            exp_res_type = np.ndarray
            cm = pytest.warns(UserWarning, match=".*converted to dense.*")

        with cm:
            z = method_to_call(y)

        if not isinstance(z, tuple):
            z_outs = (z,)
        else:
            z_outs = z

        assert all(isinstance(out.type, exp_type) for out in z_outs)

        f = aesara.function([x, y], z)
        res = f(
            [[1, 0, 2], [-1, 0, 0]],
            [[1, 1, 2], [1, 4, 1]],
        )

        if not isinstance(res, list):
            res_outs = [res]
        else:
            res_outs = res

        assert all(isinstance(out, exp_res_type) for out in res_outs)

    def test_reshape(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.reshape((3, 2))

        assert isinstance(z.type, DenseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

    def test_dimshuffle(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.dimshuffle((1, 0))

        assert isinstance(z.type, DenseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

    def test_getitem(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        z = x[:, :2]
        assert isinstance(z.type, SparseType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, csr_matrix)

    def test_dot(self):
        x = at.lmatrix("x")
        y = at.lmatrix("y")
        x = sparse.csr_from_dense(x)
        y = sparse.csr_from_dense(y)

        z = x.__dot__(y)
        assert isinstance(z.type, SparseType)

        f = aesara.function([x, y], z)
        exp_res = f(
            [[1, 0, 2], [-1, 0, 0]],
            [[-1], [2], [1]],
        )
        assert isinstance(exp_res, csr_matrix)

    def test_repeat(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.repeat(2, axis=1)

        assert isinstance(z.type, DenseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)
