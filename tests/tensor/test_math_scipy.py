import numpy as np
import pytest


scipy = pytest.importorskip("scipy")

from functools import partial

import scipy.special
import scipy.stats

from aesara import function
from aesara import tensor as aet
from aesara.compile.mode import get_default_mode
from aesara.configdefaults import config
from aesara.tensor import inplace
from tests import unittest_tools as utt
from tests.tensor.utils import (
    _good_broadcast_unary_chi2sf,
    _good_broadcast_unary_normal,
    _good_broadcast_unary_normal_float,
    _good_broadcast_unary_normal_float_no_complex,
    _good_broadcast_unary_normal_float_no_complex_small_neg_range,
    _good_broadcast_unary_normal_no_complex,
    _grad_broadcast_unary_0_2_no_complex,
    _grad_broadcast_unary_abs1_no_complex,
    _grad_broadcast_unary_normal,
    _grad_broadcast_unary_normal_small_neg_range,
    check_floatX,
    copymod,
    integers_ranged,
    makeBroadcastTester,
    random_ranged,
    upcast_int8_nfunc,
)


mode_no_scipy = get_default_mode()


def scipy_special_gammau(k, x):
    return scipy.special.gammaincc(k, x) * scipy.special.gamma(k)


def scipy_special_gammal(k, x):
    return scipy.special.gammainc(k, x) * scipy.special.gamma(k)


# Precomputing the result is brittle(it have been broken!)
# As if we do any modification to random number here,
# The input random number will change and the output!
expected_erf = scipy.special.erf
expected_erfc = scipy.special.erfc
expected_erfinv = scipy.special.erfinv
expected_erfcinv = scipy.special.erfcinv
expected_gamma = scipy.special.gamma
expected_gammaln = scipy.special.gammaln
expected_psi = scipy.special.psi
expected_tri_gamma = partial(scipy.special.polygamma, 1)
expected_chi2sf = scipy.stats.chi2.sf
expected_gammainc = scipy.special.gammainc
expected_gammaincc = scipy.special.gammaincc
expected_gammau = scipy_special_gammau
expected_gammal = scipy_special_gammal
expected_j0 = scipy.special.j0
expected_j1 = scipy.special.j1
expected_jv = scipy.special.jv
expected_i0 = scipy.special.i0
expected_i1 = scipy.special.i1
expected_iv = scipy.special.iv
expected_erfcx = scipy.special.erfcx
expected_sigmoid = scipy.special.expit

TestErfBroadcast = makeBroadcastTester(
    op=aet.erf,
    expected=expected_erf,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestErfInplaceBroadcast = makeBroadcastTester(
    op=inplace.erf_inplace,
    expected=expected_erf,
    good=_good_broadcast_unary_normal_float,
    mode=mode_no_scipy,
    eps=2e-10,
    inplace=True,
)

TestErfcBroadcast = makeBroadcastTester(
    op=aet.erfc,
    expected=expected_erfc,
    good=_good_broadcast_unary_normal_float_no_complex,
    grad=_grad_broadcast_unary_normal,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestErfcInplaceBroadcast = makeBroadcastTester(
    op=inplace.erfc_inplace,
    expected=expected_erfc,
    good=_good_broadcast_unary_normal_float_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestErfcxBroadcast = makeBroadcastTester(
    op=aet.erfcx,
    expected=expected_erfcx,
    good=_good_broadcast_unary_normal_float_no_complex_small_neg_range,
    grad=_grad_broadcast_unary_normal_small_neg_range,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestErfcxInplaceBroadcast = makeBroadcastTester(
    op=inplace.erfcx_inplace,
    expected=expected_erfcx,
    good=_good_broadcast_unary_normal_float_no_complex_small_neg_range,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestErfinvBroadcast = makeBroadcastTester(
    op=aet.erfinv,
    expected=expected_erfinv,
    good={
        "normal": [random_ranged(-0.9, 0.9, (2, 3))],
        "empty": [np.asarray([], dtype=config.floatX)],
    },
    grad=_grad_broadcast_unary_abs1_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestErfcinvBroadcast = makeBroadcastTester(
    op=aet.erfcinv,
    expected=expected_erfcinv,
    good={
        "normal": [random_ranged(0.001, 1.9, (2, 3))],
        "empty": [np.asarray([], dtype=config.floatX)],
    },
    grad=_grad_broadcast_unary_0_2_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
)

_good_broadcast_unary_gammaln = dict(
    normal=(random_ranged(-1 + 1e-2, 10, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(1, 10, (2, 3)),),
    uint8=(integers_ranged(1, 6, (2, 3)).astype("uint8"),),
    uint16=(integers_ranged(1, 10, (2, 3)).astype("uint16"),),
    uint64=(integers_ranged(1, 10, (2, 3)).astype("uint64"),),
)
_grad_broadcast_unary_gammaln = dict(
    # smaller range as our grad method does not estimate it well enough.
    normal=(random_ranged(1e-1, 8, (2, 3)),),
)

TestGammaBroadcast = makeBroadcastTester(
    op=aet.gamma,
    expected=expected_gamma,
    good=_good_broadcast_unary_gammaln,
    grad=_grad_broadcast_unary_gammaln,
    mode=mode_no_scipy,
    eps=1e-5,
)
TestGammaInplaceBroadcast = makeBroadcastTester(
    op=inplace.gamma_inplace,
    expected=expected_gamma,
    good=_good_broadcast_unary_gammaln,
    mode=mode_no_scipy,
    eps=1e-5,
    inplace=True,
)

TestGammalnBroadcast = makeBroadcastTester(
    op=aet.gammaln,
    expected=expected_gammaln,
    good=_good_broadcast_unary_gammaln,
    grad=_grad_broadcast_unary_gammaln,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestGammalnInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaln_inplace,
    expected=expected_gammaln,
    good=_good_broadcast_unary_gammaln,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

_good_broadcast_unary_psi = dict(
    normal=(random_ranged(1, 10, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(1, 10, (2, 3)),),
    uint8=(integers_ranged(1, 10, (2, 3)).astype("uint8"),),
    uint16=(integers_ranged(1, 10, (2, 3)).astype("uint16"),),
)

TestPsiBroadcast = makeBroadcastTester(
    op=aet.psi,
    expected=expected_psi,
    good=_good_broadcast_unary_psi,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestPsiInplaceBroadcast = makeBroadcastTester(
    op=inplace.psi_inplace,
    expected=expected_psi,
    good=_good_broadcast_unary_psi,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

_good_broadcast_unary_tri_gamma = _good_broadcast_unary_psi

TestTriGammaBroadcast = makeBroadcastTester(
    op=aet.tri_gamma,
    expected=expected_tri_gamma,
    good=_good_broadcast_unary_psi,
    eps=2e-8,
    mode=mode_no_scipy,
)
TestTriGammaInplaceBroadcast = makeBroadcastTester(
    op=inplace.tri_gamma_inplace,
    expected=expected_tri_gamma,
    good=_good_broadcast_unary_tri_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestChi2SFBroadcast = makeBroadcastTester(
    op=aet.chi2sf,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    name="Chi2SF",
)

TestChi2SFInplaceBroadcast = makeBroadcastTester(
    op=inplace.chi2sf_inplace,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    name="Chi2SF",
)

_good_broadcast_binary_gamma = dict(
    normal=(random_ranged(1e-2, 10, (2, 3)), random_ranged(1e-2, 10, (2, 3))),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    int=(integers_ranged(1, 10, (2, 3)), integers_ranged(1, 10, (2, 3))),
    uint8=(
        integers_ranged(1, 6, (2, 3)).astype("uint8"),
        integers_ranged(1, 6, (2, 3)).astype("uint8"),
    ),
    uint16=(
        integers_ranged(1, 10, (2, 3)).astype("uint16"),
        integers_ranged(1, 10, (2, 3)).astype("uint16"),
    ),
    uint64=(
        integers_ranged(1, 10, (2, 3)).astype("uint64"),
        integers_ranged(1, 10, (2, 3)).astype("uint64"),
    ),
)

_good_broadcast_binary_gamma_grad = dict(
    normal=_good_broadcast_binary_gamma["normal"],
    specific_branches=(
        np.array([0.7, 11.0, 19.0]),
        np.array([16.0, 31.0, 3.0]),
    ),
)

TestGammaIncBroadcast = makeBroadcastTester(
    op=aet.gammainc,
    expected=expected_gammainc,
    good=_good_broadcast_binary_gamma,
    grad=_good_broadcast_binary_gamma_grad,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaIncInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammainc_inplace,
    expected=expected_gammainc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestGammaInccBroadcast = makeBroadcastTester(
    op=aet.gammaincc,
    expected=expected_gammaincc,
    good=_good_broadcast_binary_gamma,
    grad=_good_broadcast_binary_gamma_grad,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaInccInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaincc_inplace,
    expected=expected_gammaincc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)


def test_gammainc_ddk_tabulated_values():
    # This test replicates part of the old STAN test:
    # https://github.com/stan-dev/math/blob/21333bb70b669a1bd54d444ecbe1258078d33153/test/unit/math/prim/scal/fun/grad_reg_lower_inc_gamma_test.cpp
    k, x = aet.scalars("k", "x")
    gammainc_out = aet.gammainc(k, x)
    gammaincc_ddk = aet.grad(gammainc_out, k)
    f_grad = function([k, x], gammaincc_ddk)

    for test_k, test_x, expected_ddk in (
        (0.0001, 0, 0),  # Limit condition
        (0.0001, 0.0001, -8.62594024578651),
        (0.0001, 6.2501, -0.0002705821702813008),
        (0.0001, 12.5001, -2.775406821933887e-7),
        (0.0001, 18.7501, -3.653379783274905e-10),
        (0.0001, 25.0001, -5.352425240798134e-13),
        (0.0001, 29.7501, -3.912723010174313e-15),
        (4.7501, 0.0001, 0),
        (4.7501, 6.2501, -0.1330287013623819),
        (4.7501, 12.5001, -0.004712176128251421),
        (4.7501, 18.7501, -0.00004898939126595217),
        (4.7501, 25.0001, -3.098781566343336e-7),
        (4.7501, 29.7501, -5.478399030091586e-9),
        (9.5001, 0.0001, -5.869126325643798e-15),
        (9.5001, 6.2501, -0.07717967485372858),
        (9.5001, 12.5001, -0.07661095137424883),
        (9.5001, 18.7501, -0.005594043337407605),
        (9.5001, 25.0001, -0.0001410123206233104),
        (9.5001, 29.7501, -5.75023943432906e-6),
        (14.2501, 0.0001, -7.24495484418588e-15),
        (14.2501, 6.2501, -0.003689474744087815),
        (14.2501, 12.5001, -0.1008796179460247),
        (14.2501, 18.7501, -0.05124664255610913),
        (14.2501, 25.0001, -0.005115177188580634),
        (14.2501, 29.7501, -0.0004793406401524598),
        (19.0001, 0.0001, -8.26027539153394e-15),
        (19.0001, 6.2501, -0.00003509660448733015),
        (19.0001, 12.5001, -0.02624562607393565),
        (19.0001, 18.7501, -0.0923829735092193),
        (19.0001, 25.0001, -0.03641281853907181),
        (19.0001, 29.7501, -0.007828749832965796),
    ):
        np.testing.assert_allclose(
            f_grad(test_k, test_x), expected_ddk, rtol=1e-5, atol=1e-14
        )


TestGammaUBroadcast = makeBroadcastTester(
    op=aet.gammau,
    expected=expected_gammau,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaUInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammau_inplace,
    expected=expected_gammau,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestGammaLBroadcast = makeBroadcastTester(
    op=aet.gammal,
    expected=expected_gammal,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaLInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammal_inplace,
    expected=expected_gammal,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

_good_broadcast_unary_bessel = dict(
    normal=(random_ranged(-10, 10, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(-10, 10, (2, 3)),),
    uint8=(integers_ranged(0, 10, (2, 3)).astype("uint8"),),
    uint16=(integers_ranged(0, 10, (2, 3)).astype("uint16"),),
)

_grad_broadcast_unary_bessel = dict(
    normal=(random_ranged(-10.0, 10.0, (2, 3)),),
)

_good_broadcast_binary_bessel = dict(
    normal=(random_ranged(-5, 5, (2, 3)), random_ranged(0, 10, (2, 3))),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    integers=(integers_ranged(-5, 5, (2, 3)), integers_ranged(-10, 10, (2, 3))),
    uint8=(
        integers_ranged(0, 5, (2, 3)).astype("uint8"),
        integers_ranged(0, 10, (2, 3)).astype("uint8"),
    ),
    uint16=(
        integers_ranged(0, 5, (2, 3)).astype("uint16"),
        integers_ranged(0, 10, (2, 3)).astype("uint16"),
    ),
)

_grad_broadcast_binary_bessel = dict(
    normal=(random_ranged(1, 5, (2, 3)), random_ranged(0, 10, (2, 3)))
)

TestJ0Broadcast = makeBroadcastTester(
    op=aet.j0,
    expected=expected_j0,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestJ0InplaceBroadcast = makeBroadcastTester(
    op=inplace.j0_inplace,
    expected=expected_j0,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestJ1Broadcast = makeBroadcastTester(
    op=aet.j1,
    expected=expected_j1,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestJ1InplaceBroadcast = makeBroadcastTester(
    op=inplace.j1_inplace,
    expected=expected_j1,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestJvBroadcast = makeBroadcastTester(
    op=aet.jv,
    expected=expected_jv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestJvInplaceBroadcast = makeBroadcastTester(
    op=inplace.jv_inplace,
    expected=expected_jv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)


def test_verify_jv_grad():
    # Verify Jv gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_jv(x):
        return aet.jv(v_val, x)

    utt.verify_grad(fixed_first_input_jv, [x_val])


TestI0Broadcast = makeBroadcastTester(
    op=aet.i0,
    expected=expected_i0,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestI0InplaceBroadcast = makeBroadcastTester(
    op=inplace.i0_inplace,
    expected=expected_i0,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestI1Broadcast = makeBroadcastTester(
    op=aet.i1,
    expected=expected_i1,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestI1InplaceBroadcast = makeBroadcastTester(
    op=inplace.i1_inplace,
    expected=expected_i1,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestIvBroadcast = makeBroadcastTester(
    op=aet.iv,
    expected=expected_iv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestIvInplaceBroadcast = makeBroadcastTester(
    op=inplace.iv_inplace,
    expected=expected_iv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)


def test_verify_iv_grad():
    # Verify Iv gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_iv(x):
        return aet.iv(v_val, x)

    utt.verify_grad(fixed_first_input_iv, [x_val])


TestSigmoidBroadcast = makeBroadcastTester(
    op=aet.sigmoid,
    expected=expected_sigmoid,
    good=_good_broadcast_unary_normal_no_complex,
    eps=1e-8,
)

TestSigmoidInplaceBroadcast = makeBroadcastTester(
    op=inplace.sigmoid_inplace,
    expected=expected_sigmoid,
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal,
    eps=1e-8,
    inplace=True,
)


class TestSigmoid:
    def test_elemwise(self):
        utt.verify_grad(aet.sigmoid, [np.random.random((3, 4))])


_good_broadcast_unary_softplus = dict(
    copymod(
        _good_broadcast_unary_normal_no_complex,
        without=["uint8", "uint16", "big_scalar"],
    ),  # numpy function overflows with uint16.
    uint8=[
        np.arange(0, 89, dtype="uint8")
    ],  # the range is different in new added uint8.
    int8=[np.arange(-127, 89, dtype="int8")],
)

expected_sofplus = upcast_int8_nfunc(
    lambda inputs: check_floatX(inputs, np.log1p(np.exp(inputs)))
)

TestSoftplusBroadcast = makeBroadcastTester(
    op=aet.softplus,
    expected=expected_sofplus,
    good=_good_broadcast_unary_softplus,
    eps=1e-8,
)

TestSoftplusInplaceBroadcast = makeBroadcastTester(
    op=inplace.softplus_inplace,
    expected=expected_sofplus,
    good=_good_broadcast_unary_softplus,
    grad=_grad_broadcast_unary_normal,
    eps=1e-8,
    inplace=True,
)


class TestSoftplus:
    def test_elemwise(self):
        utt.verify_grad(aet.softplus, [np.random.random((3, 4))])

    def test_accuracy(self):
        # Test all approximations are working (cutoff points are -37, 18, 33.3)
        x_test = np.array([-40.0, -17.5, 17.5, 18.5, 40.0])
        y_th = aet.softplus(x_test).eval()
        y_np = np.log1p(np.exp(x_test))
        np.testing.assert_allclose(y_th, y_np, rtol=10e-10)


_good_broadcast_unary_log1mexp = dict(
    normal=(random_ranged(-10.0, 0, (2, 3)),),
    float32=(random_ranged(-10.0, 0, (2, 3)).astype("float32"),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(-10, -1, (2, 3)),),
)

_grad_broadcast_unary_log1mexp = dict(
    normal=(random_ranged(-10.0, 0.0, (2, 3)),),
)


def expected_log1mexp(x):
    return check_floatX(x, np.log(-np.expm1(x)))


TestLog1mexpBroadcast = makeBroadcastTester(
    op=aet.log1mexp,
    expected=expected_log1mexp,
    good=_good_broadcast_unary_log1mexp,
    grad=_grad_broadcast_unary_log1mexp,
    eps=1e-8,
)

TestLog1mexpInplaceBroadcast = makeBroadcastTester(
    op=inplace.log1mexp_inplace,
    expected=expected_log1mexp,
    good=_good_broadcast_unary_log1mexp,
    eps=1e-8,
    inplace=True,
)


def test_deprecated_module():
    with pytest.warns(DeprecationWarning):
        import aesara.scalar.basic_scipy  # noqa: F401


_good_broadcast_ternary_betainc = dict(
    normal=(
        random_ranged(0, 1000, (2, 3)),
        random_ranged(0, 1000, (2, 3)),
        random_ranged(0, 1, (2, 3)),
    ),
)

TestBetaincBroadcast = makeBroadcastTester(
    op=aet.betainc,
    expected=scipy.special.betainc,
    good=_good_broadcast_ternary_betainc,
    grad=_good_broadcast_ternary_betainc,
)

TestBetaincInplaceBroadcast = makeBroadcastTester(
    op=inplace.betainc_inplace,
    expected=scipy.special.betainc,
    good=_good_broadcast_ternary_betainc,
    grad=_good_broadcast_ternary_betainc,
    inplace=True,
)


class TestBetaIncGrad:
    def test_stan_grad_partial(self):
        # This test combines the following STAN tests:
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_dda_test.cpp
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_ddb_test.cpp
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_ddz_test.cpp
        a, b, z = aet.scalars("a", "b", "z")
        betainc_out = aet.betainc(a, b, z)
        betainc_grad = aet.grad(betainc_out, [a, b, z])
        f_grad = function([a, b, z], betainc_grad)

        decimal_precision = 7 if config.floatX == "float64" else 3

        for test_a, test_b, test_z, expected_dda, expected_ddb, expected_ddz in (
            (1.5, 1.25, 0.001, -0.00028665637, 4.41357328e-05, 0.063300692),
            (1.5, 1.25, 0.5, -0.26038693947, 0.29301795, 1.1905416),
            (1.5, 1.25, 0.6, -0.23806757, 0.32279575, 1.23341068),
            (1.5, 1.25, 0.999, -0.00022264493, 0.0018969609, 0.35587692),
            (15000, 1.25, 0.001, 0, 0, 0),
            (15000, 1.25, 0.5, 0, 0, 0),
            (15000, 1.25, 0.6, 0, 0, 0),
            (15000, 1.25, 0.999, -6.59543226e-10, 2.00849793e-06, 0.009898182),
            (1.5, 12500, 0.001, -3.93756641e-05, 1.47821755e-09, 0.1848717),
            (1.5, 12500, 0.5, 0, 0, 0),
            (1.5, 12500, 0.6, 0, 0, 0),
            (1.5, 12500, 0.999, 0, 0, 0),
            (15000, 12500, 0.001, 0, 0, 0),
            (15000, 12500, 0.5, -8.72102443e-53, 9.55282792e-53, 5.01131256e-48),
            (15000, 12500, 0.6, -4.085621e-14, -5.5067062e-14, 1.15135267e-71),
            (15000, 12500, 0.999, 0, 0, 0),
        ):
            np.testing.assert_almost_equal(
                f_grad(test_a, test_b, test_z),
                [expected_dda, expected_ddb, expected_ddz],
                decimal=decimal_precision,
            )

    def test_boik_robison_cox(self):
        # This test compares against the tabulated values in:
        # Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
        # Journal of Statistical Software, 3(1), 1-20.
        a, b, z = aet.scalars("a", "b", "z")
        betainc_out = aet.betainc(a, b, z)
        betainc_grad = aet.grad(betainc_out, [a, b])
        f_grad = function([a, b, z], betainc_grad)

        for test_a, test_b, test_z, expected_dda, expected_ddb in (
            (1.5, 11.0, 0.001, -4.5720356e-03, 1.1845673e-04),
            (1.5, 11.0, 0.5, -2.5501997e-03, 9.0824388e-04),
            (1000.0, 1000.0, 0.5, -8.9224793e-03, 8.9224793e-03),
            (1000.0, 1000.0, 0.55, -3.6713108e-07, 4.0584118e-07),
        ):
            np.testing.assert_almost_equal(
                f_grad(test_a, test_b, test_z),
                [expected_dda, expected_ddb],
            )

    def test_beta_inc_stan_grad_combined(self):
        # This test replicates the following STAN test:
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/grad_reg_inc_beta_test.cpp
        a, b, z = aet.scalars("a", "b", "z")
        betainc_out = aet.betainc(a, b, z)
        betainc_grad = aet.grad(betainc_out, [a, b])
        f_grad = function([a, b, z], betainc_grad)

        for test_a, test_b, test_z, expected_dda, expected_ddb in (
            (1.0, 1.0, 1.0, 0, np.nan),
            (1.0, 1.0, 0.4, -0.36651629, 0.30649537),
        ):
            np.testing.assert_allclose(
                f_grad(test_a, test_b, test_z), [expected_dda, expected_ddb]
            )
