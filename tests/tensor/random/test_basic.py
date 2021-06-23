import pickle
from functools import partial

import numpy as np
import pytest
import scipy.stats as stats
from pytest import fixture, importorskip, raises

import aesara.tensor as aet
from aesara import shared
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.tensor.basic_opt import ShapeFeature
from aesara.tensor.random.basic import (
    bernoulli,
    beta,
    betabinom,
    binomial,
    categorical,
    cauchy,
    chisquare,
    choice,
    dirichlet,
    exponential,
    gamma,
    geometric,
    gumbel,
    halfcauchy,
    halfnormal,
    hypergeometric,
    invgamma,
    laplace,
    logistic,
    lognormal,
    multinomial,
    multivariate_normal,
    nbinom,
    normal,
    pareto,
    permutation,
    poisson,
    polyagamma,
    randint,
    triangular,
    truncexpon,
    uniform,
    vonmises,
    wald,
    weibull,
)
from aesara.tensor.type import iscalar, scalar, tensor


@fixture(scope="module", autouse=True)
def set_aesara_flags():
    with config.change_flags(cxx="", compute_test_value="raise"):
        yield


def rv_numpy_tester(rv, *params, **kwargs):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    test_fn = kwargs.pop("test_fn", None)

    if test_fn is None:
        name = getattr(rv, "name", None)

        if name is None:
            name = rv.__name__

        test_fn = getattr(np.random, name)

    aesara_res = rv(*params, **kwargs)

    param_vals = [get_test_value(p) if isinstance(p, Variable) else p for p in params]
    kwargs_vals = {
        k: get_test_value(v) if isinstance(v, Variable) else v
        for k, v in kwargs.items()
    }

    if "size" in kwargs:
        kwargs["size"] = get_test_value(kwargs["size"])

    numpy_res = np.asarray(test_fn(*param_vals, **kwargs_vals))

    assert aesara_res.type.numpy_dtype.kind == numpy_res.dtype.kind

    numpy_shape = np.shape(numpy_res)
    numpy_bcast = [s == 1 for s in numpy_shape]
    np.testing.assert_array_equal(aesara_res.type.broadcastable, numpy_bcast)

    aesara_res_val = aesara_res.get_test_value()
    np.testing.assert_array_equal(aesara_res_val.shape, numpy_res.shape)


def test_uniform_samples():

    rv_numpy_tester(uniform)
    rv_numpy_tester(uniform, size=())

    test_low = np.array(10, dtype=config.floatX)
    test_high = np.array(20, dtype=config.floatX)

    rv_numpy_tester(uniform, test_low, test_high)
    rv_numpy_tester(uniform, test_low, test_high, size=[3])


def test_triangular_samples():
    test_left = np.array(10, dtype=config.floatX)
    test_mode = np.array(12, dtype=config.floatX)
    test_right = np.array(20, dtype=config.floatX)

    rv_numpy_tester(triangular, test_left, test_mode, test_right)
    rv_numpy_tester(triangular, test_left, test_mode, test_right, size=[2, 3])


def test_beta_samples():

    test_a = np.array(0.5, dtype=config.floatX)
    test_b = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(beta, test_a, test_b)
    rv_numpy_tester(beta, test_a, test_b, size=[3])


def test_normal_infer_shape():
    M_aet = iscalar("M")
    M_aet.tag.test_value = 3
    sd_aet = scalar("sd")
    sd_aet.tag.test_value = np.array(1.0, dtype=config.floatX)

    test_params = [
        ([aet.as_tensor_variable(np.array(1.0, dtype=config.floatX)), sd_aet], None),
        (
            [aet.as_tensor_variable(np.array(1.0, dtype=config.floatX)), sd_aet],
            (M_aet,),
        ),
        (
            [aet.as_tensor_variable(np.array(1.0, dtype=config.floatX)), sd_aet],
            (2, M_aet),
        ),
        ([aet.zeros((M_aet,)), sd_aet], None),
        ([aet.zeros((M_aet,)), sd_aet], (M_aet,)),
        ([aet.zeros((M_aet,)), sd_aet], (2, M_aet)),
        ([aet.zeros((M_aet,)), aet.ones((M_aet,))], None),
        ([aet.zeros((M_aet,)), aet.ones((M_aet,))], (2, M_aet)),
        (
            [
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX),
                np.array([[1e-6, 2e-6]], dtype=config.floatX),
            ],
            (3, 2, 2),
        ),
        (
            [np.array([1], dtype=config.floatX), np.array([10], dtype=config.floatX)],
            (1, 2),
        ),
    ]
    for args, size in test_params:
        rv = normal(*args, size=size)
        rv_shape = tuple(normal._infer_shape(size or (), args, None))
        assert tuple(get_test_value(rv_shape)) == tuple(get_test_value(rv).shape)


def test_normal_ShapeFeature():
    M_aet = iscalar("M")
    M_aet.tag.test_value = 3
    sd_aet = scalar("sd")
    sd_aet.tag.test_value = np.array(1.0, dtype=config.floatX)

    d_rv = normal(aet.ones((M_aet,)), sd_aet, size=(2, M_aet))
    d_rv.tag.test_value

    fg = FunctionGraph(
        [i for i in graph_inputs([d_rv]) if not isinstance(i, Constant)],
        [d_rv],
        clone=False,
        features=[ShapeFeature()],
    )
    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert get_test_value(s1) == get_test_value(d_rv).shape[0]
    assert get_test_value(s2) == get_test_value(d_rv).shape[1]


def test_normal_samples():

    rv_numpy_tester(normal)

    test_mean = np.array(0, dtype=config.floatX)
    test_stddev = np.array(1, dtype=config.floatX)

    rv_numpy_tester(normal, test_mean, test_stddev)
    rv_numpy_tester(normal, test_mean, test_stddev, size=[3])

    # Broadcast sd over independent means...
    test_mean = np.array([0, 1, 2], dtype=config.floatX)
    test_stddev = np.array(1, dtype=config.floatX)
    rv_numpy_tester(normal, test_mean, test_stddev)
    rv_numpy_tester(normal, test_mean, test_stddev, size=[3, 3])

    test_mean = np.array([0], dtype=config.floatX)
    test_stddev = np.array([1], dtype=config.floatX)
    rv_numpy_tester(normal, test_mean, test_stddev, size=[1])
    rv_numpy_tester(normal, aet.as_tensor(test_mean), test_stddev, size=[1])
    rv_numpy_tester(
        normal,
        aet.as_tensor_variable(test_mean),
        test_stddev,
        size=aet.as_tensor_variable([1]),
    )


def test_halfnormal_samples():
    test_mean = np.array(0, dtype=config.floatX)
    test_stddev = np.array(1, dtype=config.floatX)

    rv_numpy_tester(halfnormal, test_fn=stats.halfnorm.rvs)
    rv_numpy_tester(halfnormal, test_mean, test_stddev, test_fn=stats.halfnorm.rvs)
    rv_numpy_tester(
        halfnormal,
        test_mean,
        test_stddev,
        size=[2, 3],
        test_fn=stats.halfnorm.rvs,
    )


def test_lognormal_samples():
    test_mean = np.array(0, dtype=config.floatX)
    test_sigma = np.array(1, dtype=config.floatX)

    rv_numpy_tester(lognormal)
    rv_numpy_tester(lognormal, test_mean)
    rv_numpy_tester(lognormal, test_mean, test_sigma, size=[2, 3])


def test_gamma_samples():
    test_a = np.array(0.5, dtype=config.floatX)
    test_b = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(gamma, test_a, test_b, test_fn=stats.gamma.rvs)
    rv_numpy_tester(gamma, test_a, test_b, size=[2, 3], test_fn=stats.gamma.rvs)


def test_chisquare_samples():
    test_df = np.array(2, dtype=config.floatX)

    rv_numpy_tester(chisquare, test_df, test_fn=stats.chi2.rvs)
    rv_numpy_tester(chisquare, test_df, size=[2, 3], test_fn=stats.chi2.rvs)


def test_gumbel_samples():
    test_mu = np.array(0.0, dtype=config.floatX)
    test_beta = np.array(1.0, dtype=config.floatX)

    rv_numpy_tester(gumbel, test_mu, test_beta, test_fn=stats.gumbel_r.rvs)
    rv_numpy_tester(gumbel, test_mu, test_beta, size=[2, 3], test_fn=stats.gumbel_r.rvs)


def test_exponential_samples():

    rv_numpy_tester(exponential)

    test_lambda = np.array(10, dtype=config.floatX)

    rv_numpy_tester(exponential, test_lambda)
    rv_numpy_tester(exponential, test_lambda, size=[2, 3])


def test_weibull_samples():
    test_alpha = np.array(3, dtype=config.floatX)

    rv_numpy_tester(weibull, test_alpha)
    rv_numpy_tester(weibull, test_alpha, size=[2, 3])


def test_logistic_samples():
    test_loc = np.array(2, dtype=config.floatX)
    test_scale = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(logistic)
    rv_numpy_tester(logistic, test_loc)
    rv_numpy_tester(logistic, test_loc, test_scale, size=[2, 3])


def test_vonmises_samples():
    test_mu = np.array(np.pi, dtype=config.floatX)
    test_kappa = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(vonmises, test_mu, test_kappa)
    rv_numpy_tester(vonmises, test_mu, test_kappa, size=[2, 3])


def test_pareto_samples():
    test_alpha = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(pareto, test_alpha, test_fn=stats.pareto.rvs)
    rv_numpy_tester(pareto, test_alpha, size=[2, 3], test_fn=stats.pareto.rvs)


def test_mvnormal_samples():
    def test_fn(mean=None, cov=None, size=None, rng=None):
        if mean is None:
            mean = np.array([0.0], dtype=config.floatX)
        if cov is None:
            cov = np.array([[1.0]], dtype=config.floatX)
        if size is None:
            size = ()
        return multivariate_normal.rng_fn(rng, mean, cov, size)

    rv_numpy_tester(multivariate_normal, test_fn=test_fn)

    test_mean = np.array([0], dtype=config.floatX)
    test_covar = np.diag(np.array([1], dtype=config.floatX))
    rv_numpy_tester(multivariate_normal, test_mean, test_covar, test_fn=test_fn)
    rv_numpy_tester(
        multivariate_normal, test_mean, test_covar, size=[1], test_fn=test_fn
    )
    rv_numpy_tester(
        multivariate_normal, test_mean, test_covar, size=[4], test_fn=test_fn
    )
    rv_numpy_tester(
        multivariate_normal, test_mean, test_covar, size=[4, 1], test_fn=test_fn
    )
    rv_numpy_tester(
        multivariate_normal, test_mean, test_covar, size=[4, 1, 1], test_fn=test_fn
    )
    rv_numpy_tester(
        multivariate_normal, test_mean, test_covar, size=[1, 5, 8], test_fn=test_fn
    )
    test_mean = np.array(
        [0, 1, 2],
        dtype=config.floatX,
    )
    test_covar = np.diag(np.array([1, 10, 100], dtype=config.floatX))
    rv_numpy_tester(multivariate_normal, test_mean, test_covar, test_fn=test_fn)

    # Test parameter broadcasting
    rv_numpy_tester(
        multivariate_normal,
        np.array([[0, 1, 2], [4, 5, 6]], dtype=config.floatX),
        test_covar,
        test_fn=test_fn,
    )

    test_covar = np.stack([test_covar, test_covar * 10.0])
    rv_numpy_tester(
        multivariate_normal,
        np.array([0, 1, 2], dtype=config.floatX),
        test_covar,
        size=[2, 3, 2],
        test_fn=test_fn,
    )

    test_covar = np.stack([test_covar, test_covar * 10.0])
    rv_numpy_tester(
        multivariate_normal,
        np.array([[0, 1, 2]], dtype=config.floatX),
        test_covar,
        size=[2, 3, 2, 2],
        test_fn=test_fn,
    )

    rv_numpy_tester(
        multivariate_normal,
        np.array([[0], [10], [100]], dtype=config.floatX),
        np.diag(np.array([1e-6], dtype=config.floatX)),
        size=[2, 3, 3],
        test_fn=test_fn,
    )

    with pytest.raises(ValueError, match="shape mismatch.*"):
        multivariate_normal.rng_fn(
            None, np.zeros((1, 2)), np.ones((1, 2, 2)), size=(4,)
        )


def test_mvnormal_ShapeFeature():
    M_aet = iscalar("M")
    M_aet.tag.test_value = 2

    d_rv = multivariate_normal(aet.ones((M_aet,)), aet.eye(M_aet), size=2)

    fg = FunctionGraph(
        [i for i in graph_inputs([d_rv]) if not isinstance(i, Constant)],
        [d_rv],
        clone=False,
        features=[ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert get_test_value(s1) == 2
    assert M_aet in graph_inputs([s2])

    # Test broadcasted shapes
    mean = tensor(config.floatX, [True, False])
    mean.tag.test_value = np.array([[0, 1, 2]], dtype=config.floatX)

    test_covar = np.diag(np.array([1, 10, 100], dtype=config.floatX))
    test_covar = np.stack([test_covar, test_covar * 10.0])
    cov = aet.as_tensor(test_covar).type()
    cov.tag.test_value = test_covar

    d_rv = multivariate_normal(mean, cov, size=[2, 3, 2])

    fg = FunctionGraph(
        outputs=[d_rv],
        clone=False,
        features=[ShapeFeature()],
    )

    s1, s2, s3, s4 = fg.shape_feature.shape_of[d_rv]

    assert s1.get_test_value() == 2
    assert s2.get_test_value() == 3
    assert s3.get_test_value() == 2
    assert s4.get_test_value() == 3


def test_dirichlet_samples():

    alphas = np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX)

    res = get_test_value(dirichlet(alphas))
    assert np.all(np.diag(res) >= res)

    res = get_test_value(dirichlet(alphas, size=(2, 3)))
    assert res.shape == (2, 3, 3)
    assert all(np.all(np.diag(r) >= r) for r in res)

    for i in range(alphas.shape[0]):
        res = get_test_value(dirichlet(alphas[i]))
        assert np.all(res[i] > np.delete(res, [i]))

        res = get_test_value(dirichlet(alphas[i], size=2))
        assert res.shape == (2, 3)
        assert all(np.all(r[i] > np.delete(r, [i])) for r in res)

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))

    alphas = np.array([[1000, 1, 1], [1, 1000, 1], [1, 1, 1000]], dtype=config.floatX)

    assert dirichlet.rng_fn(rng_state, alphas, size=()).shape == alphas.shape
    assert (
        dirichlet.rng_fn(rng_state, alphas, size=(10, 3)).shape == (10,) + alphas.shape
    )
    assert (
        dirichlet.rng_fn(rng_state, alphas, size=(10, 2, 3)).shape
        == (10, 2) + alphas.shape
    )

    with pytest.raises(ValueError, match="shape mismatch.*"):
        dirichlet.rng_fn(rng_state, alphas, size=(8, 2, 1))


def test_dirichlet_infer_shape():
    M_aet = iscalar("M")
    M_aet.tag.test_value = 3

    test_params = [
        ([aet.ones((M_aet,))], None),
        ([aet.ones((M_aet,))], (M_aet + 1,)),
        ([aet.ones((M_aet,))], (2, M_aet)),
        ([aet.ones((M_aet, M_aet + 1))], None),
        ([aet.ones((M_aet, M_aet + 1))], (M_aet + 2, M_aet)),
        ([aet.ones((M_aet, M_aet + 1))], (2, M_aet + 2, M_aet + 3, M_aet)),
    ]
    for args, size in test_params:
        rv = dirichlet(*args, size=size)
        rv_shape = tuple(dirichlet._infer_shape(size or (), args, None))
        assert tuple(get_test_value(rv_shape)) == tuple(get_test_value(rv).shape)


def test_dirichlet_ShapeFeature():
    """Make sure `RandomVariable.infer_shape` works with `ShapeFeature`."""
    M_aet = iscalar("M")
    M_aet.tag.test_value = 2
    N_aet = iscalar("N")
    N_aet.tag.test_value = 3

    d_rv = dirichlet(aet.ones((M_aet, N_aet)), name="Gamma")

    fg = FunctionGraph(
        outputs=[d_rv],
        clone=False,
        features=[ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert M_aet in graph_inputs([s1])
    assert N_aet in graph_inputs([s2])


def test_poisson_samples():

    rv_numpy_tester(poisson)
    rv_numpy_tester(poisson, size=aet.as_tensor((2, 3)))

    test_lambda = np.array(10, dtype="int64")

    rv_numpy_tester(poisson, test_lambda)
    rv_numpy_tester(poisson, test_lambda, size=[2, 3])


def test_geometric_samples():
    test_p = np.array(0.1, dtype=config.floatX)

    rv_numpy_tester(geometric, test_p)
    rv_numpy_tester(geometric, test_p, size=[2, 3])


def test_hypergeometric_samples():
    test_ngood = np.array(10, dtype="int64")
    test_nbad = np.array(20, dtype="int64")
    test_nsample = np.array(5, dtype="int64")

    rv_numpy_tester(hypergeometric, test_ngood, test_nbad, test_nsample)
    rv_numpy_tester(hypergeometric, test_ngood, test_nbad, test_nsample, size=[2, 3])


def test_cauchy_samples():
    rv_numpy_tester(cauchy, test_fn=stats.cauchy.rvs)

    test_loc = np.array(10, dtype=config.floatX)
    test_scale = np.array(0.1, dtype=config.floatX)

    rv_numpy_tester(cauchy, test_loc, test_scale, test_fn=stats.cauchy.rvs)
    rv_numpy_tester(cauchy, test_loc, test_scale, size=[2, 3], test_fn=stats.cauchy.rvs)


def test_halfcauchy_samples():
    rv_numpy_tester(halfcauchy, test_fn=stats.halfcauchy.rvs)

    test_loc = np.array(10, dtype=config.floatX)
    test_scale = np.array(0.1, dtype=config.floatX)

    rv_numpy_tester(halfcauchy, test_loc, test_scale, test_fn=stats.halfcauchy.rvs)
    rv_numpy_tester(
        halfcauchy,
        test_loc,
        test_scale,
        size=[2, 3],
        test_fn=stats.halfcauchy.rvs,
    )


def test_invgamma_samples():
    test_loc = np.array(2, dtype=config.floatX)
    test_scale = np.array(2, dtype=config.floatX)

    rv_numpy_tester(
        invgamma, test_loc, test_scale, test_fn=partial(invgamma.rng_fn, None)
    )
    rv_numpy_tester(
        invgamma,
        test_loc,
        test_scale,
        size=[2, 3],
        test_fn=partial(invgamma.rng_fn, None),
    )


def test_wald_samples():
    test_mean = np.array(10, dtype=config.floatX)
    test_scale = np.array(1, dtype=config.floatX)

    rv_numpy_tester(wald, test_mean, test_scale)
    rv_numpy_tester(wald, test_mean, test_scale, size=[2, 3])


def test_truncexpon_samples():
    test_b = np.array(5, dtype=config.floatX)
    test_loc = np.array(0, dtype=config.floatX)
    test_scale = np.array(1, dtype=config.floatX)

    rv_numpy_tester(
        truncexpon,
        test_b,
        test_loc,
        test_scale,
        test_fn=partial(truncexpon.rng_fn, None),
    )
    rv_numpy_tester(
        truncexpon,
        test_b,
        test_loc,
        test_scale,
        size=[2, 3],
        test_fn=partial(truncexpon.rng_fn, None),
    )


def test_bernoulli_samples():
    test_p = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(bernoulli, test_p, test_fn=partial(bernoulli.rng_fn, None))
    rv_numpy_tester(
        bernoulli,
        test_p,
        size=[2, 3],
        test_fn=partial(bernoulli.rng_fn, None),
    )


def test_laplace_samples():
    test_loc = np.array(10, dtype=config.floatX)
    test_scale = np.array(5, dtype=config.floatX)

    rv_numpy_tester(laplace, test_loc, test_scale)
    rv_numpy_tester(laplace, test_loc, test_scale, size=[2, 3])


def test_binomial_samples():
    test_M = np.array(10, dtype="int64")
    test_p = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(binomial, test_M, test_p)
    rv_numpy_tester(binomial, test_M, test_p, size=[2, 3])


def test_nbinom_samples():
    test_M = np.array(10, dtype="int64")
    test_p = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(nbinom, test_M, test_p, test_fn=partial(nbinom.rng_fn, None))
    rv_numpy_tester(
        nbinom,
        test_M,
        test_p,
        size=[2, 3],
        test_fn=partial(nbinom.rng_fn, None),
    )


def test_betabinom_samples():
    test_M = np.array(10, dtype="int64")
    test_a = np.array(0.5, dtype=config.floatX)
    test_b = np.array(0.5, dtype=config.floatX)

    rv_numpy_tester(
        betabinom, test_M, test_a, test_b, test_fn=partial(betabinom.rng_fn, None)
    )
    rv_numpy_tester(
        betabinom,
        test_M,
        test_a,
        test_b,
        size=[2, 3],
        test_fn=partial(betabinom.rng_fn, None),
    )


def test_multinomial_samples():
    test_M = np.array(10, dtype="int64")
    test_p = np.array([0.7, 0.3], dtype=config.floatX)

    rv_numpy_tester(multinomial, test_M, test_p)
    rv_numpy_tester(
        multinomial,
        test_M,
        test_p,
        size=[2, 3],
    )

    rng_state = shared(
        np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    )

    test_M = np.array([10, 20], dtype="int64")
    test_p = np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX)

    res = multinomial(test_M, test_p, rng=rng_state).eval()
    exp_res = np.array([[10, 0], [0, 20]])
    assert np.array_equal(res, exp_res)

    res = multinomial(test_M, test_p, size=(3, 2), rng=rng_state).eval()
    exp_res = np.stack([exp_res] * 3)
    assert np.array_equal(res, exp_res)

    with pytest.raises(ValueError, match="shape mismatch.*"):
        multinomial.rng_fn(None, test_M, test_p, size=(1,))


def test_categorical_samples():

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))

    assert categorical.rng_fn(rng_state, np.array([1.0 / 3.0] * 3), size=10).shape == (
        10,
    )

    p = np.array([[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]], dtype=config.floatX)
    p = p / p.sum(axis=-1)

    assert categorical.rng_fn(rng_state, p, size=None).shape == p.shape[:-1]

    with raises(ValueError):
        categorical.rng_fn(rng_state, p, size=10)

    assert categorical.rng_fn(rng_state, p, size=(10, 3)).shape == (10, 3)
    assert categorical.rng_fn(rng_state, p, size=(10, 2, 3)).shape == (10, 2, 3)

    res = categorical(p)
    assert np.array_equal(get_test_value(res), np.arange(3))

    res = categorical(p, size=(10, 3))
    exp_res = np.tile(np.arange(3), (10, 1))
    assert np.array_equal(get_test_value(res), exp_res)

    res = categorical(p, size=(10, 2, 3))
    exp_res = np.tile(np.arange(3), (10, 2, 1))
    assert np.array_equal(get_test_value(res), exp_res)


def test_polyagamma_samples():

    _ = importorskip("pypolyagamma")

    # Sampled values should be scalars
    a = np.array(1.1, dtype=config.floatX)
    b = np.array(-10.5, dtype=config.floatX)
    pg_rv = polyagamma(a, b)
    assert get_test_value(pg_rv).shape == ()

    pg_rv = polyagamma(a, b, size=[1])
    assert get_test_value(pg_rv).shape == (1,)

    pg_rv = polyagamma(a, b, size=[2, 3])
    bcast_smpl = get_test_value(pg_rv)
    assert bcast_smpl.shape == (2, 3)
    # Make sure they're not all equal
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)

    a = np.array([1.1, 3], dtype=config.floatX)
    b = np.array(-10.5, dtype=config.floatX)
    pg_rv = polyagamma(a, b)
    bcast_smpl = get_test_value(pg_rv)
    assert bcast_smpl.shape == (2,)
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)

    pg_rv = polyagamma(a, b, size=(3, 2))
    bcast_smpl = get_test_value(pg_rv)
    assert bcast_smpl.shape == (3, 2)
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)


def test_random_integer_samples():

    rv_numpy_tester(randint, 10, None)
    rv_numpy_tester(randint, 0, 1)
    rv_numpy_tester(randint, 0, 1, size=[3])
    rv_numpy_tester(randint, [0, 1, 2], 5)
    rv_numpy_tester(randint, [0, 1, 2], 5, size=[3, 3])
    rv_numpy_tester(randint, [0], [5], size=[1])
    rv_numpy_tester(randint, aet.as_tensor_variable([-1]), [1], size=[1])
    rv_numpy_tester(
        randint, aet.as_tensor_variable([-1]), [1], size=aet.as_tensor_variable([1])
    )


def test_choice_samples():
    with raises(NotImplementedError):
        choice._shape_from_params(np.asarray(5))

    rv_numpy_tester(choice, np.asarray(5))
    rv_numpy_tester(choice, np.array([1.0, 5.0], dtype=config.floatX))
    rv_numpy_tester(choice, np.asarray(5), 3)

    with raises(ValueError):
        rv_numpy_tester(choice, np.array([[1, 2], [3, 4]]))

    rv_numpy_tester(choice, [1, 2, 3], 1)
    rv_numpy_tester(choice, [1, 2, 3], 1, p=aet.as_tensor([1 / 3.0, 1 / 3.0, 1 / 3.0]))
    rv_numpy_tester(choice, [1, 2, 3], (10, 2), replace=True)
    rv_numpy_tester(choice, aet.as_tensor_variable([1, 2, 3]), 2, replace=True)


def test_permutation_samples():
    rv_numpy_tester(
        permutation, np.asarray(5), test_fn=lambda x: np.random.permutation(x.item())
    )
    rv_numpy_tester(permutation, [1, 2, 3])
    rv_numpy_tester(permutation, [[1, 2], [3, 4]])
    rv_numpy_tester(permutation, np.array([1.0, 2.0, 3.0], dtype=config.floatX))


@config.change_flags(compute_test_value="off")
def test_pickle():
    # This is an interesting `Op` case, because it has `None` types and a
    # conditional dtype
    sample_a = choice(5, size=(2, 3))

    a_pkl = pickle.dumps(sample_a)
    a_unpkl = pickle.loads(a_pkl)

    assert a_unpkl.owner.op._props() == sample_a.owner.op._props()
