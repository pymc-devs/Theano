r"""
`Op`\s that have their python implementations taken from SciPy.

As SciPy is not always available, we treat them separately.
"""

import os
import warnings

import numpy as np
import scipy.special
import scipy.stats

from aesara.configdefaults import config
from aesara.gradient import grad_not_implemented
from aesara.scalar.basic import (
    BinaryScalarOp,
    ScalarOp,
    UnaryScalarOp,
    complex_types,
    discrete_types,
    exp,
    float64,
    float_types,
    log,
    log1p,
    true_div,
    upcast,
    upgrade_to_float,
    upgrade_to_float64,
    upgrade_to_float_no_complex,
)


class Erf(UnaryScalarOp):
    nfunc_spec = ("scipy.special.erf", 1, 1)

    def impl(self, x):
        return scipy.special.erf(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            2.0 / np.sqrt(np.pi), dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (gz * cst * exp(-x * x),)

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = erf(({cast}){x});"


erf = Erf(upgrade_to_float, name="erf")


class Erfc(UnaryScalarOp):
    nfunc_spec = ("scipy.special.erfc", 1, 1)

    def impl(self, x):
        return scipy.special.erfc(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            2.0 / np.sqrt(np.pi), dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (-gz * cst * exp(-x * x),)

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = erfc(({cast}){x});"


# scipy.special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name="erfc")


class Erfcx(UnaryScalarOp):
    """
    Implements the scaled complementary error function exp(x**2)*erfc(x) in a
    numerically stable way for large x. This is useful for calculating things
    like log(erfc(x)) = log(erfcx(x)) - x ** 2 without causing underflow.
    Should only be used if x is known to be large and positive, as using
    erfcx(x) for large negative x may instead introduce overflow problems.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU an optimization will replace it with a gpu version.

    """

    nfunc_spec = ("scipy.special.erfcx", 1, 1)

    def impl(self, x):
        return scipy.special.erfcx(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            2.0 / np.sqrt(np.pi), dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (gz * (-cst + (2.0 * x) * erfcx(x)),)

    def c_header_dirs(self, **kwargs):
        # Using the Faddeeva.hh (c++) header for Faddeevva.cc
        res = super().c_header_dirs(**kwargs) + [
            os.path.join(os.path.dirname(__file__), "c_code")
        ]
        return res

    def c_support_code(self, **kwargs):
        # Using Faddeeva.cc source file from: http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
        with open(
            os.path.join(os.path.dirname(__file__), "c_code", "Faddeeva.cc")
        ) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return f"{z} = ({dtype}) Faddeeva::erfcx({x});"

        raise NotImplementedError("type not supported", type)


erfcx = Erfcx(upgrade_to_float_no_complex, name="erfcx")


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU, an optimization will replace it with a GPU version.

    (TODO) Find a C implementation of erfinv for CPU.
    """

    nfunc_spec = ("scipy.special.erfinv", 1, 1)

    def impl(self, x):
        return scipy.special.erfinv(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            np.sqrt(np.pi) / 2.0, dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (gz * cst * exp(erfinv(x) ** 2),)

    # TODO: erfinv() is not provided by the C standard library
    # def c_code(self, node, name, inp, out, sub):
    #    x, = inp
    #    z, = out
    #    if node.inputs[0].type in complex_types:
    #        raise NotImplementedError('type not supported', type)
    #    return "%(z)s = erfinv(%(x)s);" % locals()


erfinv = Erfinv(upgrade_to_float_no_complex, name="erfinv")


class Erfcinv(UnaryScalarOp):
    nfunc_spec = ("scipy.special.erfcinv", 1, 1)

    def impl(self, x):
        return scipy.special.erfcinv(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(
            np.sqrt(np.pi) / 2.0, dtype=upcast(x.type.dtype, gz.type.dtype)
        )
        return (-gz * cst * exp(erfcinv(x) ** 2),)

    # TODO: erfcinv() is not provided by the C standard library
    # def c_code(self, node, name, inp, out, sub):
    #    x, = inp
    #    z, = out
    #    if node.inputs[0].type in complex_types:
    #        raise NotImplementedError('type not supported', type)
    #    return "%(z)s = erfcinv(%(x)s);" % locals()


erfcinv = Erfcinv(upgrade_to_float_no_complex, name="erfcinv")


class Gamma(UnaryScalarOp):
    nfunc_spec = ("scipy.special.gamma", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.gamma(x)

    def impl(self, x):
        return Gamma.st_impl(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * gamma(x) * psi(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in float_types:
            return f"""{z} = tgamma({x});"""
        raise NotImplementedError("only floating point is implemented")


gamma = Gamma(upgrade_to_float, name="gamma")


class GammaLn(UnaryScalarOp):
    """
    Log gamma function.

    """

    nfunc_spec = ("scipy.special.gammaln", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.gammaln(x)

    def impl(self, x):
        return GammaLn.st_impl(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        # no c code for complex
        # [u]int* will be casted to float64 before computation
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("gammaln complex c code is not implemented")
        # For some reason, on the GPU, uint64 inputs don't get casted
        # automatically to float64. This make the compilation crash
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"""{z} = lgamma(({cast}){x});"""


gammaln = GammaLn(upgrade_to_float, name="gammaln")


class Psi(UnaryScalarOp):
    """
    Derivative of log gamma function.

    """

    nfunc_spec = ("scipy.special.psi", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.psi(x)

    def impl(self, x):
        return Psi.st_impl(x)

    def L_op(self, inputs, outputs, grads):
        (x,) = inputs
        (gz,) = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * tri_gamma(x)]

    def c_support_code(self, **kwargs):
        return """
            // For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef ga_double
            #define ga_double double
            #endif

            #ifndef _PSIFUNCDEFINED
            #define _PSIFUNCDEFINED
            DEVICE double _psi(ga_double x) {

            /*taken from
            Bernardo, J. M. (1976). Algorithm AS 103:
            Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
            http://www.uv.es/~bernardo/1976AppStatist.pdf */

            ga_double y, R, psi_ = 0;
            ga_double S  = 1.0e-5;
            ga_double C = 8.5;
            ga_double S3 = 8.333333333e-2;
            ga_double S4 = 8.333333333e-3;
            ga_double S5 = 3.968253968e-3;
            ga_double D1 = -0.5772156649;

            y = x;

            if (y <= 0.0)
               return psi_;

            if (y <= S)
                return D1 - 1.0/y;

            while (y < C) {
                psi_ = psi_ - 1.0 / y;
                y = y + 1;
            }

            R = 1.0 / y;
            psi_ = psi_ + log(y) - .5 * R ;
            R= R*R;
            psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

            return psi_;
            }
            #endif
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                _psi({x});"""
        raise NotImplementedError("only floating point is implemented")


psi = Psi(upgrade_to_float, name="psi")


class TriGamma(UnaryScalarOp):
    """
    Second derivative of log gamma function.

    """

    @staticmethod
    def st_impl(x):
        return scipy.special.polygamma(1, x)

    def impl(self, x):
        return TriGamma.st_impl(x)

    def grad(self, inputs, outputs_gradients):
        raise NotImplementedError()

    def c_support_code(self, **kwargs):
        # The implementation has been copied from
        # http://people.sc.fsu.edu/~jburkardt/cpp_src/asa121/asa121.html
        return """
            // For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef ga_double
            #define ga_double double
            #endif

            #ifndef _TRIGAMMAFUNCDEFINED
            #define _TRIGAMMAFUNCDEFINED

            DEVICE double _tri_gamma(ga_double x) {

                double a = 0.0001;
                double b = 5.0;
                double b2 =  0.1666666667;
                double b4 = -0.03333333333;
                double b6 =  0.02380952381;
                double b8 = -0.03333333333;
                double value;
                double y;
                double z;

                if (x <= 0) {
                    return 0.0;
                }

                if ( x <= a ) {
                    value = 1.0 / x / x;
                    return value;
                }

                value = 0.0;
                z = x;

                while ( z < b ) {
                    value += 1.0 / z / z;
                    z += 1.0;
                }

                y = 1.0 / z / z;

                value +=  0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8 )))) / z;

                return value;
            }
            #endif
            """

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                _tri_gamma({x});"""
        raise NotImplementedError("only floating point is implemented")


tri_gamma = TriGamma(upgrade_to_float, name="tri_gamma")


class Chi2SF(BinaryScalarOp):
    """
    Compute (1 - chi2_cdf(x))
        ie. chi2 pvalue (chi2 'survival function')
    """

    nfunc_spec = ("scipy.stats.chi2.sf", 2, 1)

    @staticmethod
    def st_impl(x, k):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, x, k):
        return Chi2SF.st_impl(x, k)

    def c_support_code(self, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), "c_code", "gamma.c")) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        x, k = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return (
                """%(z)s =
                (%(dtype)s) 1 - GammaP(%(k)s/2., %(x)s/2.);"""
                % locals()
            )
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


chi2sf = Chi2SF(upgrade_to_float64, name="chi2sf")


class GammaInc(BinaryScalarOp):
    """
    Compute the regularized lower gamma function (P).
    """

    nfunc_spec = ("scipy.special.gammainc", 2, 1)

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammainc(k, x)

    def impl(self, k, x):
        return GammaInc.st_impl(k, x)

    def grad(self, inputs, grads):
        (k, x) = inputs
        (gz,) = grads
        return [
            gz * gammainc_der(k, x),
            gz * exp(-x + (k - 1) * log(x) - gammaln(k)),
        ]

    def c_support_code(self, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), "c_code", "gamma.c")) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return (
                """%(z)s =
                (%(dtype)s) GammaP(%(k)s, %(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammainc = GammaInc(upgrade_to_float, name="gammainc")


class GammaIncC(BinaryScalarOp):
    """
    Compute the regularized upper gamma function (Q).
    """

    nfunc_spec = ("scipy.special.gammaincc", 2, 1)

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammaincc(k, x)

    def impl(self, k, x):
        return GammaIncC.st_impl(k, x)

    def grad(self, inputs, grads):
        (k, x) = inputs
        (gz,) = grads
        return [
            gz * gammaincc_der(k, x),
            gz * -exp(-x + (k - 1) * log(x) - gammaln(k)),
        ]

    def c_support_code(self, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), "c_code", "gamma.c")) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return (
                """%(z)s =
                (%(dtype)s) GammaQ(%(k)s, %(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammaincc = GammaIncC(upgrade_to_float, name="gammaincc")


class GammaIncDer(BinaryScalarOp):
    """
    Gradient of the the regularized lower gamma function (P) wrt to the first
    argument (k, a.k.a. alpha). Adapted from STAN `grad_reg_lower_inc_gamma.hpp`

    Reference: Gautschi, W. (1979). A computational procedure for incomplete gamma functions.
    ACM Transactions on Mathematical Software (TOMS), 5(4), 466-481.
    """

    def impl(self, k, x):

        if x == 0:
            return 0

        sqrt_exp = -756 - x ** 2 + 60 * x
        if (
            (k < 0.8 and x > 15)
            or (k < 12 and x > 30)
            or (sqrt_exp > 0 and k < np.sqrt(sqrt_exp))
        ):
            return -GammaIncCDer.st_impl(k, x)

        precision = 1e-10
        max_iters = int(1e5)

        log_x = np.log(x)
        log_gamma_k_plus_1 = scipy.special.gammaln(k + 1)

        k_plus_n = k
        log_gamma_k_plus_n_plus_1 = log_gamma_k_plus_1
        sum_a = 0.0
        for n in range(0, max_iters + 1):
            term = np.exp(k_plus_n * log_x - log_gamma_k_plus_n_plus_1)
            sum_a += term

            if term <= precision:
                break

            log_gamma_k_plus_n_plus_1 += np.log1p(k_plus_n)
            k_plus_n += 1

        if n >= max_iters:
            warnings.warn(
                f"gammainc_der did not converge after {n} iterations",
                RuntimeWarning,
            )
            return np.nan

        k_plus_n = k
        log_gamma_k_plus_n_plus_1 = log_gamma_k_plus_1
        sum_b = 0.0
        for n in range(0, max_iters + 1):
            term = np.exp(
                k_plus_n * log_x - log_gamma_k_plus_n_plus_1
            ) * scipy.special.digamma(k_plus_n + 1)
            sum_b += term

            if term <= precision and n >= 1:  # Require at least two iterations
                return np.exp(-x) * (log_x * sum_a - sum_b)

            log_gamma_k_plus_n_plus_1 += np.log1p(k_plus_n)
            k_plus_n += 1

        warnings.warn(
            f"gammainc_der did not converge after {n} iterations",
            RuntimeWarning,
        )
        return np.nan


gammainc_der = GammaIncDer(upgrade_to_float, name="gammainc_der")


class GammaIncCDer(BinaryScalarOp):
    """
    Gradient of the the regularized upper gamma function (Q) wrt to the first
    argument (k, a.k.a. alpha). Adapted from STAN `grad_reg_inc_gamma.hpp`
    """

    @staticmethod
    def st_impl(k, x):
        gamma_k = scipy.special.gamma(k)
        digamma_k = scipy.special.digamma(k)
        log_x = np.log(x)

        # asymptotic expansion http://dlmf.nist.gov/8.11#E2
        if (x >= k) and (x >= 8):
            S = 0
            k_minus_one_minus_n = k - 1
            fac = k_minus_one_minus_n
            dfac = 1
            xpow = x
            delta = dfac / xpow

            for n in range(1, 10):
                k_minus_one_minus_n -= 1
                S += delta
                xpow *= x
                dfac = k_minus_one_minus_n * dfac + fac
                fac *= k_minus_one_minus_n
                delta = dfac / xpow
                if np.isinf(delta):
                    warnings.warn(
                        "gammaincc_der did not converge",
                        RuntimeWarning,
                    )
                    return np.nan

            return (
                scipy.special.gammaincc(k, x) * (log_x - digamma_k)
                + np.exp(-x + (k - 1) * log_x) * S / gamma_k
            )

        # gradient of series expansion http://dlmf.nist.gov/8.7#E3
        else:
            log_precision = np.log(1e-6)
            max_iters = int(1e5)
            S = 0
            log_s = 0.0
            s_sign = 1
            log_delta = log_s - 2 * np.log(k)
            for n in range(1, max_iters + 1):
                S += np.exp(log_delta) if s_sign > 0 else -np.exp(log_delta)
                s_sign = -s_sign
                log_s += log_x - np.log(n)
                log_delta = log_s - 2 * np.log(n + k)

                if np.isinf(log_delta):
                    warnings.warn(
                        "gammaincc_der did not converge",
                        RuntimeWarning,
                    )
                    return np.nan

                if log_delta <= log_precision:
                    return (
                        scipy.special.gammainc(k, x) * (digamma_k - log_x)
                        + np.exp(k * log_x) * S / gamma_k
                    )

            warnings.warn(
                f"gammaincc_der did not converge after {n} iterations",
                RuntimeWarning,
            )
            return np.nan

    def impl(self, k, x):
        return self.st_impl(k, x)


gammaincc_der = GammaIncCDer(upgrade_to_float, name="gammaincc_der")


class GammaU(BinaryScalarOp):
    """
    compute the upper incomplete gamma function.
    """

    # Note there is no basic SciPy version so no nfunc_spec.

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammaincc(k, x) * scipy.special.gamma(k)

    def impl(self, k, x):
        return GammaU.st_impl(k, x)

    def c_support_code(self, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), "c_code", "gamma.c")) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return (
                """%(z)s =
                (%(dtype)s) upperGamma(%(k)s, %(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammau = GammaU(upgrade_to_float, name="gammau")


class GammaL(BinaryScalarOp):
    """
    Compute the lower incomplete gamma function.
    """

    # Note there is no basic SciPy version so no nfunc_spec.

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammainc(k, x) * scipy.special.gamma(k)

    def impl(self, k, x):
        return GammaL.st_impl(k, x)

    def c_support_code(self, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), "c_code", "gamma.c")) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            dtype = "npy_" + node.outputs[0].dtype
            return (
                """%(z)s =
                (%(dtype)s) lowerGamma(%(k)s, %(x)s);"""
                % locals()
            )
        raise NotImplementedError("only floatingpoint is implemented")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammal = GammaL(upgrade_to_float, name="gammal")


class Jv(BinaryScalarOp):
    """
    Bessel function of the first kind of order v (real).
    """

    nfunc_spec = ("scipy.special.jv", 2, 1)

    @staticmethod
    def st_impl(v, x):
        return scipy.special.jv(v, x)

    def impl(self, v, x):
        return self.st_impl(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, v),
            gz * (jv(v - 1, x) - jv(v + 1, x)) / 2.0,
        ]


jv = Jv(upgrade_to_float, name="jv")


class J1(UnaryScalarOp):
    """
    Bessel function of the first kind of order 1.
    """

    nfunc_spec = ("scipy.special.j1", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.j1(x)

    def impl(self, x):
        return self.st_impl(x)

    def grad(self, inputs, grads):
        (x,) = inputs
        (gz,) = grads
        return [gz * (j0(x) - jv(2, x)) / 2.0]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                j1({x});"""
        raise NotImplementedError("only floating point is implemented")


j1 = J1(upgrade_to_float, name="j1")


class J0(UnaryScalarOp):
    """
    Bessel function of the first kind of order 0.
    """

    nfunc_spec = ("scipy.special.j0", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.j0(x)

    def impl(self, x):
        return self.st_impl(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * -1 * j1(x)]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        if node.inputs[0].type in float_types:
            return f"""{z} =
                j0({x});"""
        raise NotImplementedError("only floating point is implemented")


j0 = J0(upgrade_to_float, name="j0")


class Iv(BinaryScalarOp):
    """
    Modified Bessel function of the first kind of order v (real).
    """

    nfunc_spec = ("scipy.special.iv", 2, 1)

    @staticmethod
    def st_impl(v, x):
        return scipy.special.iv(v, x)

    def impl(self, v, x):
        return self.st_impl(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        (gz,) = grads
        return [
            grad_not_implemented(self, 0, v),
            gz * (iv(v - 1, x) + iv(v + 1, x)) / 2.0,
        ]


iv = Iv(upgrade_to_float, name="iv")


class I1(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 1.
    """

    nfunc_spec = ("scipy.special.i1", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.i1(x)

    def impl(self, x):
        return self.st_impl(x)

    def grad(self, inputs, grads):
        (x,) = inputs
        (gz,) = grads
        return [gz * (i0(x) + iv(2, x)) / 2.0]


i1 = I1(upgrade_to_float, name="i1")


class I0(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 0.
    """

    nfunc_spec = ("scipy.special.i0", 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.i0(x)

    def impl(self, x):
        return self.st_impl(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * i1(x)]


i0 = I0(upgrade_to_float, name="i0")


class Sigmoid(UnaryScalarOp):
    """
    Logistic sigmoid function (1 / (1 + exp(x)), also known as expit or inverse logit
    """

    nfunc_spec = ("scipy.special.expit", 1, 1)

    def impl(self, x):
        return scipy.special.expit(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        y = sigmoid(x)
        rval = gz * y * (1.0 - y)

        assert rval.type.dtype.find("float") != -1

        return [rval]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            if node.inputs[0].type == float64:
                return f"""{z} = 1.0 / (1.0 + exp(-{x}));"""
            else:
                return f"""{z} = 1.0f / (1.0f + exp(-{x}));"""
        else:
            raise NotImplementedError("only floatingpoint is implemented")

    def c_code_cache_version(self):
        v = super().c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v


sigmoid = Sigmoid(upgrade_to_float, name="sigmoid")


class Softplus(UnaryScalarOp):
    r"""
    Compute log(1 + exp(x)), also known as softplus or log1pexp

    This function is numerically more stable than the naive approach.

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    References
    ----------
    .. [Machler2012] Martin Mächler (2012).
        "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr package"
    """

    @staticmethod
    def static_impl(x):
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        not_int8 = str(getattr(x, "dtype", "")) not in ("int8", "uint8")
        if x < -37.0:
            return np.exp(x) if not_int8 else np.exp(x, signature="f")
        elif x < 18.0:
            return (
                np.log1p(np.exp(x)) if not_int8 else np.log1p(np.exp(x, signature="f"))
            )
        elif x < 33.3:
            return x + np.exp(-x) if not_int8 else x + np.exp(-x, signature="f")
        else:
            return x

    def impl(self, x):
        return Softplus.static_impl(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * sigmoid(x)]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        # The boundary constants were obtained by looking at the output of
        # python commands like:
        # import numpy, aesara
        # dt='float32'  # or float64
        #  for i in range(750):
        #      print i, repr(numpy.log1p(numpy.exp(_asarray([i,-i], dtype=dt))))
        # the upper boundary check prevents us from generating inf, whereas the
        # the lower boundary check prevents using exp when the result will be 0 anyway.
        # The intermediate constants are taken from Machler (2012).

        # We use the float32 limits for float16 for now as the
        # computation will happen in float32 anyway.
        if node.inputs[0].type in float_types:
            if node.inputs[0].type == float64:
                return (
                    """
                    %(z)s = (
                        %(x)s < -745.0 ? 0.0 :
                        %(x)s < -37.0 ? exp(%(x)s) :
                        %(x)s < 18.0 ? log1p(exp(%(x)s)) :
                        %(x)s < 33.3 ? %(x)s + exp(-%(x)s) :
                        %(x)s
                    );
                    """
                    % locals()
                )
            else:
                return (
                    """
                    %(z)s = (
                        %(x)s < -103.0f ? 0.0 :
                        %(x)s < -37.0f ? exp(%(x)s) :
                        %(x)s < 18.0f ? log1p(exp(%(x)s)) :
                        %(x)s < 33.3f ? %(x)s + exp(-%(x)s) :
                        %(x)s
                    );
                    """
                    % locals()
                )
        else:
            raise NotImplementedError("only floatingpoint is implemented")

    def c_code_cache_version(self):
        v = super().c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v


softplus = Softplus(upgrade_to_float, name="scalar_softplus")


class Log1mexp(UnaryScalarOp):
    r"""
    Compute log(1 - exp(x)), also known as log1mexp

    This function is numerically more stable than the naive approach.

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    References
    ----------
    .. [Machler2012] Martin Mächler (2012).
        "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr package"
    """

    @staticmethod
    def static_impl(x):
        if x < np.log(0.5):
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))

    def impl(self, x):
        return Log1mexp.static_impl(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return [gz * true_div(1.0, 1.0 - exp(-x))]

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out

        if node.inputs[0].type in float_types:
            if node.inputs[0].type == float64:
                return f"{z} = {x} < -0.6931471805599453 ? log1p(-exp({x})) : log(-expm1({x}));"
            else:
                return f"{z} = {x} < -0.6931471805599453f ? log1p(-exp({x})) : log(-expm1({x}));"
        else:
            raise NotImplementedError("only floating point is implemented")


log1mexp = Log1mexp(upgrade_to_float, name="scalar_log1mexp")


class BetaInc(ScalarOp):
    """
    Regularized incomplete beta function
    """

    nin = 3
    nfunc_spec = ("scipy.special.betainc", 3, 1)

    def impl(self, a, b, x):
        return scipy.special.betainc(a, b, x)

    def grad(self, inp, grads):
        a, b, x = inp
        (gz,) = grads

        return [
            gz * betainc_der(a, b, x, True),
            gz * betainc_der(a, b, x, False),
            gz
            * exp(
                log1p(-x) * (b - 1)
                + log(x) * (a - 1)
                - (gammaln(a) + gammaln(b) - gammaln(a + b))
            ),
        ]


betainc = BetaInc(upgrade_to_float_no_complex, name="betainc")


class BetaIncDer(ScalarOp):
    """
    Gradient of the regularized incomplete beta function wrt to the first
    argument (alpha) or the second argument (beta), depending on whether the
    fourth argument to betainc_der is `True` or `False`, respectively.

    Reference: Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
    Journal of Statistical Software, 3(1), 1-20.
    """

    nin = 4

    def impl(self, p, q, x, wrtp):
        def _betainc_a_n(f, p, q, n):
            """
            Numerator (a_n) of the nth approximant of the continued fraction
            representation of the regularized incomplete beta function
            """

            if n == 1:
                return p * f * (q - 1) / (q * (p + 1))

            p2n = p + 2 * n
            F1 = p ** 2 * f ** 2 * (n - 1) / (q ** 2)
            F2 = (
                (p + q + n - 2)
                * (p + n - 1)
                * (q - n)
                / ((p2n - 3) * (p2n - 2) ** 2 * (p2n - 1))
            )

            return F1 * F2

        def _betainc_b_n(f, p, q, n):
            """
            Offset (b_n) of the nth approximant of the continued fraction
            representation of the regularized incomplete beta function
            """
            pf = p * f
            p2n = p + 2 * n

            N1 = 2 * (pf + 2 * q) * n * (n + p - 1) + p * q * (p - 2 - pf)
            D1 = q * (p2n - 2) * p2n

            return N1 / D1

        def _betainc_da_n_dp(f, p, q, n):
            """
            Derivative of a_n wrt p
            """

            if n == 1:
                return -p * f * (q - 1) / (q * (p + 1) ** 2)

            pp = p ** 2
            ppp = pp * p
            p2n = p + 2 * n

            N1 = -(n - 1) * f ** 2 * pp * (q - n)
            N2a = (-8 + 8 * p + 8 * q) * n ** 3
            N2b = (16 * pp + (-44 + 20 * q) * p + 26 - 24 * q) * n ** 2
            N2c = (10 * ppp + (14 * q - 46) * pp + (-40 * q + 66) * p - 28 + 24 * q) * n
            N2d = 2 * pp ** 2 + (-13 + 3 * q) * ppp + (-14 * q + 30) * pp
            N2e = (-29 + 19 * q) * p + 10 - 8 * q

            D1 = q ** 2 * (p2n - 3) ** 2
            D2 = (p2n - 2) ** 3 * (p2n - 1) ** 2

            return (N1 / D1) * (N2a + N2b + N2c + N2d + N2e) / D2

        def _betainc_da_n_dq(f, p, q, n):
            """
            Derivative of a_n wrt q
            """
            if n == 1:
                return p * f / (q * (p + 1))

            p2n = p + 2 * n
            F1 = (p ** 2 * f ** 2 / (q ** 2)) * (n - 1) * (p + n - 1) * (2 * q + p - 2)
            D1 = (p2n - 3) * (p2n - 2) ** 2 * (p2n - 1)

            return F1 / D1

        def _betainc_db_n_dp(f, p, q, n):
            """
            Derivative of b_n wrt p
            """
            p2n = p + 2 * n
            pp = p ** 2
            q4 = 4 * q
            p4 = 4 * p

            F1 = (p * f / q) * (
                (-p4 - q4 + 4) * n ** 2 + (p4 - 4 + q4 - 2 * pp) * n + pp * q
            )
            D1 = (p2n - 2) ** 2 * p2n ** 2

            return F1 / D1

        def _betainc_db_n_dq(f, p, q, n):
            """
            Derivative of b_n wrt to q
            """
            p2n = p + 2 * n
            return -(p ** 2 * f) / (q * (p2n - 2) * p2n)

        # Input validation
        if not (0 <= x <= 1) or p < 0 or q < 0:
            return np.nan

        if x > (p / (p + q)):
            return -self.impl(q, p, 1 - x, not wrtp)

        min_iters = 3
        max_iters = 200
        err_threshold = 1e-12

        derivative_old = 0

        Am2, Am1 = 1, 1
        Bm2, Bm1 = 0, 1
        dAm2, dAm1 = 0, 0
        dBm2, dBm1 = 0, 0

        f = (q * x) / (p * (1 - x))
        K = np.exp(
            p * np.log(x)
            + (q - 1) * np.log1p(-x)
            - np.log(p)
            - scipy.special.betaln(p, q)
        )
        if wrtp:
            dK = (
                np.log(x)
                - 1 / p
                + scipy.special.digamma(p + q)
                - scipy.special.digamma(p)
            )
        else:
            dK = np.log1p(-x) + scipy.special.digamma(p + q) - scipy.special.digamma(q)

        for n in range(1, max_iters + 1):
            a_n_ = _betainc_a_n(f, p, q, n)
            b_n_ = _betainc_b_n(f, p, q, n)
            if wrtp:
                da_n = _betainc_da_n_dp(f, p, q, n)
                db_n = _betainc_db_n_dp(f, p, q, n)
            else:
                da_n = _betainc_da_n_dq(f, p, q, n)
                db_n = _betainc_db_n_dq(f, p, q, n)

            A = a_n_ * Am2 + b_n_ * Am1
            B = a_n_ * Bm2 + b_n_ * Bm1
            dA = da_n * Am2 + a_n_ * dAm2 + db_n * Am1 + b_n_ * dAm1
            dB = da_n * Bm2 + a_n_ * dBm2 + db_n * Bm1 + b_n_ * dBm1

            Am2, Am1 = Am1, A
            Bm2, Bm1 = Bm1, B
            dAm2, dAm1 = dAm1, dA
            dBm2, dBm1 = dBm1, dB

            if n < min_iters - 1:
                continue

            F1 = A / B
            F2 = (dA - F1 * dB) / B
            derivative = K * (F1 * dK + F2)

            errapx = abs(derivative_old - derivative)
            d_errapx = errapx / max(err_threshold, abs(derivative))
            derivative_old = derivative

            if d_errapx <= err_threshold:
                return derivative

        warnings.warn(
            f"betainc_der did not converge after {n} iterations",
            RuntimeWarning,
        )
        return np.nan


betainc_der = BetaIncDer(upgrade_to_float_no_complex, name="betainc_der")
