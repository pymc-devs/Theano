import distutils.spawn
import errno
import logging
import os
import platform
import re
import socket
import sys
import textwrap

import numpy as np

import aesara
import aesara.configparser
from aesara.configparser import (
    BoolParam,
    ConfigParam,
    ContextsParam,
    DeviceParam,
    EnumStr,
    FloatParam,
    IntParam,
    StrParam,
)
from aesara.utils import (
    LOCAL_BITWIDTH,
    PYTHON_INT_BITWIDTH,
    call_subprocess_Popen,
    maybe_add_to_os_environ_pathlist,
    output_subprocess_Popen,
)


_logger = logging.getLogger("aesara.configdefaults")


def get_cuda_root():
    # We look for the cuda path since we need headers from there
    v = os.getenv("CUDA_ROOT", "")
    if v:
        return v
    v = os.getenv("CUDA_PATH", "")
    if v:
        return v
    s = os.getenv("PATH")
    if not s:
        return ""
    for dir in s.split(os.path.pathsep):
        if os.path.exists(os.path.join(dir, "nvcc")):
            return os.path.dirname(os.path.abspath(dir))
    return ""


def default_cuda_include():
    if config.cuda__root:
        return os.path.join(config.cuda__root, "include")
    return ""


def default_dnn_base_path():
    # We want to default to the cuda root if cudnn is installed there
    root = config.cuda__root
    # The include doesn't change location between OS.
    if root and os.path.exists(os.path.join(root, "include", "cudnn.h")):
        return root
    return ""


def default_dnn_inc_path():
    if config.dnn__base_path != "":
        return os.path.join(config.dnn__base_path, "include")
    return ""


def default_dnn_lib_path():
    if config.dnn__base_path != "":
        if sys.platform == "win32":
            path = os.path.join(config.dnn__base_path, "lib", "x64")
        elif sys.platform == "darwin":
            path = os.path.join(config.dnn__base_path, "lib")
        else:
            # This is linux
            path = os.path.join(config.dnn__base_path, "lib64")
        return path
    return ""


def default_dnn_bin_path():
    if config.dnn__base_path != "":
        if sys.platform == "win32":
            return os.path.join(config.dnn__base_path, "bin")
        else:
            return config.dnn__library_path
    return ""


def _filter_mode(val):
    # Do not add FAST_RUN_NOGC to this list (nor any other ALL CAPS shortcut).
    # The way to get FAST_RUN_NOGC is with the flag 'linker=c|py_nogc'.
    # The old all capital letter way of working is deprecated as it is not
    # scalable.
    str_options = [
        "Mode",
        "DebugMode",
        "FAST_RUN",
        "NanGuardMode",
        "FAST_COMPILE",
        "DEBUG_MODE",
        "JAX",
        "NUMBA",
    ]
    if val in str_options:
        return val
    # This can be executed before Aesara is completely imported, so
    # aesara.compile.mode.Mode is not always available.
    # Instead of isinstance(val, aesara.compile.mode.Mode),
    # we can inspect the __mro__ of the object!
    for type_ in type(val).__mro__:
        if "aesara.compile.mode.Mode" in str(type_):
            return val
    raise ValueError(
        f"Expected one of {str_options}, or an instance of aesara.compile.mode.Mode. "
        f"Instead got: {val}."
    )


def _warn_cxx(val):
    """We only support clang++ as otherwise we hit strange g++/OSX bugs."""
    if sys.platform == "darwin" and val and "clang++" not in val:
        _logger.warning(
            "Only clang++ is supported. With g++,"
            " we end up with strange g++/OSX bugs."
        )
    return True


def _split_version(version):
    """
    Take version as a dot-separated string, return a tuple of int
    """
    return tuple(int(i) for i in version.split("."))


def _warn_default(version):
    """
    Return True iff we should warn about bugs fixed after a given version.
    """
    if config.warn__ignore_bug_before == "None":
        return True
    if config.warn__ignore_bug_before == "all":
        return False
    if _split_version(config.warn__ignore_bug_before) >= _split_version(version):
        return False
    return True


def _good_seem_param(seed):
    if seed == "random":
        return True
    try:
        int(seed)
    except Exception:
        return False
    return True


def _is_valid_check_preallocated_output_param(param):
    if not isinstance(param, str):
        return False
    valid = [
        "initial",
        "previous",
        "c_contiguous",
        "f_contiguous",
        "strided",
        "wrong_size",
        "ALL",
        "",
    ]
    for p in param.split(":"):
        if p not in valid:
            return False
    return True


def _timeout_default():
    return config.compile__wait * 24


def _filter_vm_lazy(val):
    if val == "False" or val is False:
        return False
    elif val == "True" or val is True:
        return True
    elif val == "None" or val is None:
        return None
    else:
        raise ValueError(
            "Valid values for an vm__lazy parameter "
            f"should be None, False or True, not `{val}`."
        )


def short_platform(r=None, p=None):
    """
    Return a safe shorter version of platform.platform().

    The old default Aesara compiledir used platform.platform in
    it. This use the platform.version() as a substring. This is too
    specific as it contain the full kernel number and package
    version. This cause the compiledir to change each time there is a
    new linux kernel update. This function remove the part of platform
    that are too precise.

    If we have something else then expected, we do nothing. So this
    should be safe on other OS.

    Some example if we use platform.platform() direction. On the same
    OS, with just some kernel updates.

    compiledir_Linux-2.6.32-504.el6.x86_64-x86_64-with-redhat-6.6-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.29.2.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.23.3.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.20.3.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.17.1.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.11.2.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.23.2.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.6.2.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.6.1.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.2.1.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-279.14.1.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-279.14.1.el6.x86_64-x86_64-with-redhat-6.3-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-279.5.2.el6.x86_64-x86_64-with-redhat-6.3-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.13.1.el6.x86_64-x86_64-with-redhat-6.3-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.13.1.el6.x86_64-x86_64-with-redhat-6.2-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.7.1.el6.x86_64-x86_64-with-redhat-6.2-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.4.1.el6.x86_64-x86_64-with-redhat-6.2-Santiago-x86_64-2.6.6

    We suppose the version are ``X.Y[.*]-(digit)*(anything)*``. We keep ``X.Y``
    and don't keep less important digit in the part before ``-`` and we remove
    the leading digit after the first ``-``.

    If the information don't fit that pattern, we do not modify platform.

    """
    if r is None:
        r = platform.release()
    if p is None:
        p = platform.platform()
    sp = r.split("-")
    if len(sp) < 2:
        return p

    # For the split before the first -, we remove all learning digit:
    kernel_version = sp[0].split(".")
    if len(kernel_version) <= 2:
        # kernel version should always have at least 3 number.
        # If not, it use another semantic, so don't change it.
        return p
    sp[0] = ".".join(kernel_version[:2])

    # For the split after the first -, we remove leading non-digit value.
    rest = sp[1].split(".")
    while len(rest):
        if rest[0].isdigit():
            del rest[0]
        else:
            break
    sp[1] = ".".join(rest)

    # For sp[2:], we don't change anything.
    sr = "-".join(sp)
    p = p.replace(r, sr)

    return p


def add_basic_configvars():

    config.add(
        "floatX",
        "Default floating-point precision for python casts.\n"
        "\n"
        "Note: float16 support is experimental, use at your own risk.",
        EnumStr("float64", ["float32", "float16"]),
        # TODO: see gh-4466 for how to remove it.
        in_c_key=True,
    )

    config.add(
        "warn_float64",
        "Do an action when a tensor variable with float64 dtype is"
        " created. They can't be run on the GPU with the current(old)"
        " gpu back-end and are slow with gamer GPUs.",
        EnumStr("ignore", ["warn", "raise", "pdb"]),
        in_c_key=False,
    )

    config.add(
        "pickle_test_value",
        "Dump test values while pickling model. "
        "If True, test values will be dumped with model.",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "cast_policy",
        "Rules for implicit type casting",
        EnumStr(
            "custom",
            ["numpy+floatX"],
            # The 'numpy' policy was originally planned to provide a
            # smooth transition from numpy. It was meant to behave the
            # same as numpy+floatX, but keeping float64 when numpy
            # would. However the current implementation of some cast
            # mechanisms makes it a bit more complex to add than what
            # was expected, so it is currently not available.
            # numpy,
        ),
    )

    config.add(
        "deterministic",
        "If `more`, sometimes we will select some implementation that "
        "are more deterministic, but slower. In particular, on the GPU, "
        "we will avoid using AtomicAdd. Sometimes we will still use "
        "non-deterministic implementation, e.g. when we do not have a GPU "
        "implementation that is deterministic. Also see "
        "the dnn.conv.algo* flags to cover more cases.",
        EnumStr("default", ["more"]),
        in_c_key=False,
    )

    config.add(
        "device",
        (
            "Default device for computations. If cuda* or opencl*, change the"
            "default to try to move computation to the GPU. Do not use upper case"
            "letters, only lower case even if NVIDIA uses capital letters. "
            "'gpu' means let the driver select the gpu (needed for gpu in exclusive mode). "
            "'gpuX' mean use the gpu number X."
        ),
        DeviceParam("cpu", mutable=False),
        in_c_key=False,
    )

    config.add(
        "init_gpu_device",
        (
            "Initialize the gpu device to use, works only if device=cpu. "
            "Unlike 'device', setting this option will NOT move computations, "
            "nor shared variables, to the specified GPU. "
            "It can be used to run GPU-specific tests on a particular GPU."
        ),
        DeviceParam("", mutable=False),
        in_c_key=False,
    )

    config.add(
        "force_device",
        "Raise an error if we can't use the specified device",
        BoolParam(False, mutable=False),
        in_c_key=False,
    )

    config.add(
        "conv__assert_shape",
        "If True, AbstractConv* ops will verify that user-provided"
        " shapes match the runtime shapes (debugging option,"
        " may slow down compilation)",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "print_global_stats",
        "Print some global statistics (time spent) at the end",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "contexts",
        """
        Context map for multi-gpu operation. Format is a
        semicolon-separated list of names and device names in the
        'name->dev_name' format. An example that would map name 'test' to
        device 'cuda0' and name 'test2' to device 'opencl0:0' follows:
        "test->cuda0;test2->opencl0:0".

        Invalid context names are 'cpu', 'cuda*' and 'opencl*'
        """,
        ContextsParam(),
        in_c_key=False,
    )

    config.add(
        "print_active_device",
        "Print active device at when the GPU device is initialized.",
        BoolParam(True, mutable=False),
        in_c_key=False,
    )

    config.add(
        "gpuarray__preallocate",
        """If negative it disables the allocation cache. If
                 between 0 and 1 it enables the allocation cache and
                 preallocates that fraction of the total GPU memory.  If 1
                 or greater it will preallocate that amount of memory (in
                 megabytes).""",
        FloatParam(0, mutable=False),
        in_c_key=False,
    )

    config.add(
        "gpuarray__sched",
        """The sched parameter passed for context creation to pygpu.
                    With CUDA, using "multi" is equivalent to using the parameter
                    cudaDeviceScheduleBlockingSync. This is useful to lower the
                    CPU overhead when waiting for GPU. One user found that it
                    speeds up his other processes that was doing data augmentation.
                 """,
        EnumStr("default", ["multi", "single"]),
    )

    config.add(
        "gpuarray__single_stream",
        """
                 If your computations are mostly lots of small elements,
                 using single-stream will avoid the synchronization
                 overhead and usually be faster.  For larger elements it
                 does not make a difference yet.  In the future when true
                 multi-stream is enabled in libgpuarray, this may change.
                 If you want to make sure to have optimal performance,
                 check both options.
                 """,
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "cuda__root",
        "Location of the cuda installation",
        StrParam(get_cuda_root),
        in_c_key=False,
    )

    config.add(
        "cuda__include_path",
        "Location of the cuda includes",
        StrParam(default_cuda_include),
        in_c_key=False,
    )

    # This flag determines whether or not to raise error/warning message if
    # there is a CPU Op in the computational graph.
    config.add(
        "assert_no_cpu_op",
        "Raise an error/warning if there is a CPU op in the computational graph.",
        EnumStr("ignore", ["warn", "raise", "pdb"], mutable=True),
        in_c_key=False,
    )
    config.add(
        "unpickle_function",
        (
            "Replace unpickled Aesara functions with None. "
            "This is useful to unpickle old graphs that pickled"
            " them when it shouldn't"
        ),
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "reoptimize_unpickled_function",
        "Re-optimize the graph when an Aesara function is unpickled from the disk.",
        BoolParam(False, mutable=True),
        in_c_key=False,
    )


def add_dnn_configvars():
    config.add(
        "dnn__conv__algo_fwd",
        "Default implementation to use for cuDNN forward convolution.",
        EnumStr("small", SUPPORTED_DNN_CONV_ALGO_FWD),
        in_c_key=False,
    )

    config.add(
        "dnn__conv__algo_bwd_data",
        "Default implementation to use for cuDNN backward convolution to "
        "get the gradients of the convolution with regard to the inputs.",
        EnumStr("none", SUPPORTED_DNN_CONV_ALGO_BWD_DATA),
        in_c_key=False,
    )

    config.add(
        "dnn__conv__algo_bwd_filter",
        "Default implementation to use for cuDNN backward convolution to "
        "get the gradients of the convolution with regard to the "
        "filters.",
        EnumStr("none", SUPPORTED_DNN_CONV_ALGO_BWD_FILTER),
        in_c_key=False,
    )

    config.add(
        "dnn__conv__precision",
        "Default data precision to use for the computation in cuDNN "
        "convolutions (defaults to the same dtype as the inputs of the "
        "convolutions, or float32 if inputs are float16).",
        EnumStr("as_input_f32", SUPPORTED_DNN_CONV_PRECISION),
        in_c_key=False,
    )

    config.add(
        "dnn__base_path",
        "Install location of cuDNN.",
        StrParam(default_dnn_base_path),
        in_c_key=False,
    )

    config.add(
        "dnn__include_path",
        "Location of the cudnn header",
        StrParam(default_dnn_inc_path),
        in_c_key=False,
    )

    config.add(
        "dnn__library_path",
        "Location of the cudnn link library.",
        StrParam(default_dnn_lib_path),
        in_c_key=False,
    )

    config.add(
        "dnn__bin_path",
        "Location of the cuDNN load library "
        "(on non-windows platforms, "
        "this is the same as dnn__library_path)",
        StrParam(default_dnn_bin_path),
        in_c_key=False,
    )

    config.add(
        "dnn__enabled",
        "'auto', use cuDNN if available, but silently fall back"
        " to not using it if not present."
        " If True and cuDNN can not be used, raise an error."
        " If False, disable cudnn even if present."
        " If no_check, assume present and the version between header and library match (so less compilation at context init)",
        EnumStr("auto", ["True", "False", "no_check"]),
        in_c_key=False,
    )


def add_magma_configvars():
    config.add(
        "magma__include_path",
        "Location of the magma header",
        StrParam(""),
        in_c_key=False,
    )

    config.add(
        "magma__library_path",
        "Location of the magma library",
        StrParam(""),
        in_c_key=False,
    )

    config.add(
        "magma__enabled",
        " If True, use magma for matrix computation." " If False, disable magma",
        BoolParam(False),
        in_c_key=False,
    )


def _is_gt_0(x):
    return x > 0


def _is_greater_or_equal_0(x):
    return x >= 0


def add_compile_configvars():

    config.add(
        "mode",
        "Default compilation mode",
        ConfigParam("Mode", apply=_filter_mode),
        in_c_key=False,
    )

    param = "g++"

    # Test whether or not g++ is present: disable C code if it is not.
    try:
        rc = call_subprocess_Popen(["g++", "-v"])
    except OSError:
        rc = 1

    # Anaconda on Windows has mingw-w64 packages including GCC, but it may not be on PATH.
    if rc != 0:
        if sys.platform == "win32":
            mingw_w64_gcc = os.path.join(
                os.path.dirname(sys.executable), "Library", "mingw-w64", "bin", "g++"
            )
            try:
                rc = call_subprocess_Popen([mingw_w64_gcc, "-v"])
                if rc == 0:
                    maybe_add_to_os_environ_pathlist(
                        "PATH", os.path.dirname(mingw_w64_gcc)
                    )
            except OSError:
                rc = 1
            if rc != 0:
                _logger.warning(
                    "g++ not available, if using conda: `conda install m2w64-toolchain`"
                )

    if rc != 0:
        param = ""

    # On Mac/FreeBSD we test for 'clang++' and use it by default
    if sys.platform == "darwin" or sys.platform.startswith("freebsd"):
        try:
            rc = call_subprocess_Popen(["clang++", "-v"])
            if rc == 0:
                param = "clang++"
        except OSError:
            pass

    # Try to find the full compiler path from the name
    if param != "":
        newp = distutils.spawn.find_executable(param)
        if newp is not None:
            param = newp
        del newp

    # to support path that includes spaces, we need to wrap it with double quotes on Windows
    if param and os.name == "nt":
        param = f'"{param}"'

    config.add(
        "cxx",
        "The C++ compiler to use. Currently only g++ is"
        " supported, but supporting additional compilers should not be "
        "too difficult. "
        "If it is empty, no C++ code is compiled.",
        StrParam(param, validate=_warn_cxx),
        in_c_key=False,
    )
    del param

    if rc == 0 and config.cxx != "":
        # Keep the default linker the same as the one for the mode FAST_RUN
        config.add(
            "linker",
            "Default linker used if the aesara flags mode is Mode",
            EnumStr(
                "cvm", ["c|py", "py", "c", "c|py_nogc", "vm", "vm_nogc", "cvm_nogc"]
            ),
            in_c_key=False,
        )
    else:
        # g++ is not present or the user disabled it,
        # linker should default to python only.
        config.add(
            "linker",
            "Default linker used if the aesara flags mode is Mode",
            EnumStr("vm", ["py", "vm_nogc"]),
            in_c_key=False,
        )
        if type(config).cxx.is_default:
            # If the user provided an empty value for cxx, do not warn.
            _logger.warning(
                "g++ not detected ! Aesara will be unable to execute "
                "optimized C-implementations (for both CPU and GPU) and will "
                "default to Python implementations. Performance will be severely "
                "degraded. To remove this warning, set Aesara flags cxx to an "
                "empty string."
            )

    # Keep the default value the same as the one for the mode FAST_RUN
    config.add(
        "allow_gc",
        "Do we default to delete intermediate results during Aesara"
        " function calls? Doing so lowers the memory requirement, but"
        " asks that we reallocate memory at the next function call."
        " This is implemented for the default linker, but may not work"
        " for all linkers.",
        BoolParam(True),
        in_c_key=False,
    )

    # Keep the default optimizer the same as the one for the mode FAST_RUN
    config.add(
        "optimizer",
        "Default optimizer. If not None, will use this optimizer with the Mode",
        EnumStr(
            "o4",
            ["o3", "o2", "o1", "unsafe", "fast_run", "fast_compile", "merge", "None"],
        ),
        in_c_key=False,
    )

    config.add(
        "optimizer_verbose",
        "If True, we print all optimization being applied",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "on_opt_error",
        (
            "What to do when an optimization crashes: warn and skip it, raise "
            "the exception, or fall into the pdb debugger."
        ),
        EnumStr("warn", ["raise", "pdb", "ignore"]),
        in_c_key=False,
    )

    config.add(
        "nocleanup",
        "Suppress the deletion of code files that did not compile cleanly",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "on_unused_input",
        "What to do if a variable in the 'inputs' list of "
        " aesara.function() is not used in the graph.",
        EnumStr("raise", ["warn", "ignore"]),
        in_c_key=False,
    )

    config.add(
        "gcc__cxxflags",
        "Extra compiler flags for gcc",
        StrParam(""),
        # Added elsewhere in the c key only when needed.
        in_c_key=False,
    )

    config.add(
        "cmodule__warn_no_version",
        "If True, will print a warning when compiling one or more Op "
        "with C code that can't be cached because there is no "
        "c_code_cache_version() function associated to at least one of "
        "those Ops.",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "cmodule__remove_gxx_opt",
        "If True, will remove the -O* parameter passed to g++."
        "This is useful to debug in gdb modules compiled by Aesara."
        "The parameter -g is passed by default to g++",
        BoolParam(False),
        # TODO: change so that this isn't needed.
        # This can be done by handing this in compile_args()
        in_c_key=True,
    )

    config.add(
        "cmodule__compilation_warning",
        "If True, will print compilation warnings.",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "cmodule__preload_cache",
        "If set to True, will preload the C module cache at import time",
        BoolParam(False, mutable=False),
        in_c_key=False,
    )

    config.add(
        "cmodule__age_thresh_use",
        "In seconds. The time after which " "Aesara won't reuse a compile c module.",
        # 24 days
        IntParam(60 * 60 * 24 * 24, mutable=False),
        in_c_key=False,
    )

    config.add(
        "cmodule__debug",
        "If True, define a DEBUG macro (if not exists) for any compiled C code.",
        BoolParam(False),
        in_c_key=True,
    )

    config.add(
        "compile__wait",
        """Time to wait before retrying to acquire the compile lock.""",
        IntParam(5, validate=_is_gt_0, mutable=False),
        in_c_key=False,
    )

    config.add(
        "compile__timeout",
        """In seconds, time that a process will wait before deciding to
    override an existing lock. An override only happens when the existing
    lock is held by the same owner *and* has not been 'refreshed' by this
    owner for more than this period. Refreshes are done every half timeout
    period for running processes.""",
        IntParam(_timeout_default, validate=_is_greater_or_equal_0, mutable=False),
        in_c_key=False,
    )

    config.add(
        "ctc__root",
        "Directory which contains the root of Baidu CTC library. It is assumed \
        that the compiled library is either inside the build, lib or lib64 \
        subdirectory, and the header inside the include directory.",
        StrParam("", mutable=False),
        in_c_key=False,
    )


def _is_valid_cmp_sloppy(v):
    return v in (0, 1, 2)


def add_tensor_configvars():

    # This flag is used when we import Aesara to initialize global variables.
    # So changing it after import will not modify these global variables.
    # This could be done differently... but for now we simply prevent it from being
    # changed at runtime.
    config.add(
        "tensor__cmp_sloppy",
        "Relax aesara.tensor.math._allclose (0) not at all, (1) a bit, (2) more",
        IntParam(0, _is_valid_cmp_sloppy, mutable=False),
        in_c_key=False,
    )

    config.add(
        "tensor__local_elemwise_fusion",
        (
            "Enable or not in fast_run mode(fast_run optimization) the elemwise "
            "fusion optimization"
        ),
        BoolParam(True),
        in_c_key=False,
    )

    # http://developer.amd.com/CPU/LIBRARIES/LIBM/Pages/default.aspx
    config.add(
        "lib__amblibm",
        "Use amd's amdlibm numerical library",
        BoolParam(False),
        # Added elsewhere in the c key only when needed.
        in_c_key=False,
    )

    config.add(
        "tensor__insert_inplace_optimizer_validate_nb",
        "-1: auto, if graph have less then 500 nodes 1, else 10",
        IntParam(-1),
        in_c_key=False,
    )


def add_traceback_configvars():
    config.add(
        "traceback__limit",
        "The number of stack to trace. -1 mean all.",
        # We default to a number to be able to know where v1 + v2 is created in the
        # user script. The bigger this number is, the more run time it takes.
        # We need to default to 8 to support aesara.tensor.type.tensor(...).
        # import aesara, numpy
        # X = aesara.tensor.matrix()
        # y = X.reshape((5,3,1))
        # assert y.tag.trace
        IntParam(8),
        in_c_key=False,
    )

    config.add(
        "traceback__compile_limit",
        "The number of stack to trace to keep during compilation. -1 mean all."
        " If greater then 0, will also make us save Aesara internal stack trace.",
        IntParam(0),
        in_c_key=False,
    )


def add_experimental_configvars():
    config.add(
        "experimental__unpickle_gpu_on_cpu",
        "Allow unpickling of pickled GpuArrays as numpy.ndarrays."
        "This is useful, if you want to open a GpuArray without "
        "having cuda installed."
        "If you have cuda installed, this will force unpickling to"
        "be done on the cpu to numpy.ndarray."
        "Please be aware that this may get you access to the data,"
        "however, trying to unpicke gpu functions will not succeed."
        "This flag is experimental and may be removed any time, when"
        "gpu<>cpu transparency is solved.",
        BoolParam(default=False),
        in_c_key=False,
    )

    config.add(
        "experimental__local_alloc_elemwise",
        "DEPRECATED: If True, enable the experimental"
        " optimization local_alloc_elemwise."
        " Generates error if not True. Use"
        " optimizer_excluding=local_alloc_elemwise"
        " to disable.",
        BoolParam(True),
        in_c_key=False,
    )

    # False could make the graph faster but not as safe.
    config.add(
        "experimental__local_alloc_elemwise_assert",
        "When the local_alloc_elemwise is applied, add"
        " an assert to highlight shape errors.",
        BoolParam(True),
        in_c_key=False,
    )


def add_error_and_warning_configvars():

    ###
    # To disable some warning about old bug that are fixed now.
    ###
    config.add(
        "warn__ignore_bug_before",
        (
            "If 'None', we warn about all Aesara bugs found by default. "
            "If 'all', we don't warn about Aesara bugs found by default. "
            "If a version, we print only the warnings relative to Aesara "
            "bugs found after that version. "
            "Warning for specific bugs can be configured with specific "
            "[warn] flags."
        ),
        EnumStr(
            "0.9",
            [
                "None",
                "all",
                "0.3",
                "0.4",
                "0.4.1",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.8.1",
                "0.8.2",
                "0.9",
                "0.10",
                "1.0",
                "1.0.1",
                "1.0.2",
                "1.0.3",
                "1.0.4",
                "1.0.5",
            ],
            mutable=False,
        ),
        in_c_key=False,
    )

    # Note to developers:
    # Generally your exceptions should use an apply node's __str__
    # method when exception_verbosity == 'low'. When exception_verbosity
    # == 'high', you should include a call to printing.min_informative_str
    # on all important apply nodes.
    config.add(
        "exception_verbosity",
        "If 'low', the text of exceptions will generally refer "
        "to apply nodes with short names such as "
        "Elemwise{add_no_inplace}. If 'high', some exceptions "
        "will also refer to apply nodes with long descriptions "
        """ like:
        A. Elemwise{add_no_inplace}
                B. log_likelihood_v_given_h
                C. log_likelihood_h""",
        EnumStr("low", ["high"]),
        in_c_key=False,
    )


def _has_cxx():
    return bool(config.cxx)


def _is_valid_check_strides(v):
    return v in (0, 1, 2)


def add_testvalue_and_checking_configvars():
    config.add(
        "print_test_value",
        (
            "If 'True', the __eval__ of an Aesara variable will return its test_value "
            "when this is available. This has the practical conseguence that, e.g., "
            "in debugging `my_var` will print the same as `my_var.tag.test_value` "
            "when a test value is defined."
        ),
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "compute_test_value",
        (
            "If 'True', Aesara will run each op at graph build time, using "
            "Constants, SharedVariables and the tag 'test_value' as inputs "
            "to the function. This helps the user track down problems in the "
            "graph before it gets optimized."
        ),
        EnumStr("off", ["ignore", "warn", "raise", "pdb"]),
        in_c_key=False,
    )

    config.add(
        "compute_test_value_opt",
        (
            "For debugging Aesara optimization only."
            " Same as compute_test_value, but is used"
            " during Aesara optimization"
        ),
        EnumStr("off", ["ignore", "warn", "raise", "pdb"]),
        in_c_key=False,
    )
    config.add(
        "check_input",
        "Specify if types should check their input in their C code. "
        "It can be used to speed up compilation, reduce overhead "
        "(particularly for scalars) and reduce the number of generated C "
        "files.",
        BoolParam(True),
        in_c_key=True,
    )
    config.add(
        "NanGuardMode__nan_is_error",
        "Default value for nan_is_error",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "NanGuardMode__inf_is_error",
        "Default value for inf_is_error",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "NanGuardMode__big_is_error",
        "Default value for big_is_error",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "NanGuardMode__action",
        "What NanGuardMode does when it finds a problem",
        EnumStr("raise", ["warn", "pdb"]),
        in_c_key=False,
    )

    config.add(
        "DebugMode__patience",
        "Optimize graph this many times to detect inconsistency",
        IntParam(10, _is_gt_0),
        in_c_key=False,
    )

    config.add(
        "DebugMode__check_c",
        "Run C implementations where possible",
        BoolParam(_has_cxx),
        in_c_key=False,
    )

    config.add(
        "DebugMode__check_py",
        "Run Python implementations where possible",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "DebugMode__check_finite",
        "True -> complain about NaN/Inf results",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "DebugMode__check_strides",
        (
            "Check that Python- and C-produced ndarrays have same strides. "
            "On difference: (0) - ignore, (1) warn, or (2) raise error"
        ),
        # TODO: make this an Enum setting
        IntParam(0, _is_valid_check_strides),
        in_c_key=False,
    )

    config.add(
        "DebugMode__warn_input_not_reused",
        (
            "Generate a warning when destroy_map or view_map says that an "
            "op works inplace, but the op did not reuse the input for its "
            "output."
        ),
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "DebugMode__check_preallocated_output",
        (
            "Test thunks with pre-allocated memory as output storage. "
            'This is a list of strings separated by ":". Valid values are: '
            '"initial" (initial storage in storage map, happens with Scan),'
            '"previous" (previously-returned memory), '
            '"c_contiguous", "f_contiguous", '
            '"strided" (positive and negative strides), '
            '"wrong_size" (larger and smaller dimensions), and '
            '"ALL" (all of the above).'
        ),
        StrParam("", validate=_is_valid_check_preallocated_output_param),
        in_c_key=False,
    )

    config.add(
        "DebugMode__check_preallocated_output_ndim",
        (
            'When testing with "strided" preallocated output memory, '
            "test all combinations of strides over that number of "
            "(inner-most) dimensions. You may want to reduce that number "
            "to reduce memory or time usage, but it is advised to keep a "
            "minimum of 2."
        ),
        IntParam(4, _is_gt_0),
        in_c_key=False,
    )

    config.add(
        "profiling__time_thunks",
        """Time individual thunks when profiling""",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "profiling__n_apply",
        "Number of Apply instances to print by default",
        IntParam(20, _is_gt_0),
        in_c_key=False,
    )

    config.add(
        "profiling__n_ops",
        "Number of Ops to print by default",
        IntParam(20, _is_gt_0),
        in_c_key=False,
    )

    config.add(
        "profiling__output_line_width",
        "Max line width for the profiling output",
        IntParam(512, _is_gt_0),
        in_c_key=False,
    )

    config.add(
        "profiling__min_memory_size",
        """For the memory profile, do not print Apply nodes if the size
                 of their outputs (in bytes) is lower than this threshold""",
        IntParam(1024, _is_greater_or_equal_0),
        in_c_key=False,
    )

    config.add(
        "profiling__min_peak_memory",
        """The min peak memory usage of the order""",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "profiling__destination",
        """File destination of the profiling output""",
        StrParam("stderr"),
        in_c_key=False,
    )

    config.add(
        "profiling__debugprint",
        """Do a debugprint of the profiled functions""",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "profiling__ignore_first_call",
        """Do we ignore the first call of an Aesara function.""",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "on_shape_error",
        "warn: print a warning and use the default" " value. raise: raise an error",
        EnumStr("warn", ["raise"]),
        in_c_key=False,
    )


def add_multiprocessing_configvars():
    # Test if the env variable is set
    var = os.getenv("OMP_NUM_THREADS", None)
    if var:
        try:
            int(var)
        except ValueError:
            raise TypeError(
                f"The environment variable OMP_NUM_THREADS should be a number, got '{var}'."
            )
        else:
            default_openmp = not int(var) == 1
    else:
        # Check the number of cores availables.
        count = os.cpu_count()
        if count is None:
            _logger.warning(
                "We are not able to detect the number of CPU cores."
                " We disable openmp by default. To remove this"
                " warning, set the environment variable"
                " OMP_NUM_THREADS to the number of threads you"
                " want aesara to use."
            )
        default_openmp = count > 1

    # Disable it by default for now as currently only the ConvOp supports
    # it, and this causes slowdown by default as we do not disable it for
    # too small convolution.
    default_openmp = False

    config.add(
        "openmp",
        "Allow (or not) parallel computation on the CPU with OpenMP. "
        "This is the default value used when creating an Op that "
        "supports OpenMP parallelization. It is preferable to define it "
        "via the Aesara configuration file ~/.aesararc or with the "
        "environment variable AESARA_FLAGS. Parallelization is only "
        "done for some operations that implement it, and even for "
        "operations that implement parallelism, each operation is free "
        "to respect this flag or not. You can control the number of "
        "threads used with the environment variable OMP_NUM_THREADS."
        " If it is set to 1, we disable openmp in Aesara by default.",
        BoolParam(default_openmp),
        in_c_key=False,
    )

    config.add(
        "openmp_elemwise_minsize",
        "If OpenMP is enabled, this is the minimum size of vectors "
        "for which the openmp parallelization is enabled "
        "in element wise ops.",
        IntParam(200000),
        in_c_key=False,
    )


def add_optimizer_configvars():
    config.add(
        "optimizer_excluding",
        (
            "When using the default mode, we will remove optimizer with "
            "these tags. Separate tags with ':'."
        ),
        StrParam("", mutable=False),
        in_c_key=False,
    )

    config.add(
        "optimizer_including",
        (
            "When using the default mode, we will add optimizer with "
            "these tags. Separate tags with ':'."
        ),
        StrParam("", mutable=False),
        in_c_key=False,
    )

    config.add(
        "optimizer_requiring",
        (
            "When using the default mode, we will require optimizer with "
            "these tags. Separate tags with ':'."
        ),
        StrParam("", mutable=False),
        in_c_key=False,
    )

    config.add(
        "optdb__position_cutoff",
        "Where to stop eariler during optimization. It represent the"
        " position of the optimizer where to stop.",
        FloatParam(np.inf),
        in_c_key=False,
    )

    config.add(
        "optdb__max_use_ratio",
        "A ratio that prevent infinite loop in EquilibriumOptimizer.",
        FloatParam(8),
        in_c_key=False,
    )
    config.add(
        "cycle_detection",
        "If cycle_detection is set to regular, most inplaces are allowed,"
        "but it is slower. If cycle_detection is set to faster, less inplaces"
        "are allowed, but it makes the compilation faster."
        "The interaction of which one give the lower peak memory usage is"
        "complicated and not predictable, so if you are close to the peak"
        "memory usage, triyng both could give you a small gain.",
        EnumStr("regular", ["fast"]),
        in_c_key=False,
    )

    config.add(
        "check_stack_trace",
        "A flag for checking the stack trace during the optimization process. "
        "default (off): does not check the stack trace of any optimization "
        "log: inserts a dummy stack trace that identifies the optimization"
        "that inserted the variable that had an empty stack trace."
        "warn: prints a warning if a stack trace is missing and also a dummy"
        "stack trace is inserted that indicates which optimization inserted"
        "the variable that had an empty stack trace."
        "raise: raises an exception if a stack trace is missing",
        EnumStr("off", ["log", "warn", "raise"]),
        in_c_key=False,
    )


def add_metaopt_configvars():
    config.add(
        "metaopt__verbose",
        "0 for silent, 1 for only warnings, 2 for full output with"
        "timings and selected implementation",
        IntParam(0),
        in_c_key=False,
    )

    config.add(
        "metaopt__optimizer_excluding",
        ("exclude optimizers with these tags. " "Separate tags with ':'."),
        StrParam(""),
        in_c_key=False,
    )

    config.add(
        "metaopt__optimizer_including",
        ("include optimizers with these tags. " "Separate tags with ':'."),
        StrParam(""),
        in_c_key=False,
    )


def add_vm_configvars():
    config.add(
        "profile",
        "If VM should collect profile information",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "profile_optimizer",
        "If VM should collect optimizer profile information",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "profile_memory",
        "If VM should collect memory profile information and print it",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "vm__lazy",
        "Useful only for the vm linkers. When lazy is None,"
        " auto detect if lazy evaluation is needed and use the appropriate"
        " version. If lazy is True/False, force the version used between"
        " Loop/LoopGC and Stack.",
        ConfigParam("None", apply=_filter_vm_lazy),
        in_c_key=False,
    )


def add_deprecated_configvars():
    # TODO: remove this?
    config.add(
        "cache_optimizations",
        "WARNING: work in progress, does not work yet. "
        "Specify if the optimization cache should be used. This cache will "
        "any optimized graph and its optimization. Actually slow downs a lot "
        "the first optimization, and could possibly still contains some bugs. "
        "Use at your own risks.",
        BoolParam(False),
        in_c_key=False,
    )

    # TODO: remove this?
    config.add(
        "unittests__rseed",
        "Seed to use for randomized unit tests. "
        "Special value 'random' means using a seed of None.",
        StrParam(666, validate=_good_seem_param),
        in_c_key=False,
    )

    # TODO: remove?
    config.add(
        "warn__identify_1pexp_bug",
        "Warn if Aesara versions prior to 7987b51 (2011-12-18) could have "
        "yielded a wrong result due to a bug in the is_1pexp function",
        BoolParam(_warn_default("0.4.1")),
        in_c_key=False,
    )
    # TODO: this setting is not used anywhere
    config.add(
        "gpu__local_elemwise_fusion",
        (
            "Enable or not in fast_run mode(fast_run optimization) the gpu "
            "elemwise fusion optimization"
        ),
        BoolParam(True),
        in_c_key=False,
    )
    # TODO: this setting is not used anywhere
    config.add(
        "gpuelemwise__sync",
        "when true, wait that the gpu fct finished and check it error code.",
        BoolParam(True),
        in_c_key=False,
    )
    # TODO: most of these bugfix-related warnings can probably be removed
    config.add(
        "warn__argmax_pushdown_bug",
        (
            "Warn if in past version of Aesara we generated a bug with the "
            "aesara.tensor.nnet.basic.local_argmax_pushdown optimization. "
            "Was fixed 27 may 2010"
        ),
        BoolParam(_warn_default("0.3")),
        in_c_key=False,
    )

    config.add(
        "warn__gpusum_01_011_0111_bug",
        (
            "Warn if we are in a case where old version of Aesara had a "
            "silent bug with GpuSum pattern 01,011 and 0111 when the first "
            "dimensions was bigger then 4096. Was fixed 31 may 2010"
        ),
        BoolParam(_warn_default("0.3")),
        in_c_key=False,
    )

    config.add(
        "warn__sum_sum_bug",
        (
            "Warn if we are in a case where Aesara version between version "
            "9923a40c7b7a and the 2 august 2010 (fixed date), generated an "
            "error in that case. This happens when there are 2 consecutive "
            "sums in the graph, bad code was generated. "
            "Was fixed 2 August 2010"
        ),
        BoolParam(_warn_default("0.3")),
        in_c_key=False,
    )

    config.add(
        "warn__sum_div_dimshuffle_bug",
        (
            "Warn if previous versions of Aesara (between rev. "
            "3bd9b789f5e8, 2010-06-16, and cfc6322e5ad4, 2010-08-03) "
            "would have given incorrect result. This bug was triggered by "
            "sum of division of dimshuffled tensors."
        ),
        BoolParam(_warn_default("0.3")),
        in_c_key=False,
    )

    config.add(
        "warn__subtensor_merge_bug",
        "Warn if previous versions of Aesara (before 0.5rc2) could have given "
        "incorrect results when indexing into a subtensor with negative "
        "stride (for instance, for instance, x[a:b:-1][c]).",
        BoolParam(_warn_default("0.5")),
        in_c_key=False,
    )

    config.add(
        "warn__gpu_set_subtensor1",
        "Warn if previous versions of Aesara (before 0.6) could have given "
        "incorrect results when moving to the gpu "
        "set_subtensor(x[int vector], new_value)",
        BoolParam(_warn_default("0.6")),
        in_c_key=False,
    )

    config.add(
        "warn__vm_gc_bug",
        "There was a bug that existed in the default Aesara configuration,"
        " only in the development version between July 5th 2012"
        " and July 30th 2012. This was not in a released version."
        " If your code was affected by this bug, a warning"
        " will be printed during the code execution if you use the"
        " `linker=vm,vm__lazy=True,warn__vm_gc_bug=True` Aesara flags."
        " This warning is disabled by default as the bug was not released.",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "warn__signal_conv2d_interface",
        (
            "Warn we use the new signal.conv2d() when its interface"
            " changed mid June 2014"
        ),
        BoolParam(_warn_default("0.7")),
        in_c_key=False,
    )

    config.add(
        "warn__reduce_join",
        (
            "Your current code is fine, but Aesara versions "
            "prior to 0.7 (or this development version) "
            "might have given an incorrect result. "
            "To disable this warning, set the Aesara flag "
            "warn__reduce_join to False. The problem was an "
            "optimization, that modified the pattern "
            '"Reduce{scalar.op}(Join(axis=0, a, b), axis=0)", '
            "did not check the reduction axis. So if the "
            "reduction axis was not 0, you got a wrong answer."
        ),
        BoolParam(_warn_default("0.7")),
        in_c_key=False,
    )

    config.add(
        "warn__inc_set_subtensor1",
        (
            "Warn if previous versions of Aesara (before 0.7) could have "
            "given incorrect results for inc_subtensor and set_subtensor "
            "when using some patterns of advanced indexing (indexing with "
            "one vector or matrix of ints)."
        ),
        BoolParam(_warn_default("0.7")),
        in_c_key=False,
    )

    config.add(
        "warn__round",
        "Warn when using `tensor.round` with the default mode. "
        "Round changed its default from `half_away_from_zero` to "
        "`half_to_even` to have the same default as NumPy.",
        BoolParam(_warn_default("0.9")),
        in_c_key=False,
    )

    config.add(
        "warn__inc_subtensor1_opt",
        "Warn if previous versions of Aesara (before 0.10) could have "
        "given incorrect results when computing "
        "inc_subtensor(zeros[idx], x)[idx], when idx is an array of integers "
        "with duplicated values.",
        BoolParam(_warn_default("0.10")),
        in_c_key=False,
    )


def add_scan_configvars():
    config.add(
        "scan__allow_gc",
        "Allow/disallow gc inside of Scan (default: False)",
        BoolParam(False),
        in_c_key=False,
    )

    config.add(
        "scan__allow_output_prealloc",
        "Allow/disallow memory preallocation for outputs inside of scan "
        "(default: True)",
        BoolParam(True),
        in_c_key=False,
    )

    config.add(
        "scan__debug",
        "If True, enable extra verbose output related to scan",
        BoolParam(False),
        in_c_key=False,
    )


def add_numba_configvars():
    config.add(
        "numba__vectorize_target",
        ("Default target for numba.vectorize."),
        EnumStr("cpu", ["parallel", "cuda"], mutable=True),
        in_c_key=False,
    )
    config.add(
        "numba__fastmath",
        ("If True, use Numba's fastmath mode."),
        BoolParam(True),
        in_c_key=False,
    )


def _get_default_gpuarray__cache_path():
    return os.path.join(config.compiledir, "gpuarray_kernels")


def _default_compiledirname():
    formatted = config.compiledir_format % _compiledir_format_dict
    safe = re.sub(r"[\(\)\s,]+", "_", formatted)
    return safe


def _filter_base_compiledir(path):
    # Expand '~' in path
    return os.path.expanduser(str(path))


def _filter_compiledir(path):
    # Expand '~' in path
    path = os.path.expanduser(path)
    # Turn path into the 'real' path. This ensures that:
    #   1. There is no relative path, which would fail e.g. when trying to
    #      import modules from the compile dir.
    #   2. The path is stable w.r.t. e.g. symlinks (which makes it easier
    #      to re-use compiled modules).
    path = os.path.realpath(path)
    if os.access(path, os.F_OK):  # Do it exist?
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            # If it exist we need read, write and listing access
            raise ValueError(
                f"compiledir '{path}' exists but you don't have read, write"
                " or listing permissions."
            )
    else:
        try:
            os.makedirs(path, 0o770)  # read-write-execute for user and group
        except OSError as e:
            # Maybe another parallel execution of aesara was trying to create
            # the same directory at the same time.
            if e.errno != errno.EEXIST:
                raise ValueError(
                    "Unable to create the compiledir directory"
                    f" '{path}'. Check the permissions."
                )

    # PROBLEM: sometimes the initial approach based on
    # os.system('touch') returned -1 for an unknown reason; the
    # alternate approach here worked in all cases... it was weird.
    # No error should happen as we checked the permissions.
    init_file = os.path.join(path, "__init__.py")
    if not os.path.exists(init_file):
        try:
            open(init_file, "w").close()
        except OSError as e:
            if os.path.exists(init_file):
                pass  # has already been created
            else:
                e.args += (f"{path} exist? {os.path.exists(path)}",)
                raise
    return path


def _get_home_dir():
    """
    Return location of the user's home directory.

    """
    home = os.getenv("HOME")
    if home is None:
        # This expanduser usually works on Windows (see discussion on
        # theano-users, July 13 2010).
        home = os.path.expanduser("~")
        if home == "~":
            # This might happen when expanduser fails. Although the cause of
            # failure is a mystery, it has been seen on some Windows system.
            home = os.getenv("USERPROFILE")
    assert home is not None
    return home


_compiledir_format_dict = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "python_bitwidth": LOCAL_BITWIDTH,
    "python_int_bitwidth": PYTHON_INT_BITWIDTH,
    "aesara_version": aesara.__version__,
    "numpy_version": np.__version__,
    "gxx_version": "xxx",
    "hostname": socket.gethostname(),
}


def _default_compiledir():
    return os.path.join(config.base_compiledir, _default_compiledirname())


def add_caching_dir_configvars():
    _compiledir_format_dict["gxx_version"] = (gcc_version_str.replace(" ", "_"),)
    _compiledir_format_dict["short_platform"] = short_platform()
    # Allow to have easily one compiledir per device.
    _compiledir_format_dict["device"] = config.device
    compiledir_format_keys = ", ".join(sorted(_compiledir_format_dict.keys()))
    _default_compiledir_format = (
        "compiledir_%(short_platform)s-%(processor)s-"
        "%(python_version)s-%(python_bitwidth)s"
    )

    config.add(
        "compiledir_format",
        textwrap.fill(
            textwrap.dedent(
                f"""\
                     Format string for platform-dependent compiled
                     module subdirectory (relative to base_compiledir).
                     Available keys: {compiledir_format_keys}. Defaults to {_default_compiledir_format}.
                 """
            )
        ),
        StrParam(_default_compiledir_format, mutable=False),
        in_c_key=False,
    )

    # On Windows we should avoid writing temporary files to a directory that is
    # part of the roaming part of the user profile. Instead we use the local part
    # of the user profile, when available.
    if sys.platform == "win32" and os.getenv("LOCALAPPDATA") is not None:
        default_base_compiledir = os.path.join(os.getenv("LOCALAPPDATA"), "Aesara")
    else:
        default_base_compiledir = os.path.join(_get_home_dir(), ".aesara")

    config.add(
        "base_compiledir",
        "platform-independent root directory for compiled modules",
        ConfigParam(
            default_base_compiledir, apply=_filter_base_compiledir, mutable=False
        ),
        in_c_key=False,
    )

    config.add(
        "compiledir",
        "platform-dependent cache directory for compiled modules",
        ConfigParam(_default_compiledir, apply=_filter_compiledir, mutable=False),
        in_c_key=False,
    )

    config.add(
        "gpuarray__cache_path",
        "Directory to cache pre-compiled kernels for the gpuarray backend.",
        ConfigParam(
            _get_default_gpuarray__cache_path,
            apply=_filter_base_compiledir,
            mutable=False,
        ),
        in_c_key=False,
    )


# Those are the options provided by Aesara to choose algorithms at runtime.
SUPPORTED_DNN_CONV_ALGO_RUNTIME = (
    "guess_once",
    "guess_on_shape_change",
    "time_once",
    "time_on_shape_change",
)

# Those are the supported algorithm by Aesara,
# The tests will reference those lists.
SUPPORTED_DNN_CONV_ALGO_FWD = (
    "small",
    "none",
    "large",
    "fft",
    "fft_tiling",
    "winograd",
    "winograd_non_fused",
) + SUPPORTED_DNN_CONV_ALGO_RUNTIME

SUPPORTED_DNN_CONV_ALGO_BWD_DATA = (
    "none",
    "deterministic",
    "fft",
    "fft_tiling",
    "winograd",
    "winograd_non_fused",
) + SUPPORTED_DNN_CONV_ALGO_RUNTIME

SUPPORTED_DNN_CONV_ALGO_BWD_FILTER = (
    "none",
    "deterministic",
    "fft",
    "small",
    "winograd_non_fused",
    "fft_tiling",
) + SUPPORTED_DNN_CONV_ALGO_RUNTIME

SUPPORTED_DNN_CONV_PRECISION = (
    "as_input_f32",
    "as_input",
    "float16",
    "float32",
    "float64",
)

# Eventually, the instance of `AesaraConfigParser` should be created right here,
# where it is also populated with settings.  But for a transition period, it
# remains as `configparser._config`, while everybody accessing it through
# `configparser.config` is flooded with deprecation warnings. These warnings
# instruct one to use `aesara.config`, which is an alias for
# `aesara.configdefaults.config`.
config = aesara.configparser._config

# The functions below register config variables into the config instance above.
add_basic_configvars()
add_dnn_configvars()
add_magma_configvars()
add_compile_configvars()
# TODO: "tensor", "gpuarray" and compilation options are closely related.. Grouping is not great.
add_tensor_configvars()
add_traceback_configvars()
add_experimental_configvars()
add_error_and_warning_configvars()
add_testvalue_and_checking_configvars()
add_multiprocessing_configvars()
add_optimizer_configvars()
# TODO: Module-specific configs should probably be added upon import of the module.
# This would mean either calling the function from there, or even moving all the related code there.
# Blas-related config are a special pain-point, because their addition depends on a lot of stuff from
# that module, which introduces a circular dependency!
add_metaopt_configvars()
add_vm_configvars()
add_deprecated_configvars()
add_numba_configvars()

# TODO: `gcc_version_str` is used by other modules.. Should it become an immutable config var?
try:
    p_out = output_subprocess_Popen([config.cxx, "-dumpversion"])
    gcc_version_str = p_out[0].strip().decode()
except OSError:
    # Typically means gcc cannot be found.
    gcc_version_str = "GCC_NOT_FOUND"
# TODO: The caching dir resolution is a procedural mess of helper functions, local variables
# and config definitions. And the result is also not particularly pretty..
add_caching_dir_configvars()
