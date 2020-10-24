from theano.compile.builders import OpFromGraph, ops_with_inner_function
from theano.compile.debugmode import DebugMode

# This is for some backward compatibility
from theano.compile.function import types as function_module
from theano.compile.function.pfunc import Param, pfunc, rebuild_collect_shared
from theano.compile.function.types import (
    AliasedMemoryError,
    Function,
    FunctionMaker,
    Supervisor,
    UnusedInputError,
    alias_root,
    check_equal,
    convert_function_input,
    fgraph_updated_vars,
    get_info_on_inputs,
    infer_reuse_pattern,
    insert_deepcopy,
    orig_function,
    register_checker,
    std_fgraph,
    view_tree_set,
)
from theano.compile.io import In, Out, SymbolicInput, SymbolicOutput
from theano.compile.mode import (
    FAST_COMPILE,
    FAST_RUN,
    JAX,
    OPT_FAST_COMPILE,
    OPT_FAST_RUN,
    OPT_FAST_RUN_STABLE,
    OPT_MERGE,
    OPT_NONE,
    OPT_O2,
    OPT_O3,
    OPT_STABILIZE,
    OPT_UNSAFE,
    AddDestroyHandler,
    AddFeatureOptimizer,
    Mode,
    PrintCurrentFunctionGraph,
    get_default_mode,
    get_mode,
    instantiated_default_mode,
    local_useless,
    optdb,
    predefined_linkers,
    predefined_modes,
    predefined_optimizers,
    register_linker,
    register_mode,
    register_optimizer,
)
from theano.compile.monitormode import MonitorMode
from theano.compile.ops import (
    DeepCopyOp,
    FromFunctionOp,
    Rebroadcast,
    Shape,
    Shape_i,
    SpecifyShape,
    ViewOp,
    as_op,
    deep_copy_op,
    register_deep_copy_op_c_code,
    register_rebroadcast_c_code,
    register_shape_c_code,
    register_shape_i_c_code,
    register_specify_shape_c_code,
    register_view_op_c_code,
    shape,
    specify_shape,
    view_op,
)
from theano.compile.profiling import ProfileStats, ScanProfileStats
from theano.compile.sharedvalue import SharedVariable, shared, shared_constructor
