"""This module provides utility functions for the `Scan` `Op`."""

import copy
import logging
import warnings
from collections import OrderedDict, namedtuple

import numpy as np

from aesara import scalar as aes
from aesara import tensor as aet
from aesara.configdefaults import config
from aesara.graph.basic import (
    Constant,
    Variable,
    clone_replace,
    equal_computations,
    graph_inputs,
)
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.graph.opt import TopoOptimizer, local_optimizer
from aesara.graph.utils import TestValueError
from aesara.tensor.basic import AllocEmpty, get_scalar_constant_value
from aesara.tensor.subtensor import set_subtensor
from aesara.tensor.var import TensorConstant


_logger = logging.getLogger("aesara.scan.utils")


def safe_new(x, tag="", dtype=None):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name (old name + tag). This function is used
    by gradient, or the R-op to construct new variables for the inputs of
    the inner graph such that there is no interference between the original
    graph and the newly constructed graph.

    """
    if hasattr(x, "name") and x.name is not None:
        nw_name = x.name + tag
    else:
        nw_name = None

    if isinstance(x, Constant):
        if dtype and x.dtype != dtype:
            casted_x = x.astype(dtype)
            nwx = x.__class__(casted_x.type, x.data, x.name)
            nwx.tag = copy.copy(x.tag)
            return nwx
        else:
            return x.clone()
    # Note, as_tensor_variable will convert the Scalar into a
    # TensorScalar that will require a ScalarFromTensor op,
    # making the pushout optimization fail
    elif isinstance(x, aes.ScalarVariable):
        if dtype:
            nw_x = aes.get_scalar_type(dtype=dtype)()
        else:
            nw_x = x.type()
        nw_x.name = nw_name
        if config.compute_test_value != "off":
            # Copy test value, cast it if necessary
            try:
                x_test_value = get_test_value(x)
            except TestValueError:
                pass
            else:
                # This clause is executed if no exception was raised
                nw_x.tag.test_value = nw_x.type.filter(x_test_value)
        return nw_x
    else:
        try:
            x = aet.as_tensor_variable(x)
        except TypeError:
            # This could happen for example for random states
            pass

    # Cast x if needed. If x has a test value, this will also cast it.
    if dtype and x.dtype != dtype:
        x = x.astype(dtype)

    nw_x = x.type()
    nw_x.name = nw_name
    # Preserve test values so that the 'compute_test_value' option can be used.
    # The test value is deep-copied to ensure there can be no interactions
    # between test values, due to inplace operations for instance. This may
    # not be the most efficient memory-wise, though.
    if config.compute_test_value != "off":
        try:
            nw_x.tag.test_value = copy.deepcopy(get_test_value(x))
        except TestValueError:
            pass

    return nw_x


class until:
    """
    Class used to encode the different things the inner function of scan can
    (or needs) to return.

    This class has to be used when scan needs to halt when a condition is
    met, otherwise the list of outputs and dictionary can directly be return
    as a tuple. The reason is that otherwise scan has no way to distinguish
    between the condition and the list of outputs ( unless we enforce and
    order, but since this was not impose up to know it can make quite a bit
    of code to fail).

    """

    def __init__(self, condition):
        self.condition = aet.as_tensor_variable(condition)
        assert self.condition.ndim == 0


def traverse(out, x, x_copy, d, visited=None):
    """
    Function used by scan to parse the tree and figure out which nodes
    it needs to replace.

    There are two options :
        1) x and x_copy or on host, then you would replace x with x_copy
        2) x is on gpu, x_copy on host, then you need to replace
        host_from_gpu(x) with x_copy
    This happens because initially shared variables are on GPU... which is
    fine for the main computational graph but confuses things a bit for the
    inner graph of scan.

    """
    # ``visited`` is a set of nodes that are already known and don't need to be
    # checked again, speeding up the traversal of multiply-connected graphs.
    # if a ``visited`` set is given, it will be updated in-place so the callee
    # knows which nodes we have seen.
    if visited is None:
        visited = set()
    if out in visited:
        return d
    visited.add(out)
    from aesara.gpuarray import pygpu_activated
    from aesara.gpuarray.basic_ops import GpuFromHost, host_from_gpu
    from aesara.gpuarray.type import GpuArrayType

    if out == x:
        assert isinstance(x.type, GpuArrayType)
        d[out] = GpuFromHost(x.type.context_name)(x_copy)
        return d
    elif out.owner is None:
        return d
    elif pygpu_activated and out.owner.op == host_from_gpu and out.owner.inputs == [x]:
        d[out] = aet.as_tensor_variable(x_copy)
        return d
    else:
        for inp in out.owner.inputs:
            d = traverse(inp, x, x_copy, d, visited)
        return d


# Hashing a dictionary/list/tuple by xoring the hash of each element
def hash_listsDictsTuples(x):
    hash_value = 0
    if isinstance(x, dict):
        for k, v in x.items():
            hash_value ^= hash_listsDictsTuples(k)
            hash_value ^= hash_listsDictsTuples(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            hash_value ^= hash_listsDictsTuples(v)
    else:
        hash_value ^= hash(x)
    return hash_value


def map_variables(replacer, graphs, additional_inputs=None):
    """Construct new graphs based on 'graphs' with some variables replaced
    according to 'replacer'.

    :param replacer: function that takes a variable and returns its
         replacement.
    :param graphs: an iterable of graphs in which to replace variables
    :param additional_inputs: an iterable of graph inputs not used in any
         of 'graphs' but possibly used in the graphs returned by `replacer`
    :return: the new graphs, in the same order as 'graphs'

    Example:

    .. code-block:: python

        tag = "replaceme"

        a = aesara.tensor.type.scalar("a")
        b = aesara.tensor.type.scalar("b")
        c = aesara.tensor.type.scalar("c")

        ab = a + b
        ab.tag.replacement = a * b

        u = ab + c
        v, = map_variables(lambda graph:
            return getattr(graph.tag, "replacement", graph),
            [u])

        # v is now equal to a * b + c
    """
    if additional_inputs is None:
        additional_inputs = []

    # wrap replacer to avoid replacing things we just put there.
    graphs_seen = set()

    def wrapped_replacer(graph):
        if graph in graphs_seen:
            return graph
        else:
            new_graph = replacer(graph)
            graphs_seen.add(new_graph)
            return new_graph

    graphs = list(graphs)
    inputs_ = list(set(list(graph_inputs(graphs)) + list(additional_inputs)))

    # perform any desired replacement of input variables.  these
    # aren't replaced by the local optimizer approach because they are
    # not outputs of any Apply node.
    new_inputs = [wrapped_replacer(i) for i in inputs_]
    replacements = [
        (input_, new_input)
        for input_, new_input in zip(inputs_, new_inputs)
        if new_input is not input_
    ]
    graphs = clone_replace(graphs, share_inputs=True, replace=replacements)
    inputs_ = list(set(list(graph_inputs(graphs)) + list(additional_inputs)))

    fg = FunctionGraph(inputs_, graphs, clone=False)

    nodes_seen = set()

    @local_optimizer(None)
    def local_transform(fgraph, node):
        if node in nodes_seen:
            return False

        # importing Scan into module scope would be circular
        from aesara.compile.builders import OpFromGraph
        from aesara.scan.op import Scan

        if isinstance(node.op, (Scan, OpFromGraph)):
            # recurse on the inner graph
            (
                new_inner_inputs,
                new_outer_inputs,
                new_inner_outputs,
            ) = _map_variables_inner(
                wrapped_replacer,
                inner_inputs=node.op.inputs,
                outer_inputs=node.inputs,
                inner_outputs=node.op.outputs,
                containing_op=node.op,
            )
            # reinstantiate the op
            if isinstance(node.op, Scan):
                new_op = Scan(
                    new_inner_inputs,
                    new_inner_outputs,
                    node.op.info,
                    # FIXME: infer this someday?
                    typeConstructor=None,
                )
            elif isinstance(node.op, OpFromGraph):
                new_op = OpFromGraph(
                    new_inner_inputs, new_inner_outputs, **node.op.kwargs
                )
            # make a new node to replace the old one
            new_node = new_op.make_node(*new_outer_inputs)
            nodes_seen.add(new_node)
            return new_node.outputs
        else:
            nodes_seen.add(node)
            replacements = [wrapped_replacer(o) for o in node.outputs]

            # Add inputs to replacement graphs as inputs to this `fgraph`
            for i in graph_inputs(replacements):
                fgraph.add_input(i)

            return replacements

    topo_transform = TopoOptimizer(local_transform, "out_to_in")
    topo_transform.optimize(fg)

    new_graphs = fg.outputs
    fg.disown()
    return new_graphs


def _map_variables_inner(
    replacer, inner_inputs, outer_inputs, inner_outputs, containing_op
):
    # the replacements returned by the replacer may involve variables
    # that are already owned by the outer fgraph (`fg` in the caller)
    # and so cannot be added to the inner fgraph (`fg` in the
    # recursive call).  wrap the replacer to catch these before they
    # are added.

    # additionally, some of these may be fgraph inputs or shared
    # variables, which we cannot directly use inside the inner graph.
    # we need to create inner inputs to access them through.

    outer_to_inner = dict(zip(outer_inputs, inner_inputs))
    extra_inner_inputs = []
    extra_outer_inputs = []

    from itertools import chain

    from aesara.scan import utils

    def inner_replacer(graph):
        new_graph = replacer(graph)

        other_inputs = []
        constants = []
        for input_ in graph_inputs([new_graph]):
            if isinstance(input_, Variable):
                if isinstance(input_, Constant):
                    constants.append(input_)
                else:
                    other_inputs.append(input_)

        # foreign inputs are fgraph inputs and shared variables that we need
        # to access through inner inputs
        foreign_inputs = list(set(other_inputs) - set(outer_to_inner.values()))

        # skip further processing if there is nothing to do
        if not constants and not foreign_inputs:
            return new_graph

        replacements = []

        # constants just need to be replaced by copies that the inner
        # `fg` can take ownership of
        for input_ in constants:
            new_input = input_.clone()
            new_input.name = f"{new_input.name}_copied"
            replacements.append((input_, new_input))

        for outer_input in foreign_inputs:
            if getattr(outer_input, "update", False):
                # when aesara.scan() constructs a scan node, it detects
                # shared variables with updates and returns these updates
                # to the user.  we need to do the same thing for every new
                # use of such a variable that is introduced.  it's hard to
                # do that at this point.
                # shared variables with updates inside the inner graph of
                # OpFromGraph are not supported at all, so we don't support
                # introducing those either.
                raise NotImplementedError(
                    f"Replacement introduces shared variable {outer_input} "
                    "which has an update associated with it into "
                    f"the inner graph of {containing_op}. This is not currently "
                    "supported."
                )
            # if this foreign input is not already available
            # as an inner input, connect it through a new
            # inner input
            if outer_input not in outer_to_inner.keys():
                inner_input = utils.safe_new(outer_input, tag="_copy")
                outer_to_inner[outer_input] = inner_input
                extra_inner_inputs.append(inner_input)
                extra_outer_inputs.append(outer_input)

        replacements.extend(outer_to_inner.items())

        (new_graph,) = clone_replace(
            [new_graph], share_inputs=True, replace=replacements
        )
        return new_graph

    new_inner_outputs = map_variables(inner_replacer, inner_outputs)
    new_inner_inputs = list(chain(inner_inputs, extra_inner_inputs))
    new_outer_inputs = list(chain(outer_inputs, extra_outer_inputs))

    return new_inner_inputs, new_outer_inputs, new_inner_outputs


def get_updates_and_outputs(ls):
    """
    This function tries to recognize the updates OrderedDict, the
    list of outputs and the stopping condition returned by the
    lambda expression and arrange them in a predefined order.

    WRITEME: what is the type of ls? how is it formatted?
            if it's not in the predefined order already, how does
            this function know how to put it in that order?

    """

    def is_outputs(elem):
        if isinstance(elem, (list, tuple)) and all(
            [isinstance(x, Variable) for x in elem]
        ):
            return True
        if isinstance(elem, Variable):
            return True
        return False

    def is_updates(elem):
        if isinstance(elem, dict):
            # Make sure the updates will be applied in a deterministic order
            if not isinstance(elem, OrderedDict) and len(elem) > 1:
                warnings.warn(
                    "Expected OrderedDict or OrderedUpdates, got "
                    + str(type(elem))
                    + ". This can make your script non-"
                    "deterministic."
                )
            return True
        # Dictionaries can be given as lists of tuples
        if isinstance(elem, (list, tuple)) and all(
            [isinstance(x, (list, tuple)) and len(x) == 2 for x in elem]
        ):
            return True
        return False

    def is_condition(elem):
        return isinstance(elem, until)

    def _list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        else:
            return [x]

    def _filter(x):
        """
        Ensure `x` is made only of allowed data types.

        Return True iff `x` is made only of lists, tuples, dictionaries, Aesara
        variables or `aesara.scan.utils.until` objects.

        """
        # Is `x` a container we can iterate on?
        iter_on = None
        if isinstance(x, list) or isinstance(x, tuple):
            iter_on = x
        elif isinstance(x, dict):
            iter_on = x.items()
        if iter_on is not None:
            return all(_filter(y) for y in iter_on)
        else:
            return isinstance(x, Variable) or isinstance(x, until)

    if not _filter(ls):
        raise ValueError(
            "The return value of your scan lambda expression may only be "
            "made of lists, tuples, or dictionaries containing Aesara "
            "variables (or `aesara.scan.utils.until` objects for "
            "conditions). In particular if you need to use constant "
            "values, you can use `tensor.constant` to turn them into "
            "Aesara variables."
        )

    if is_outputs(ls):
        return None, _list(ls), OrderedDict()
    if is_updates(ls):
        return None, [], OrderedDict(ls)
    error_msg = (
        f"Scan cannot parse the return value of your lambda expression, which is: {ls}"
    )
    if not isinstance(ls, (list, tuple)):
        raise ValueError(error_msg)
    ls = list(ls)
    deprecation_msg = (
        "The return value of the lambda function"
        " has been restricted. you have to always return first the"
        " outputs (if any), afterwards the updates (if any) and"
        " at the end the conclusion"
    )
    if len(ls) == 2:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                return (None, _list(ls[0]), OrderedDict(ls[1]))
            elif is_condition(ls[1]):
                return (ls[1].condition, _list(ls[0]), OrderedDict())
            else:
                raise ValueError(error_msg)
        elif is_updates(ls[0]):
            if is_outputs(ls[1]):
                raise ValueError(deprecation_msg)
            elif is_condition(ls[1]):
                return (ls[1].condition, [], OrderedDict(ls[0]))
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    elif len(ls) == 3:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                if is_condition(ls[2]):
                    return (ls[2].condition, _list(ls[0]), OrderedDict(ls[1]))
                else:
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    else:
        raise ValueError(error_msg)


def isNaN_or_Inf_or_None(x):
    isNone = x is None
    try:
        isNaN = np.isnan(x)
        isInf = np.isinf(x)
        isStr = isinstance(x, str)
    except Exception:
        isNaN = False
        isInf = False
        isStr = False
    if not isNaN and not isInf:
        try:
            val = get_scalar_constant_value(x)
            isInf = np.isinf(val)
            isNaN = np.isnan(val)
        except Exception:
            isNaN = False
            isInf = False
    if isinstance(x, Constant) and isinstance(x.data, str):
        isStr = True
    else:
        isStr = False
    return isNone or isNaN or isInf or isStr


def expand_empty(tensor_var, size):
    """
    Transforms the shape of a tensor from (d1, d2 ... ) to ( d1+size, d2, ..)
    by adding uninitialized memory at the end of the tensor.

    """

    if size == 0:
        return tensor_var
    shapes = [tensor_var.shape[x] for x in range(tensor_var.ndim)]
    new_shape = [size + shapes[0]] + shapes[1:]
    empty = AllocEmpty(tensor_var.dtype)(*new_shape)

    ret = set_subtensor(empty[: shapes[0]], tensor_var)
    ret.tag.nan_guard_mode_check = False
    return ret


class Validator:
    """
    Check if variables can be expressed without using variables in invalid.

    Parameters
    ----------
    valid_equivalent
        Provides a dictionary mapping some invalid variables to valid ones that
        can be used instead.

    """

    def __init__(self, valid=None, invalid=None, valid_equivalent=None):
        if valid is None:
            valid = []
        if invalid is None:
            invalid = []
        if valid_equivalent is None:
            valid_equivalent = OrderedDict()

        # Nodes that are valid to have in the graph computing outputs
        self.valid = set(valid)

        # Nodes that are NOT valid to have in the graph computing outputs
        self.invalid = set(invalid)

        # Mapping from invalid variables to equivalent valid ones.
        self.valid_equivalent = valid_equivalent.copy()
        self.valid.update(list(valid_equivalent.values()))
        self.invalid.update(list(valid_equivalent.keys()))

    def check(self, out):
        """
        Go backwards in the graph, from out, and check if out is valid.

        If out is a valid node, (out, True) is returned.
        If out is not valid, but has an equivalent e, (e, False) is returned.
        If out is not valid and has no equivalent, None is returned.

        """

        def get_value(out):
            if out in self.valid:
                return out, True
            elif out in self.valid_equivalent:
                return self.valid_equivalent[out], False
            elif out in self.invalid:
                return None
            else:
                raise RuntimeError("This should not happen")

        q = [out]
        while q:
            out = q.pop()
            if out in self.valid:
                continue
            elif out in self.invalid:
                continue

            if out.owner is None:
                if isinstance(out, TensorConstant):
                    self.valid.add(out)
                    continue
                else:
                    # This is an input node and it has not been
                    # explicitly marked as invalid so we can use it
                    self.valid.add(out)
                    continue

            # Process the input if needed
            continue_while = False
            for inp in out.owner.inputs:
                if inp not in self.valid and inp not in self.invalid:
                    q.append(out)
                    q.extend(out.owner.inputs)
                    continue_while = True
                    break
            if continue_while:
                continue
            inputs = [get_value(i) for i in out.owner.inputs]

            # If some inputs are invalid without equivalent, so is out
            if None in inputs:
                self.invalid.add(out)
                continue

            # If some inputs are invalid with equivalent,
            # an equivalent out should be built and returned
            all_inputs = [inp for (inp, is_valid) in inputs]
            equiv_inputs = [inp for (inp, is_valid) in inputs if not is_valid]
            if equiv_inputs:
                cloned_node = out.owner.clone_with_new_inputs(all_inputs)
                cloned_out = cloned_node.outputs[out.index]
                self.invalid.add(out)
                self.valid.add(cloned_out)
                self.valid_equivalent[out] = cloned_out
                continue

            # All inputs are valid, so is out
            self.valid.add(out)

        return get_value(out)


def scan_can_remove_outs(op, out_idxs):
    """
    Looks at all outputs defined by indices ``out_idxs`` and see whom can be
    removed from the scan op without affecting the rest. Return two lists,
    the first one with the indices of outs that can be removed, the second
    with the outputs that can not be removed.

    """
    non_removable = [o for i, o in enumerate(op.outputs) if i not in out_idxs]
    required_inputs = list(graph_inputs(non_removable))

    out_ins = []
    offset = op.n_seqs
    lim = op.n_mit_mot + op.n_mit_sot + op.n_sit_sot
    for idx in range(lim):
        n_ins = len(op.info["tap_array"][idx])
        out_ins += [op.inputs[offset : offset + n_ins]]
        offset += n_ins
    out_ins += [[] for k in range(op.n_nit_sot)]
    out_ins += [[op.inputs[offset + k]] for k in range(op.n_shared_outs)]

    added = True
    out_idxs_mask = [1 for idx in out_idxs]
    while added:
        added = False
        for pos, idx in enumerate(out_idxs):
            if out_idxs_mask[pos] and any([x in required_inputs for x in out_ins[idx]]):
                # This output is required ..
                out_idxs_mask[pos] = 0
                required_inputs += list(graph_inputs([op.outputs[idx]]))
                added = True

    required_outs = [x for i, x in enumerate(out_idxs) if out_idxs_mask[i] == 0]
    not_required = [x for i, x in enumerate(out_idxs) if out_idxs_mask[i] == 1]
    return (required_outs, not_required)


def compress_outs(op, not_required, inputs):
    """
    Helpful function that gets a Scan op, a list of indices indicating
    which outputs are not required anymore and should be removed, and
    a list of inputs to the apply node corresponding to the scan op and
    produces the list of inputs and outputs and the info dictionary where
    the indicated outputs are eliminated. Note that eliminating an output
    means removing its inputs from the inner function and from the
    node inputs, and changing the dictionary.

    """
    info = OrderedDict()
    info["tap_array"] = []
    info["n_seqs"] = op.info["n_seqs"]
    info["n_mit_mot"] = 0
    info["n_mit_mot_outs"] = 0
    info["mit_mot_out_slices"] = []
    info["n_mit_sot"] = 0
    info["n_sit_sot"] = 0
    info["n_shared_outs"] = 0
    info["n_nit_sot"] = 0
    info["truncate_gradient"] = op.info["truncate_gradient"]
    info["name"] = op.info["name"]
    info["gpua"] = op.info["gpua"]
    info["mode"] = op.info["mode"]
    info["as_while"] = op.info["as_while"]
    info["profile"] = op.info["profile"]
    info["allow_gc"] = op.info["allow_gc"]

    op_inputs = op.inputs[: op.n_seqs]
    op_outputs = []
    node_inputs = inputs[: op.n_seqs + 1]
    map_old_new = OrderedDict()

    offset = 0
    ni_offset = op.n_seqs + 1
    i_offset = op.n_seqs
    o_offset = 0
    curr_pos = 0
    for idx in range(op.info["n_mit_mot"]):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info["n_mit_mot"] += 1
            info["tap_array"] += [op.tap_array[offset + idx]]
            info["mit_mot_out_slices"] += [op.mit_mot_out_slices[offset + idx]]
            # input taps
            for jdx in op.tap_array[offset + idx]:
                op_inputs += [op.inputs[i_offset]]
                i_offset += 1
            # output taps
            for jdx in op.mit_mot_out_slices[offset + idx]:
                op_outputs += [op.outputs[o_offset]]
                o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset + idx]]
        else:
            o_offset += len(op.mit_mot_out_slices[offset + idx])
            i_offset += len(op.tap_array[offset + idx])
    info["n_mit_mot_outs"] = len(op_outputs)
    offset += op.n_mit_mot
    ni_offset += op.n_mit_mot

    for idx in range(op.info["n_mit_sot"]):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info["n_mit_sot"] += 1
            info["tap_array"] += [op.tap_array[offset + idx]]
            # input taps
            for jdx in op.tap_array[offset + idx]:
                op_inputs += [op.inputs[i_offset]]
                i_offset += 1
            # output taps
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset + idx]]
        else:
            o_offset += 1
            i_offset += len(op.tap_array[offset + idx])

    offset += op.n_mit_sot
    ni_offset += op.n_mit_sot
    for idx in range(op.info["n_sit_sot"]):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info["n_sit_sot"] += 1
            info["tap_array"] += [op.tap_array[offset + idx]]
            # input taps
            op_inputs += [op.inputs[i_offset]]
            i_offset += 1
            # output taps
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset + idx]]
        else:
            o_offset += 1
            i_offset += 1

    offset += op.n_sit_sot
    ni_offset += op.n_sit_sot
    nit_sot_ins = []
    for idx in range(op.info["n_nit_sot"]):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info["n_nit_sot"] += 1
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            nit_sot_ins += [inputs[ni_offset + idx + op.n_shared_outs]]
        else:
            o_offset += 1

    offset += op.n_nit_sot
    shared_ins = []
    for idx in range(op.info["n_shared_outs"]):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info["n_shared_outs"] += 1
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            op_inputs += [op.inputs[i_offset]]
            i_offset += 1
            shared_ins += [inputs[ni_offset + idx]]
        else:
            o_offset += 1
            i_offset += 1
    node_inputs += shared_ins
    node_inputs += nit_sot_ins
    # other stuff
    op_inputs += op.inputs[i_offset:]
    node_inputs += inputs[ni_offset + op.n_shared_outs + op.n_nit_sot :]
    if op.as_while:
        op_outputs += [op.outputs[o_offset]]
        map_old_new[o_offset] = len(op_outputs) - 1
        # map_old_new[len(op_outputs)-1] = o_offset

    return (op_inputs, op_outputs, info, node_inputs, map_old_new)


def reconstruct_graph(inputs, outputs, tag=None):
    """
    Different interface to clone, that allows you to pass inputs.
    Compared to clone, this method always replaces the inputs with
    new variables of the same type, and returns those (in the same
    order as the original inputs).

    """
    if tag is None:
        tag = ""
    nw_inputs = [safe_new(x, tag) for x in inputs]
    givens = OrderedDict()
    for nw_x, x in zip(nw_inputs, inputs):
        givens[x] = nw_x
    allinputs = list(graph_inputs(outputs))
    for inp in allinputs:
        if isinstance(inp, Constant):
            givens[inp] = inp.clone()

    nw_outputs = clone_replace(outputs, replace=givens)
    return (nw_inputs, nw_outputs)


FieldInfo = namedtuple(
    "FieldInfo", ("name", "agg_name", "index", "inner_index", "agg_index")
)


def safe_index(lst, x):
    try:
        return lst.index(x)
    except ValueError:
        return None


def default_filter_scanargs(x):
    return x.startswith("inner_") or x.startswith("outer_")


class ScanArgs:
    """Parses the inputs and outputs of `Scan` in an easy to manipulate format."""

    default_filter = default_filter_scanargs
    nested_list_fields = ("inner_in_mit_mot", "inner_in_mit_sot", "inner_out_mit_mot")

    def __init__(
        self,
        outer_inputs,
        outer_outputs,
        _inner_inputs,
        _inner_outputs,
        info,
        clone=True,
    ):
        self.n_steps = outer_inputs[0]

        if clone:
            rval = reconstruct_graph(_inner_inputs, _inner_outputs, "")
        else:
            rval = (_inner_inputs, _inner_outputs)

        if info["as_while"]:
            self.cond = [rval[1][-1]]
            inner_outputs = rval[1][:-1]
        else:
            self.cond = []
            inner_outputs = rval[1]
        inner_inputs = rval[0]

        p = 1
        q = 0

        n_seqs = info["n_seqs"]
        self.outer_in_seqs = outer_inputs[p : p + n_seqs]
        self.inner_in_seqs = inner_inputs[q : q + n_seqs]
        p += n_seqs
        q += n_seqs

        n_mit_mot = info["n_mit_mot"]
        n_mit_sot = info["n_mit_sot"]

        self.mit_mot_in_slices = info["tap_array"][:n_mit_mot]
        self.mit_sot_in_slices = info["tap_array"][n_mit_mot : n_mit_mot + n_mit_sot]

        n_mit_mot_ins = sum(len(s) for s in self.mit_mot_in_slices)
        n_mit_sot_ins = sum(len(s) for s in self.mit_sot_in_slices)

        iimm = inner_inputs[q : q + n_mit_mot_ins]
        self.inner_in_mit_mot = []
        qq = 0
        for sl in self.mit_mot_in_slices:
            self.inner_in_mit_mot.append(iimm[qq : qq + len(sl)])
            qq += len(sl)
        q += n_mit_mot_ins

        iims = inner_inputs[q : q + n_mit_sot_ins]
        self.inner_in_mit_sot = []
        qq = 0
        for sl in self.mit_sot_in_slices:
            self.inner_in_mit_sot.append(iims[qq : qq + len(sl)])
            qq += len(sl)
        q += n_mit_sot_ins

        self.outer_in_mit_mot = outer_inputs[p : p + n_mit_mot]
        p += n_mit_mot
        self.outer_in_mit_sot = outer_inputs[p : p + n_mit_sot]
        p += n_mit_sot

        n_sit_sot = info["n_sit_sot"]
        self.outer_in_sit_sot = outer_inputs[p : p + n_sit_sot]
        self.inner_in_sit_sot = inner_inputs[q : q + n_sit_sot]
        p += n_sit_sot
        q += n_sit_sot

        n_shared_outs = info["n_shared_outs"]
        self.outer_in_shared = outer_inputs[p : p + n_shared_outs]
        self.inner_in_shared = inner_inputs[q : q + n_shared_outs]
        p += n_shared_outs
        q += n_shared_outs

        n_nit_sot = info["n_nit_sot"]
        self.outer_in_nit_sot = outer_inputs[p : p + n_nit_sot]
        p += n_nit_sot

        self.outer_in_non_seqs = outer_inputs[p:]
        self.inner_in_non_seqs = inner_inputs[q:]

        # now for the outputs
        p = 0
        q = 0

        self.mit_mot_out_slices = info["mit_mot_out_slices"]
        n_mit_mot_outs = info["n_mit_mot_outs"]
        self.outer_out_mit_mot = outer_outputs[p : p + n_mit_mot]
        iomm = inner_outputs[q : q + n_mit_mot_outs]
        self.inner_out_mit_mot = []
        qq = 0
        for sl in self.mit_mot_out_slices:
            self.inner_out_mit_mot.append(iomm[qq : qq + len(sl)])
            qq += len(sl)
        p += n_mit_mot
        q += n_mit_mot_outs

        self.outer_out_mit_sot = outer_outputs[p : p + n_mit_sot]
        self.inner_out_mit_sot = inner_outputs[q : q + n_mit_sot]
        p += n_mit_sot
        q += n_mit_sot

        self.outer_out_sit_sot = outer_outputs[p : p + n_sit_sot]
        self.inner_out_sit_sot = inner_outputs[q : q + n_sit_sot]
        p += n_sit_sot
        q += n_sit_sot

        self.outer_out_nit_sot = outer_outputs[p : p + n_nit_sot]
        self.inner_out_nit_sot = inner_outputs[q : q + n_nit_sot]
        p += n_nit_sot
        q += n_nit_sot

        self.outer_out_shared = outer_outputs[p : p + n_shared_outs]
        self.inner_out_shared = inner_outputs[q : q + n_shared_outs]
        p += n_shared_outs
        q += n_shared_outs

        assert p == len(outer_outputs)
        assert q == len(inner_outputs)

        self.other_info = OrderedDict()
        for k in (
            "truncate_gradient",
            "name",
            "mode",
            "destroy_map",
            "gpua",
            "as_while",
            "profile",
            "allow_gc",
        ):
            if k in info:
                self.other_info[k] = info[k]

    @staticmethod
    def from_node(node, clone=False):
        from aesara.scan.op import Scan

        if not isinstance(node.op, Scan):
            raise TypeError("{} is not a Scan node".format(node))
        return ScanArgs(
            node.inputs,
            node.outputs,
            node.op.inputs,
            node.op.outputs,
            node.op.info,
            clone=clone,
        )

    @classmethod
    def create_empty(cls):
        info = OrderedDict(
            [
                ("n_seqs", 0),
                ("n_mit_mot", 0),
                ("n_mit_sot", 0),
                ("tap_array", []),
                ("n_sit_sot", 0),
                ("n_nit_sot", 0),
                ("n_shared_outs", 0),
                ("n_mit_mot_outs", 0),
                ("mit_mot_out_slices", []),
                ("truncate_gradient", -1),
                ("name", None),
                ("mode", None),
                ("destroy_map", OrderedDict()),
                ("gpua", False),
                ("as_while", False),
                ("profile", False),
                ("allow_gc", False),
            ]
        )
        res = cls([1], [], [], [], info)
        res.n_steps = None
        return res

    @property
    def n_nit_sot(self):
        # This is just a hack that allows us to use `Scan.get_oinp_iinp_iout_oout_mappings`
        return self.info["n_nit_sot"]

    @property
    def inputs(self):
        # This is just a hack that allows us to use `Scan.get_oinp_iinp_iout_oout_mappings`
        return self.inner_inputs

    @property
    def n_mit_mot(self):
        # This is just a hack that allows us to use `Scan.get_oinp_iinp_iout_oout_mappings`
        return self.info["n_mit_mot"]

    @property
    def var_mappings(self):
        from aesara.scan.op import Scan

        return Scan.get_oinp_iinp_iout_oout_mappings(self)

    @property
    def field_names(self):
        res = ["mit_mot_out_slices", "mit_mot_in_slices", "mit_sot_in_slices"]
        res.extend(
            [
                attr
                for attr in self.__dict__
                if attr.startswith("inner_in")
                or attr.startswith("inner_out")
                or attr.startswith("outer_in")
                or attr.startswith("outer_out")
                or attr == "n_steps"
            ]
        )
        return res

    @property
    def inner_inputs(self):
        return (
            self.inner_in_seqs
            + sum(self.inner_in_mit_mot, [])
            + sum(self.inner_in_mit_sot, [])
            + self.inner_in_sit_sot
            + self.inner_in_shared
            + self.inner_in_non_seqs
        )

    @property
    def outer_inputs(self):
        return (
            [self.n_steps]
            + self.outer_in_seqs
            + self.outer_in_mit_mot
            + self.outer_in_mit_sot
            + self.outer_in_sit_sot
            + self.outer_in_shared
            + self.outer_in_nit_sot
            + self.outer_in_non_seqs
        )

    @property
    def inner_outputs(self):
        return (
            sum(self.inner_out_mit_mot, [])
            + self.inner_out_mit_sot
            + self.inner_out_sit_sot
            + self.inner_out_nit_sot
            + self.inner_out_shared
            + self.cond
        )

    @property
    def outer_outputs(self):
        return (
            self.outer_out_mit_mot
            + self.outer_out_mit_sot
            + self.outer_out_sit_sot
            + self.outer_out_nit_sot
            + self.outer_out_shared
        )

    @property
    def info(self):
        return OrderedDict(
            n_seqs=len(self.outer_in_seqs),
            n_mit_mot=len(self.outer_in_mit_mot),
            n_mit_sot=len(self.outer_in_mit_sot),
            tap_array=(
                self.mit_mot_in_slices
                + self.mit_sot_in_slices
                + [[-1]] * len(self.inner_in_sit_sot)
            ),
            n_sit_sot=len(self.outer_in_sit_sot),
            n_nit_sot=len(self.outer_in_nit_sot),
            n_shared_outs=len(self.outer_in_shared),
            n_mit_mot_outs=sum(len(s) for s in self.mit_mot_out_slices),
            mit_mot_out_slices=self.mit_mot_out_slices,
            **self.other_info,
        )

    def get_alt_field(self, var_info, alt_prefix):
        """Get the alternate input/output field for a given element of `ScanArgs`.
        For example, if `var_info` is in `ScanArgs.outer_out_sit_sot`, then
        `get_alt_field(var_info, "inner_out")` returns the element corresponding
        `var_info` in `ScanArgs.inner_out_sit_sot`.
        Parameters
        ----------
        var_info: TensorVariable or FieldInfo
            The element for which we want the alternate
        alt_prefix: str
            The string prefix for the alternate field type.  It can be one of
            the following: "inner_out", "inner_in", "outer_in", and "outer_out".
        Outputs
        -------
        TensorVariable
        Returns the alternate variable.
        """
        if not isinstance(var_info, FieldInfo):
            var_info = self.find_among_fields(var_info)

        alt_type = var_info.name[(var_info.name.index("_", 6) + 1) :]
        alt_var = getattr(self, "inner_out_{}".format(alt_type))[var_info.index]
        return alt_var

    def find_among_fields(self, i, field_filter=default_filter):
        """Find the type and indices of the field containing a given element.
        NOTE: This only returns the *first* field containing the given element.
        Parameters
        ----------
        i: theano.gof.graph.Variable
            The element to find among this object's fields.
        field_filter: function
            A function passed to `filter` that determines which fields to
            consider.  It must take a string field name and return a truthy
            value.
        Returns
        -------
        A tuple of length 4 containing the field name string, the first index,
        the second index (for nested lists), and the "major" index (i.e. the
        index within the aggregate lists like `self.inner_inputs`,
        `self.outer_outputs`, etc.), or a triple of `None` when no match is
        found.
        """

        field_names = filter(field_filter, self.field_names)

        for field_name in field_names:
            lst = getattr(self, field_name)

            field_prefix = field_name[:8]
            if field_prefix.endswith("in"):
                agg_field_name = "{}puts".format(field_prefix)
            else:
                agg_field_name = "{}tputs".format(field_prefix)

            agg_list = getattr(self, agg_field_name)

            if field_name in self.nested_list_fields:
                for n, sub_lst in enumerate(lst):
                    idx = safe_index(sub_lst, i)
                    if idx is not None:
                        agg_idx = safe_index(agg_list, i)
                        return FieldInfo(field_name, agg_field_name, n, idx, agg_idx)
            else:
                idx = safe_index(lst, i)
                if idx is not None:
                    agg_idx = safe_index(agg_list, i)
                    return FieldInfo(field_name, agg_field_name, idx, None, agg_idx)

        return None

    def _remove_from_fields(self, i, field_filter=default_filter):

        field_info = self.find_among_fields(i, field_filter=field_filter)

        if field_info is None:
            return None

        if field_info.inner_index is not None:
            getattr(self, field_info.name)[field_info.index].remove(i)
        else:
            getattr(self, field_info.name).remove(i)

        return field_info

    def get_dependent_nodes(self, i, seen=None):
        if seen is None:
            seen = {i}
        else:
            seen.add(i)

        var_mappings = self.var_mappings

        field_info = self.find_among_fields(i)

        if field_info is None:
            raise ValueError("{} not found among fields.".format(i))

        # Find the `var_mappings` key suffix that matches the field/set of
        # arguments containing our source node
        if field_info.name[:8].endswith("_in"):
            map_key_suffix = "{}p".format(field_info.name[:8])
        else:
            map_key_suffix = field_info.name[:9]

        dependent_nodes = set()
        for k, v in var_mappings.items():

            if not k.endswith(map_key_suffix):
                continue

            dependent_idx = v[field_info.agg_index]
            dependent_idx = (
                dependent_idx if isinstance(dependent_idx, list) else [dependent_idx]
            )

            # Get the `ScanArgs` field name for the aggregate list property
            # corresponding to these dependent argument types (i.e. either
            # "outer_inputs", "inner_inputs", "inner_outputs", or
            # "outer_outputs").
            # To do this, we need to parse the "shared" prefix of the
            # current `var_mappings` key and append the missing parts so that
            # it either forms `"*_inputs"` or `"*_outputs"`.
            to_agg_field_prefix = k[:9]
            if to_agg_field_prefix.endswith("p"):
                to_agg_field_name = "{}uts".format(to_agg_field_prefix)
            else:
                to_agg_field_name = "{}puts".format(to_agg_field_prefix)

            to_agg_field = getattr(self, to_agg_field_name)

            for d_id in dependent_idx:
                if d_id < 0:
                    continue

                dependent_var = to_agg_field[d_id]

                if dependent_var not in seen:
                    dependent_nodes.add(dependent_var)

        if field_info.name.startswith("inner_in"):
            # If starting from an inner-input, then we need to find any
            # inner-outputs that depend on it.
            for out_n in self.inner_outputs:
                if i in graph_inputs([out_n]):
                    if out_n not in seen:
                        dependent_nodes.add(out_n)

        for n in tuple(dependent_nodes):
            if n in seen:
                continue
            sub_dependent_nodes = self.get_dependent_nodes(n, seen=seen)
            dependent_nodes |= sub_dependent_nodes
            seen |= sub_dependent_nodes

        return dependent_nodes

    def remove_from_fields(self, i, rm_dependents=True):

        if rm_dependents:
            vars_to_remove = self.get_dependent_nodes(i) | {i}
        else:
            vars_to_remove = {i}

        rm_info = []
        for v in vars_to_remove:
            dependent_rm_info = self._remove_from_fields(v)
            rm_info.append((v, dependent_rm_info))

        return rm_info

    def __copy__(self):
        res = object.__new__(type(self))
        res.__dict__.update(self.__dict__)
        # also copy mutable attrs
        for attr in self.__dict__:
            if (
                attr.startswith("inner_in")
                or attr.startswith("inner_out")
                or attr.startswith("outer_in")
                or attr.startswith("outer_out")
                or attr
                in (
                    "mit_mot_out_slices",
                    "mit_mot_in_slices",
                    "mit_sot_in_slices",
                    "other_info",
                )
            ):
                setattr(res, attr, copy.copy(getattr(self, attr)))
        return res

    def merge(self, other):
        res = copy.copy(self)
        for attr in self.__dict__:
            if (
                attr.startswith("inner_in")
                or attr.startswith("inner_out")
                or attr.startswith("outer_in")
                or attr.startswith("outer_out")
                or attr
                in ("mit_mot_out_slices", "mit_mot_in_slices", "mit_sot_in_slices")
            ):
                getattr(res, attr).extend(getattr(other, attr))
        return res

    def __str__(self):
        inner_arg_strs = [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("outer_in") or p == "n_steps"
        ]
        inner_arg_strs += [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("inner_in")
        ]
        inner_arg_strs += [
            "\tmit_mot_in_slices={}".format(self.mit_mot_in_slices),
            "\tmit_sot_in_slices={}".format(self.mit_sot_in_slices),
        ]
        inner_arg_strs += [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("inner_out")
        ]
        inner_arg_strs += [
            "\tmit_mot_out_slices={}".format(self.mit_mot_out_slices),
        ]
        inner_arg_strs += [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("outer_out")
        ]
        res = "ScanArgs(\n{})".format(",\n".join(inner_arg_strs))
        return res

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        for field_name in self.field_names:
            if not hasattr(other, field_name) or getattr(self, field_name) != getattr(
                other, field_name
            ):
                return False

        return True


def forced_replace(out, x, y):
    """
    Check all internal values of the graph that compute the variable ``out``
    for occurrences of values identical with ``x``. If such occurrences are
    encountered then they are replaced with variable ``y``.

    Parameters
    ----------
    out : Aesara Variable
    x : Aesara Variable
    y : Aesara Variable

    Examples
    --------
    out := sigmoid(wu)*(1-sigmoid(wu))
    x := sigmoid(wu)
    forced_replace(out, x, y) := y*(1-y)

    Notes
    -----
    When it find a match, it don't continue on the corresponding inputs.
    """
    if out is None:
        return None

    # ``visited`` is a set of nodes that are already known and don't need to be
    # checked again, speeding up the traversal of multiply-connected graphs.
    visited = set()
    from collections import deque

    q = deque()
    q.append(out)
    to_replace = []
    while q:
        graph = q.popleft()
        if graph in visited:
            continue
        visited.add(graph)
        if equal_computations([graph], [x]):
            to_replace.append((graph, y))
        elif graph.owner:
            q.extendleft(graph.owner.inputs)

    if len(to_replace) == 0:
        return out
    return clone_replace(out, replace=to_replace)
