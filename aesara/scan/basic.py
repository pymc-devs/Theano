import logging
from collections import OrderedDict

import numpy as np

import aesara.tensor as aet
from aesara.compile import SharedVariable, ops
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable, clone_replace, graph_inputs
from aesara.graph.fg import MissingInputError
from aesara.graph.op import get_test_value
from aesara.graph.utils import TestValueError
from aesara.scan import utils
from aesara.scan.op import Scan, ScanInfo
from aesara.scan.utils import safe_new, traverse
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import minimum
from aesara.tensor.shape import shape_padleft
from aesara.tensor.type import TensorType, integer_dtypes
from aesara.updates import OrderedUpdates


_logger = logging.getLogger("aesara.scan.basic")


def scan(
    fn,
    sequences=None,
    outputs_info=None,
    non_sequences=None,
    n_steps=None,
    truncate_gradient=-1,
    go_backwards=False,
    mode=None,
    name=None,
    profile=False,
    allow_gc=None,
    strict=False,
    return_list=False,
):
    r"""This function constructs and applies a `Scan` `Op` to the provided arguments.

    Parameters
    ----------
    fn
        `fn` is a function that describes the operations involved in one
        step of `scan`. `fn` should construct variables describing the
        output of one iteration step. It should expect as input
        `Variable`\s representing all the slices of the input sequences
        and previous values of the outputs, as well as all other arguments
        given to scan as `non_sequences`. The order in which scan passes
        these variables to `fn`  is the following :

        * all time slices of the first sequence
        * all time slices of the second sequence
        * ...
        * all time slices of the last sequence
        * all past slices of the first output
        * all past slices of the second output
        * ...
        * all past slices of the last output
        * all other arguments (the list given as `non_sequences` to
            `scan`)

        The order of the sequences is the same as the one in the list
        `sequences` given to `scan`. The order of the outputs is the same
        as the order of `outputs_info`. For any sequence or output the
        order of the time slices is the same as the one in which they have
        been given as taps. For example if one writes the following :

        .. code-block:: python

            scan(fn, sequences = [ dict(input= Sequence1, taps = [-3,2,-1])
                                 , Sequence2
                                 , dict(input =  Sequence3, taps = 3) ]
                   , outputs_info = [ dict(initial =  Output1, taps = [-3,-5])
                                    , dict(initial = Output2, taps = None)
                                    , Output3 ]
                   , non_sequences = [ Argument1, Argument2])

        `fn` should expect the following arguments in this given order:

        #. ``sequence1[t-3]``
        #. ``sequence1[t+2]``
        #. ``sequence1[t-1]``
        #. ``sequence2[t]``
        #. ``sequence3[t+3]``
        #. ``output1[t-3]``
        #. ``output1[t-5]``
        #. ``output3[t-1]``
        #. ``argument1``
        #. ``argument2``

        The list of `non_sequences` can also contain shared variables
        used in the function, though `scan` is able to figure those
        out on its own so they can be skipped. For the clarity of the
        code we recommend though to provide them to `scan`. To some extend
        `scan` can also figure out other `non sequences` (not shared)
        even if not passed to `scan` (but used by `fn`). A simple example of
        this would be :

        .. code-block:: python

            import aesara.tensor as aet

            W   = aet.matrix()
            W_2 = W**2

            def f(x):
                return aet.dot(x,W_2)

        The function `fn` is expected to return two things. One is a list of
        outputs ordered in the same order as `outputs_info`, with the
        difference that there should be only one output variable per
        output initial state (even if no tap value is used). Secondly
        `fn` should return an update dictionary (that tells how to
        update any shared variable after each iteration step). The
        dictionary can optionally be given as a list of tuples. There is
        no constraint on the order of these two list, `fn` can return
        either ``(outputs_list, update_dictionary)`` or
        ``(update_dictionary, outputs_list)`` or just one of the two (in
        case the other is empty).

        To use `scan` as a ``while`` loop, the user needs to change the
        function `fn` such that also a stopping condition is returned.
        To do so, one needs to wrap the condition in an `until` class.
        The condition should be returned as a third element, for example:

        .. code-block:: python

            ...
            return [y1_t, y2_t], {x:x+1}, until(x < 50)

        Note that a number of steps--considered in here as the maximum
        number of steps--is still required even though a condition is
        passed.  It is used to allocate memory if needed.

    sequences
        `sequences` is the list of `Variable`\s or ``dict``\s
        describing the sequences `scan` has to iterate over. If a
        sequence is given as wrapped in a ``dict``, then a set of optional
        information can be provided about the sequence. The ``dict``
        should have the following keys:

        * ``input`` (*mandatory*) -- `Variable` representing the
          sequence.

        * ``taps`` -- Temporal taps of the sequence required by `fn`.
          They are provided as a list of integers, where a value ``k``
          impiles that at iteration step ``t`` scan will pass to `fn`
          the slice ``t+k``. Default value is ``[0]``

        All `Variable`\s in the list `sequences` are automatically
        wrapped into a ``dict`` where ``taps`` is set to ``[0]``

    outputs_info
        `outputs_info` is the list of `Variable`\s or ``dict``\s
        describing the initial state of the outputs computed
        recurrently. When the initial states are given as ``dict``\s,
        optional information can be provided about the output corresponding
        to those initial states. The ``dict`` should have the following
        keys:

        * ``initial`` -- A `Variable` that represents the initial
          state of a given output. In case the output is not computed
          recursively (e.g. a ``map``-like function) and does not require an initial
          state, this field can be skipped. Given that only the previous
          time step of the output is used by `fn`, the initial state
          **should have the same shape** as the output and **should not
          involve a downcast** of the data type of the output. If multiple
          time taps are used, the initial state should have one extra
          dimension that covers all the possible taps. For example
          if we use ``-5``, ``-2`` and ``-1`` as past taps, at step ``0``,
          `fn` will require (by an abuse of notation) ``output[-5]``,
          ``output[-2]`` and ``output[-1]``. This will be given by
          the initial state, which in this case should have the shape
          ``(5,) + output.shape``. If this `Variable` containing the initial
          state is called ``init_y`` then ``init_y[0]`` corresponds to
          ``output[-5]``. ``init_y[1]`` corresponds to ``output[-4]``,
          ``init_y[2]`` corresponds to ``output[-3]``, ``init_y[3]``
          corresponds to ``output[-2]``, ``init_y[4]`` corresponds to
          ``output[-1]``.
          While this order might seem strange, it comes natural from splitting
          an array at a given point. assume that we have a array ``x``, and we
          choose ``k`` to be time step ``0``. Then our initial state would be
          ``x[:k]``, while the output will be ``x[k:]``. Looking at this split,
          elements in ``x[:k]`` are ordered exactly like those in ``init_y``.
        * ``taps`` -- Temporal taps of the output that will be passed to
          `fn`. They are provided as a list of *negative* integers,
          where a value ``k`` implies that at iteration step ``t`` scan
          will pass to `fn` the slice ``t+k``.

        `scan` will follow this logic if partial information is given:

        * If an output is not wrapped in a ``dict``, `scan` will wrap
          it in one assuming that you use only the last step of the output
          (i.e. it makes your tap value list equal to ``[-1]``).
        * If you wrap an output in a ``dict`` and you do not provide any
          taps but you provide an initial state it will assume that you are
          using only a tap value of ``-1``.
        * If you wrap an output in a ``dict`` but you do not provide any
          initial state, it assumes that you are not using any form of
          taps.
        * If you provide a ``None`` instead of a `Variable` or a empty
          ``dict`` `scan` assumes that you will not use any taps for
          this output (like for example in case of a ``map``)

        If `outputs_info` is an empty ``list`` or ``None``, `scan` assumes
        that no tap is used for any of the outputs. If information is
        provided just for a subset of the outputs, an exception is
        raised, because there is no convention on how scan should map
        the provided information to the outputs of `fn`.

    non_sequences
        `non_sequences` is the list of arguments that are passed to
        `fn` at each steps. One can choose to exclude variables
        used in `fn` from this list, as long as they are part of the
        computational graph, although--for clarity--this is *not* encouraged.

    n_steps
        `n_steps` is the number of steps to iterate given as an ``int``
        or a scalar `Variable`. If any of the input sequences do not have
        enough elements, `scan` will raise an error. If the value is ``0``, the
        outputs will have ``0`` rows. If `n_steps` is not provided, `scan` will
        figure out the amount of steps it should run given its input
        sequences. ``n_steps < 0`` is not supported anymore.

    truncate_gradient
        `truncate_gradient` is the number of steps to use in truncated
        back-propagation through time (BPTT).  If you compute gradients through
        a `Scan` `Op`, they are computed using BPTT. By providing a different
        value then ``-1``, you choose to use truncated BPTT instead of classical
        BPTT, where you go for only `truncate_gradient` number of steps back in
        time.

    go_backwards
        `go_backwards` is a flag indicating if `scan` should go
        backwards through the sequences. If you think of each sequence
        as indexed by time, making this flag ``True`` would mean that
        `scan` goes back in time, namely that for any sequence it
        starts from the end and goes towards ``0``.

    name
        When profiling `scan`, it is helpful to provide a name for any
        instance of `scan`.
        For example, the profiler will produce an overall profile of your code
        as well as profiles for the computation of one step of each instance of
        `Scan`. The `name` of the instance appears in those profiles and can
        greatly help to disambiguate information.

    mode
        The mode used to compile the inner-graph.
        If you prefer the computations of one step of `scan` to be done
        differently then the entire function, you can use this parameter to
        describe how the computations in this loop are done (see
        `aesara.function` for details about possible values and their meaning).

    profile
        If ``True`` or a non-empty string, a profile object will be created and
        attached to the inner graph of `Scan`. When `profile` is ``True``, the
        profiler results will use the name of the `Scan` instance, otherwise it
        will use the passed string.  The profiler only collects and prints
        information when running the inner graph with the `CVM` `Linker`.

    allow_gc
        Set the value of `allow_gc` for the internal graph of the `Scan`.  If
        set to ``None``, this will use the value of
        `aesara.config.scan__allow_gc`.

        The full `Scan` behavior related to allocation is determined by this
        value and the flag `aesara.config.allow_gc`. If the flag
        `allow_gc` is ``True`` (default) and this `allow_gc` is ``False``
        (default), then we let `Scan` allocate all intermediate memory
        on the first iteration, and they are not garbage collected
        after that first iteration; this is determined by `allow_gc`. This can
        speed up allocation of the subsequent iterations. All those temporary
        allocations are freed at the end of all iterations; this is what the
        flag `aesara.config.allow_gc` means.

        If you use pre-allocation and this `Scan` is on GPU, the speed up from
        `allow_gc` is small. If you are missing memory, disabling `allow_gc`
        could help you run graph that request much memory.

    strict
        If ``True``, all the shared variables used in `fn` must be provided as a
        part of `non_sequences` or `sequences`.

    return_list
        If ``True``, will always return a ``list``, even if there is only one output.

    Returns
    -------
    tuple
        ``tuple`` of the form ``(outputs, updates)``.
        ``outputs`` is either a `Variable` or a ``list`` of `Variable`\s
        representing the outputs in the same order as in `outputs_info`.
        ``updates`` is a subclass of ``dict`` specifying the update rules for
        all shared variables used in `Scan`.
        This ``dict`` should be passed to `aesara.function` when you compile
        your function.

    """
    # General observation : this code is executed only once, at creation
    # of the computational graph, so we don't yet need to be smart about
    # anything (to speed things up)

    ##
    # Step 1. Wrap all inputs in dictionaries and add default values
    ##

    # check if inputs are just single variables instead of lists
    def wrap_into_list(x):
        """
        Wrap the input into a list if it is not already a list.

        """
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    seqs = wrap_into_list(sequences)
    outs_info = wrap_into_list(outputs_info)

    # Make sure we get rid of numpy arrays or ints or anything like that
    # passed as inputs to scan
    non_seqs = []
    for elem in wrap_into_list(non_sequences):
        if not isinstance(elem, Variable):
            non_seqs.append(aet.as_tensor_variable(elem))
        else:
            non_seqs.append(elem)

    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    n_fixed_steps = None

    if isinstance(n_steps, (float, int)):
        n_fixed_steps = int(n_steps)
    else:
        try:
            n_fixed_steps = aet.get_scalar_constant_value(n_steps)
        except NotScalarConstantError:
            n_fixed_steps = None

    # Check n_steps is an int
    if hasattr(n_steps, "dtype") and str(n_steps.dtype) not in integer_dtypes:
        raise ValueError(f" n_steps must be an int. dtype provided is {n_steps.dtype}")

    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)

    return_steps = OrderedDict()
    # wrap sequences in a dictionary if they are not already dictionaries
    for i in range(n_seqs):
        if not isinstance(seqs[i], dict):
            seqs[i] = OrderedDict([("input", seqs[i]), ("taps", [0])])
        elif seqs[i].get("taps", None) is not None:
            seqs[i]["taps"] = wrap_into_list(seqs[i]["taps"])
        elif seqs[i].get("taps", None) is None:
            # seqs dictionary does not have the ``taps`` key
            seqs[i]["taps"] = [0]

    # wrap outputs info in a dictionary if they are not already in one
    for i in range(n_outs):
        if outs_info[i] is not None:
            if isinstance(outs_info[i], dict):
                if outs_info[i].get("return_steps", None) is not None:
                    raise DeprecationWarning(
                        "Using `return_steps` has been deprecated. "
                        "Simply select the entries you need using a "
                        "subtensor. Scan will optimize memory "
                        "consumption, so do not worry about that."
                    )
                # END

            if not isinstance(outs_info[i], dict):
                # by default any output has a tap value of -1
                outs_info[i] = OrderedDict([("initial", outs_info[i]), ("taps", [-1])])
            elif (
                outs_info[i].get("initial", None) is None
                and outs_info[i].get("taps", None) is not None
            ):
                # ^ no initial state but taps provided
                raise ValueError(
                    "If you are using slices of an output "
                    "you need to provide a initial state "
                    f"for it: {outs_info[i]}"
                )
            elif (
                outs_info[i].get("initial", None) is not None
                and outs_info[i].get("taps", None) is None
            ):
                # ^ initial state but taps not provided
                if "taps" in outs_info[i]:
                    # ^ explicitly provided a None for taps
                    _logger.warning(
                        f"Output {getattr(outs_info[i]['initial'], 'name', 'None')} (index {i}) has a initial "
                        "state but taps is explicitly set to None ",
                    )
                outs_info[i]["taps"] = [-1]
            elif outs_info[i].get("taps", None) is not None:
                # Check that taps are valid (< 0 and all dfferent)
                taps = outs_info[i]["taps"]
                if len(taps) > len(set(taps)):
                    raise ValueError(
                        ("All the taps must be different in " " `outputs_info`"),
                        outs_info[i],
                    )
                for t in taps:
                    if t >= 0:
                        raise ValueError(
                            ("All the tap values must be " "smaller than 0."),
                            outs_info[i],
                        )
        else:
            # if a None is provided as the output info we replace it
            # with an empty OrdereDict() to simplify handling
            outs_info[i] = OrderedDict()

    ##
    # Step 2. Generate inputs and outputs of the inner functions
    # for compiling a dummy function (Iteration #1)
    ##

    # create aesara inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new and complete list of inputs and
    #        outputs

    n_seqs = 0
    scan_seqs = []  # Variables passed as inputs to the scan op
    inner_seqs = []  # Variables passed as inputs to the inner function
    inner_slices = []  # Actual slices if scan is removed from the picture
    # go through sequences picking up time slices as needed
    for i, seq in enumerate(seqs):
        # Note that you can have something like no taps for
        # a sequence, though is highly unlikely in practice
        if "taps" in seq:
            # go through the indicated slice
            mintap = np.min(seq["taps"])
            maxtap = np.max(seq["taps"])
            # We cut the sequence such that seq[i] to correspond to
            # seq[i-k]. For the purposes of cutting the sequences, we
            # need to pretend tap 0 is used to avoid cutting the sequences
            # too long if the taps are all lower or all higher than 0.
            maxtap_proxy = max(maxtap, 0)
            mintap_proxy = min(mintap, 0)
            for k in seq["taps"]:
                # create one slice of the input
                # Later on, if we decide not to use scan because we are
                # going for just one step, it makes things easier if we
                # compute the correct outputs here. This way we can use
                # the output of the lambda expression directly to replace
                # the output of scan.

                # If not we need to use copies, that will be replaced at
                # each frame by the corresponding slice
                actual_slice = seq["input"][k - mintap_proxy]
                _seq_val = aet.as_tensor_variable(seq["input"])
                _seq_val_slice = _seq_val[k - mintap_proxy]
                nw_slice = _seq_val_slice.type()

                # Try to transfer test_value to the new variable
                if config.compute_test_value != "off":
                    try:
                        nw_slice.tag.test_value = get_test_value(_seq_val_slice)
                    except TestValueError:
                        if config.compute_test_value != "ignore":
                            # No need to print a warning or raise an error now,
                            # it will be done when fn will be called.
                            _logger.warning(
                                (
                                    "Cannot compute test value for "
                                    "the inner function of scan, input value "
                                    "missing {}"
                                ).format(_seq_val_slice)
                            )

                # Add names to slices for debugging and pretty printing ..
                # that is if the input already has a name
                if getattr(seq["input"], "name", None) is not None:
                    if k > 0:
                        nw_name = seq["input"].name + f"[t+{int(k)}]"
                    elif k == 0:
                        nw_name = seq["input"].name + "[t]"
                    else:
                        nw_name = seq["input"].name + f"[t{int(k)}]"
                    nw_slice.name = nw_name

                start = k - mintap_proxy
                nw_name = None
                if k == maxtap_proxy:
                    nw_seq = seq["input"][start:]
                    if getattr(seq["input"], "name", None) is not None:
                        nw_name = seq["input"].name + f"[{int(start)}:]"
                else:
                    end = -(maxtap_proxy - k)
                    nw_seq = seq["input"][start:end]
                    if getattr(seq["input"], "name", None) is not None:
                        nw_name = seq["input"].name + f"[{int(start)}:{int(end)}]"

                if go_backwards:
                    nw_seq = nw_seq[::-1]

                scan_seqs.append(nw_seq)
                inner_seqs.append(nw_slice)
                inner_slices.append(actual_slice)
                n_seqs += 1
                # Add names -- it helps a lot when debugging
                if nw_name is not None:
                    nw_seq.name = nw_name

    # Since we've added all sequences now we need to level them up based on
    # n_steps or their different shapes
    lengths_vec = []
    for seq in scan_seqs:
        lengths_vec.append(seq.shape[0])

    if not utils.isNaN_or_Inf_or_None(n_steps):
        # ^ N_steps should also be considered
        lengths_vec.append(aet.as_tensor(n_steps))

    if len(lengths_vec) == 0:
        # ^ No information about the number of steps
        raise ValueError(
            "No information about the number of steps "
            "provided. Either provide a value for "
            "n_steps argument of scan or provide an input "
            "sequence"
        )

    # If the user has provided the number of steps, do that regardless ( and
    # raise an error if the sequences are not long enough )
    if utils.isNaN_or_Inf_or_None(n_steps):
        actual_n_steps = lengths_vec[0]
        for contestant in lengths_vec[1:]:
            actual_n_steps = minimum(actual_n_steps, contestant)
    else:
        actual_n_steps = aet.as_tensor(n_steps)

    scan_seqs = [seq[:actual_n_steps] for seq in scan_seqs]
    # Conventions :
    #   mit_mot = multiple input taps, multiple output taps ( only provided
    #             by the gradient function )
    #   mit_sot = multiple input taps, single output tap (t + 0)
    #   sit_sot = single input tap, single output tap (t + 0)
    #   nit_sot = no input tap, single output tap (t + 0)

    # MIT_MOT -- not provided by the user only by the grad function
    n_mit_mot = 0
    n_mit_mot_outs = 0
    mit_mot_scan_inputs = []
    mit_mot_inner_inputs = []
    mit_mot_inner_outputs = []
    mit_mot_out_slices = []

    # SIT_SOT -- provided by the user
    n_mit_sot = 0
    mit_sot_scan_inputs = []
    mit_sot_inner_inputs = []
    mit_sot_inner_slices = []
    mit_sot_inner_outputs = []
    mit_sot_return_steps = OrderedDict()
    mit_sot_tap_array = []
    mit_sot_rightOrder = []

    n_sit_sot = 0
    sit_sot_scan_inputs = []
    sit_sot_inner_inputs = []
    sit_sot_inner_slices = []
    sit_sot_inner_outputs = []
    sit_sot_return_steps = OrderedDict()
    sit_sot_rightOrder = []

    # go through outputs picking up time slices as needed
    for i, init_out in enumerate(outs_info):
        # Note that our convention dictates that if an output uses
        # just the previous time step, as a initial state we will only
        # provide a tensor of the same dimension as one time step; This
        # makes code much cleaner for those who do not use taps. Otherwise
        # they would always had to shape_padleft the initial state ..
        # which is ugly
        if init_out.get("taps", None) == [-1]:

            actual_arg = init_out["initial"]
            if not isinstance(actual_arg, Variable):
                actual_arg = aet.as_tensor_variable(actual_arg)
            arg = safe_new(actual_arg)
            if isinstance(arg, Constant):
                # safe new returns a clone of the constants, but that is not
                # what we need for initial states
                arg = arg.type()

            # Try to transfer test_value to the new variable
            if config.compute_test_value != "off":
                try:
                    arg.tag.test_value = get_test_value(actual_arg)
                except TestValueError:
                    if config.compute_test_value != "ignore":
                        _logger.warning(
                            (
                                "Cannot compute test value for the "
                                "inner function of scan, test value missing: {}"
                            ).format(actual_arg)
                        )

            if getattr(init_out["initial"], "name", None) is not None:
                arg.name = init_out["initial"].name + "[t-1]"

            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            sit_sot_scan_inputs.append(
                utils.expand_empty(
                    aet.unbroadcast(shape_padleft(actual_arg), 0),
                    actual_n_steps,
                )
            )

            sit_sot_inner_slices.append(actual_arg)
            if i in return_steps:
                sit_sot_return_steps[n_sit_sot] = return_steps[i]
            sit_sot_inner_inputs.append(arg)
            sit_sot_rightOrder.append(i)
            n_sit_sot += 1

        elif init_out.get("taps", None):

            if np.any(np.array(init_out.get("taps", [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError("Can not use future taps of outputs", init_out)
            # go through the taps
            mintap = abs(np.min(init_out["taps"]))
            mit_sot_tap_array.append(init_out["taps"])
            # Sequence
            mit_sot_scan_inputs.append(
                utils.expand_empty(init_out["initial"][:mintap], actual_n_steps)
            )

            if i in return_steps:
                mit_sot_return_steps[n_mit_sot] = return_steps[i]
            mit_sot_rightOrder.append(i)
            n_mit_sot += 1
            for k in init_out["taps"]:
                # create a new slice
                actual_nw_slice = init_out["initial"][k + mintap]
                _init_out_var = aet.as_tensor_variable(init_out["initial"])
                _init_out_var_slice = _init_out_var[k + mintap]
                nw_slice = _init_out_var_slice.type()

                # Try to transfer test_value to the new variable
                if config.compute_test_value != "off":
                    try:
                        nw_slice.tag.test_value = get_test_value(_init_out_var_slice)
                    except TestValueError:
                        if config.compute_test_value != "ignore":
                            _logger.warning(
                                (
                                    "Cannot compute test value for "
                                    "the inner function of scan, test value "
                                    "missing: {}"
                                ).format(_init_out_var_slice)
                            )

                # give it a name or debugging and pretty printing
                if getattr(init_out["initial"], "name", None) is not None:
                    if k > 0:
                        nw_slice.name = init_out["initial"].name + f"[t+{int(k)}]"
                    elif k == 0:
                        nw_slice.name = init_out["initial"].name + "[t]"
                    else:
                        nw_slice.name = init_out["initial"].name + f"[t{int(k)}]"
                mit_sot_inner_inputs.append(nw_slice)
                mit_sot_inner_slices.append(actual_nw_slice)
        # NOTE: there is another case, in which we do not want to provide
        #      any previous value of the output to the inner function (i.e.
        #      a map); in that case we do not have to do anything ..

    # Re-order args
    max_mit_sot = np.max([-1] + mit_sot_rightOrder) + 1
    max_sit_sot = np.max([-1] + sit_sot_rightOrder) + 1
    n_elems = np.max([max_mit_sot, max_sit_sot])
    _ordered_args = [[] for x in range(n_elems)]
    offset = 0
    for idx in range(n_mit_sot):
        n_inputs = len(mit_sot_tap_array[idx])
        if n_fixed_steps in [1, -1]:
            _ordered_args[mit_sot_rightOrder[idx]] = mit_sot_inner_slices[
                offset : offset + n_inputs
            ]
        else:
            _ordered_args[mit_sot_rightOrder[idx]] = mit_sot_inner_inputs[
                offset : offset + n_inputs
            ]
        offset += n_inputs

    for idx in range(n_sit_sot):
        if n_fixed_steps in [1, -1]:
            _ordered_args[sit_sot_rightOrder[idx]] = [sit_sot_inner_slices[idx]]
        else:
            _ordered_args[sit_sot_rightOrder[idx]] = [sit_sot_inner_inputs[idx]]

    ordered_args = []
    for ls in _ordered_args:
        ordered_args += ls
    if n_fixed_steps in [1, -1]:
        args = inner_slices + ordered_args + non_seqs

    else:
        args = inner_seqs + ordered_args + non_seqs

    # add only the non-shared variables and non-constants to the arguments of
    # the dummy function [ a function should not get shared variables or
    # constants as input ]
    dummy_args = [
        arg
        for arg in args
        if (not isinstance(arg, SharedVariable) and not isinstance(arg, Constant))
    ]
    # when we apply the lambda expression we get a mixture of update rules
    # and outputs that needs to be separated

    condition, outputs, updates = utils.get_updates_and_outputs(fn(*args))
    if condition is not None:
        as_while = True
    else:
        as_while = False
    ##
    # Step 3. Check if we actually need scan and remove it if we don't
    ##

    if n_fixed_steps in [1, -1]:
        # We do not need to use the scan op anymore, so we can just return
        # the outputs and updates we have
        if condition is not None:
            _logger.warning(
                (
                    "When the number of steps is fixed and equal "
                    "to 1, the provided stopping condition, {} is ignored",
                ).format(condition)
            )

        for pos, inner_out in enumerate(outputs):
            # we need to see if we need to pad our sequences with an
            # unbroadcastable dimension; case example : we return an
            # output for which we want all intermediate. If n_steps is 1
            # then, if we return the output as given by the innner function
            # this will represent only a slice and it will have one
            # dimension less.
            if isinstance(inner_out.type, TensorType) and return_steps.get(pos, 0) != 1:
                outputs[pos] = aet.unbroadcast(shape_padleft(inner_out), 0)

        if return_list is not True and len(outputs) == 1:
            outputs = outputs[0]

        return (outputs, updates)

    ##
    # Step 4. Compile the dummy function
    ##

    # We can now compile a dummy function just to see what shared variable
    # we have and what are their update rules (note that the user has
    # the option not to pass the shared variable to scan, so we need to
    # pick them manually and add them to scan)
    # make the compilation as fast as possible by not applying any
    # optimization or conversion to C [ note this region is not important
    # for performance so we can do stuff as unoptimal as we wish ]

    # extract still missing inputs (there still might be so) and add them
    # as non sequences at the end of our args
    if condition is not None:
        outputs.append(condition)
    fake_nonseqs = [x.type() for x in non_seqs]
    fake_outputs = clone_replace(
        outputs, replace=OrderedDict(zip(non_seqs, fake_nonseqs))
    )
    all_inputs = filter(
        lambda x: (
            isinstance(x, Variable)
            and not isinstance(x, SharedVariable)
            and not isinstance(x, Constant)
        ),
        graph_inputs(fake_outputs),
    )
    extra_inputs = [x for x in all_inputs if x not in args + fake_nonseqs]
    non_seqs += extra_inputs
    # Note we do not use all_inputs directly since the order of variables
    # in args is quite important
    dummy_args += extra_inputs

    dummy_outs = outputs
    # Perform a try-except to provide a meaningful error message to the
    # user if inputs of the inner function are missing.
    try:
        dummy_f = function(
            dummy_args,
            dummy_outs,
            updates=updates,
            mode=Mode(linker="py", optimizer=None),
            on_unused_input="ignore",
            profile=False,
        )
    except MissingInputError as err:
        msg = (
            "\nPlease pass this variable to the scan's inner function. Do "
            "not forget to also pass it to the `non_sequences` attribute "
            "of scan."
        )
        raise MissingInputError(err.args[0] + msg)
    ##
    # Step 5. Re-arange inputs of scan into a more strict order
    ##

    # Step 5.0 Check the outputs of the dummy function to see if they
    # match with user provided data

    # if the number of outputs to the function does not match the number of
    # assumed outputs until now (provided by the user) there can be
    # only one explanation: No information is provided for any of the
    # outputs (i.e. we are dealing with a map)
    tmp_dummy_f_outs = len(dummy_f.maker.outputs)
    if as_while:
        tmp_dummy_f_outs -= 1
    if not (tmp_dummy_f_outs == n_outs or outs_info == []):
        raise ValueError(
            "Please provide None as outputs_info for "
            "any output that does not feed back into "
            "scan (i.e. it behaves like a map) "
        )

    if outs_info == []:
        n_outs = len(dummy_f.maker.outputs)
        if as_while:
            n_outs = n_outs - 1
        outs_info = [OrderedDict() for x in range(n_outs)]

    # Step 5.1 Outputs with taps different then -1

    for i, out in enumerate(outs_info):
        if "taps" in out and out["taps"] != [-1]:
            mit_sot_inner_outputs.append(outputs[i])

    # Step 5.2 Outputs with tap equal to -1
    for i, out in enumerate(outs_info):
        if "taps" in out and out["taps"] == [-1]:
            sit_sot_inner_outputs.append(outputs[i])

    # Step 5.3 Outputs that correspond to update rules of shared variables
    givens = OrderedDict()
    n_shared_outs = 0
    shared_scan_inputs = []
    shared_inner_inputs = []
    shared_inner_outputs = []
    sit_sot_shared = []
    for input in dummy_f.maker.expanded_inputs:
        if isinstance(input.variable, SharedVariable) and input.update:
            new_var = safe_new(input.variable)
            if getattr(input.variable, "name", None) is not None:
                new_var.name = input.variable.name + "_copy"
            if isinstance(new_var.type, ops.expandable_types):
                sit_sot_inner_inputs.append(new_var)
                sit_sot_scan_inputs.append(
                    utils.expand_empty(
                        aet.unbroadcast(shape_padleft(input.variable), 0),
                        actual_n_steps,
                    )
                )
                tensor_update = aet.as_tensor_variable(input.update)
                sit_sot_inner_outputs.append(tensor_update)
                # Not that pos is not a negative index. The sign of pos is used
                # as a flag to indicate if this output should be part of the
                # update rules or part of the standard outputs of scan.
                # If `pos` is positive than it corresponds to the standard
                # outputs of scan and it refers to output of index `pos`. If `pos`
                # is negative that it corresponds to update rules of scan and it
                # refers to update rule of index -1 - `pos`.
                sit_sot_rightOrder.append(-1 - len(sit_sot_shared))
                sit_sot_shared.append(input.variable)
                givens[input.variable] = new_var

            else:
                shared_inner_inputs.append(new_var)
                shared_scan_inputs.append(input.variable)
                shared_inner_outputs.append(input.update)
                givens[input.variable] = new_var
                n_shared_outs += 1
    n_sit_sot = len(sit_sot_inner_inputs)
    # Step 5.4 Outputs with no taps used in the input
    n_nit_sot = 0
    nit_sot_inner_outputs = []
    nit_sot_return_steps = OrderedDict()
    nit_sot_rightOrder = []
    for i, out in enumerate(outs_info):
        if "taps" not in out:
            nit_sot_inner_outputs.append(outputs[i])
            if i in return_steps:
                nit_sot_return_steps[n_nit_sot] = return_steps[i]
            nit_sot_rightOrder.append(i)
            n_nit_sot += 1

    # Step 5.5 all other arguments including extra inputs
    other_scan_args = []
    other_inner_args = []

    other_scan_args += [
        arg
        for arg in non_seqs
        if (not isinstance(arg, SharedVariable) and not isinstance(arg, Constant))
    ]

    # Step 5.6 all shared variables with no update rules
    other_inner_args += [
        safe_new(arg, "_copy")
        for arg in non_seqs
        if (not isinstance(arg, SharedVariable) and not isinstance(arg, Constant))
    ]

    givens.update(OrderedDict(zip(other_scan_args, other_inner_args)))

    if strict:
        non_seqs_set = set(non_sequences if non_sequences is not None else [])

        other_shared_scan_args = [
            arg.variable
            for arg in dummy_f.maker.expanded_inputs
            if (
                isinstance(arg.variable, SharedVariable)
                and not arg.update
                and arg.variable in non_seqs_set
            )
        ]
        other_shared_inner_args = [
            safe_new(arg.variable, "_copy")
            for arg in dummy_f.maker.expanded_inputs
            if (
                isinstance(arg.variable, SharedVariable)
                and not arg.update
                and arg.variable in non_seqs_set
            )
        ]
    else:
        other_shared_scan_args = [
            arg.variable
            for arg in dummy_f.maker.expanded_inputs
            if (isinstance(arg.variable, SharedVariable) and not arg.update)
        ]
        other_shared_inner_args = [
            safe_new(arg.variable, "_copy")
            for arg in dummy_f.maker.expanded_inputs
            if (isinstance(arg.variable, SharedVariable) and not arg.update)
        ]
    givens.update(OrderedDict(zip(other_shared_scan_args, other_shared_inner_args)))

    ##
    # Step 6. Re-order the outputs and clone them replacing things
    # using the givens
    ##
    inner_inputs = (
        inner_seqs
        + mit_mot_inner_inputs
        + mit_sot_inner_inputs
        + sit_sot_inner_inputs
        + shared_inner_inputs
        + other_shared_inner_args
        + other_inner_args
    )

    inner_outs = (
        mit_mot_inner_outputs
        + mit_sot_inner_outputs
        + sit_sot_inner_outputs
        + nit_sot_inner_outputs
        + shared_inner_outputs
    )
    if condition is not None:
        inner_outs.append(condition)
    # gpuarray is imported here, instead of being imported on top of
    # the file because that would force on the user some dependencies that we
    # might do not want to. Currently we are working on removing the
    # dependencies on sandbox code completely.
    from aesara import gpuarray

    if gpuarray.pygpu_activated:
        # very often we end up in this situation when we want to
        # replace w with w_copy, where w is a GPU variable
        # and w_copy is TensorType. This is caused because shared
        # variables are put on GPU right away >:| ,
        new_givens = OrderedDict()

        for w, w_copy in givens.items():
            if isinstance(w.type, gpuarray.GpuArrayType) and isinstance(
                w_copy.type, TensorType
            ):
                for o in inner_outs:
                    new_givens = traverse(o, w, w_copy, new_givens)
            else:
                new_givens[w] = w_copy
    else:
        new_givens = givens

    new_outs = clone_replace(inner_outs, replace=new_givens)

    ##
    # Step 7. Create the Scan Op
    ##

    tap_array = tuple(tuple(v) for v in mit_sot_tap_array) + tuple(
        (-1,) for x in range(n_sit_sot)
    )
    if allow_gc is None:
        allow_gc = config.scan__allow_gc

    info = ScanInfo(
        tap_array=tap_array,
        n_seqs=n_seqs,
        n_mit_mot=n_mit_mot,
        n_mit_mot_outs=n_mit_mot_outs,
        mit_mot_out_slices=tuple(tuple(v) for v in mit_mot_out_slices),
        n_mit_sot=n_mit_sot,
        n_sit_sot=n_sit_sot,
        n_shared_outs=n_shared_outs,
        n_nit_sot=n_nit_sot,
        truncate_gradient=truncate_gradient,
        name=name,
        gpua=False,
        as_while=as_while,
        profile=profile,
        allow_gc=allow_gc,
        strict=strict,
    )

    local_op = Scan(inner_inputs, new_outs, info, mode)

    ##
    # Step 8. Compute the outputs using the scan op
    ##
    _scan_inputs = (
        scan_seqs
        + mit_mot_scan_inputs
        + mit_sot_scan_inputs
        + sit_sot_scan_inputs
        + shared_scan_inputs
        + [actual_n_steps for x in range(n_nit_sot)]
        + other_shared_scan_args
        + other_scan_args
    )

    scan_inputs = []
    for arg in [actual_n_steps] + _scan_inputs:
        try:
            arg = aet.as_tensor_variable(arg)
        except TypeError:
            # This happens for Random States for e.g. but it is a good way
            # to make sure all inputs are tensors.
            pass
        scan_inputs += [arg]
    scan_outs = local_op(*scan_inputs)
    if type(scan_outs) not in (list, tuple):
        scan_outs = [scan_outs]
    ##
    # Step 9. Figure out which outs are update rules for shared variables
    # and so on ...
    ##

    update_map = OrderedUpdates()

    def remove_dimensions(outs, steps_return, offsets=None):
        out_ls = []
        for idx, out in enumerate(outs):
            if idx in steps_return:
                if steps_return[idx] > 1:
                    out_ls.append(out[-steps_return[idx] :])
                else:
                    out_ls.append(out[-1])
            else:
                if offsets is None:
                    out_ls.append(out)
                else:
                    out_ls.append(out[offsets[idx] :])
        return out_ls

    offset = n_mit_mot
    offsets = [abs(np.min(x)) for x in mit_sot_tap_array]
    mit_sot_outs = remove_dimensions(
        scan_outs[offset : offset + n_mit_sot], mit_sot_return_steps, offsets
    )

    offset += n_mit_sot
    offsets = [1 for x in range(n_sit_sot)]
    sit_sot_outs = remove_dimensions(
        scan_outs[offset : offset + n_sit_sot], sit_sot_return_steps, offsets
    )

    offset += n_sit_sot
    nit_sot_outs = remove_dimensions(
        scan_outs[offset : offset + n_nit_sot], nit_sot_return_steps
    )

    offset += n_nit_sot
    for idx, update_rule in enumerate(scan_outs[offset : offset + n_shared_outs]):
        update_map[shared_scan_inputs[idx]] = update_rule

    _scan_out_list = mit_sot_outs + sit_sot_outs + nit_sot_outs
    # Step 10. I need to reorder the outputs to be in the order expected by
    # the user
    rightOrder = mit_sot_rightOrder + sit_sot_rightOrder + nit_sot_rightOrder
    scan_out_list = [None] * len(rightOrder)
    for idx, pos in enumerate(rightOrder):
        if pos >= 0:
            scan_out_list[pos] = _scan_out_list[idx]
        else:
            # Not that pos is not a negative index. The sign of pos is used
            # as a flag to indicate if this output should be part of the
            # update rules or part of the standard outputs of scan.
            # If `pos` is positive than it corresponds to the standard
            # outputs of scan and it refers to output of index `pos`. If `pos`
            # is negative that it corresponds to update rules of scan and it
            # refers to update rule of index -1 - `pos`.
            update_map[sit_sot_shared[abs(pos) - 1]] = _scan_out_list[idx][-1]
    scan_out_list = [x for x in scan_out_list if x is not None]
    if return_list is not True and len(scan_out_list) == 1:
        scan_out_list = scan_out_list[0]
    elif len(scan_out_list) == 0:
        scan_out_list = None

    return (scan_out_list, update_map)
