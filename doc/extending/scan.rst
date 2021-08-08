.. _scan_internals:

Developer documentation for `Scan`
++++++++++++++++++++++++++++++++++

Context
=======

This document is meant to act as reference material for developers working
on Aesara's loop mechanism. This mechanism is called `Scan` and its internals
are highly complex, hence the need for a centralized repository of knowledge
regarding its inner workings.

The `aesara.scan` function is the public-facing interface for looping in
Aesara. Under the hood, this function will perform some processing on its
inputs and instantiate the `Scan` `Op` class which implements the looping
mechanism. It achieves this by compiling its own Aesara function representing
the computation to be done at every iteration of the loop and calling it as
many times as necessary.

The correspondence between the parameters and behaviors of the function and the
`Op` is not always simple since the former is meant for usability and the second
for performance. Since this document is intended to be used by developers
working inside `Scan` itself, it will mostly discuss things from the point of view
of the `Scan` `Op` class. Nonetheless, it will attempt to link those elements to
their corresponding concepts in the `Scan` function as often as is reasonably
practical.


Pre-requisites
==============

The following sections assumes the reader is familiar with the following :

1. Aesara's :ref:`graph structure <extending_aesara>` (`Apply` nodes, `Variable` nodes and `Op`\s)

2. The interface and usage of Aesara's :ref:`scan <lib_scan>` function

Additionally, the :ref:`scan_internals_optimizations` section below assumes
knowledge of:

3. Aesara's :ref:`graph optimizations <optimization>`


Relevant code files
===================

The implementation of `Scan` is spread over several files in
``aesara/scan``.  The different files, and sections of the code they
deal with, are :

* ``basic.py`` implements the `scan` function. The `scan` function
  arranges the arguments of `scan` correctly, constructs the `Scan` `Op` and
  afterwards calls the constructed `Scan` `Op` on the arguments. This function
  takes care of figuring out missing inputs and shared variables.

* ``op.py`` implements the `Scan` `Op` class. The `Scan` respects
  the `Op` interface, and contains most of the logic of the `Scan` operator.

* ``utils.py`` contains several helpful functions used throughout out the
  other files that are specific of the `Scan` operator.

* ``views.py`` contains different views of the `Scan` `Op` that have
  simpler and easier signatures to be used in specific cases.

* ``opt.py`` contains the list of all Aesara graph optimizations for the
  `Scan` operator.


Notation
========

`Scan` being a sizeable and complex module, it has its own naming convention for
functions and variables which this section will attempt to introduce.

A `Scan` `Op` contains an Aesara function representing the computation
that is done in a single iteration of the loop represented by the `Scan` `Op` (in
other words, the computation given by the function provided as value to
`aesara.scan`'s ``fn`` argument ). Whenever we discuss a `Scan` `Op`, the **outer
function** refers to the Aesara function that *contains* the `Scan` `Op` whereas the
**inner function** refers to the Aesara function that is *contained* inside the
`Scan` `Op`.

In the same spirit, the inputs and outputs of the *Apply node wrapping the `Scan`
`Op`* (or *`Scan` node* for short) are referred to as **outer inputs** and **outer
outputs**, respectively, because these inputs and outputs are variables in the
outer function graph. The inputs and outputs of `Scan`'s inner function are
designated **inner inputs** and **inner outputs**, respectively.


`Scan` variables
================

The following are the different types of variables that `Scan` has the
capacity to handle, along with their various caracteristics.

**Sequence** : A sequence is an Aesara variable which `Scan` will iterate
over and give sub-elements to its inner function as input. A sequence
has no associated output. For a sequence variable ``X``, at timestep
``t``, the inner function will receive as input the sequence element
``X[t]``. These variables are used through the argument ``sequences``
of the `aesara.scan` function.

**Non-sequences** : A non-sequence is an Aesara variable which Scan
*will provide as-is* to its inner function. Like a sequence, a
non-sequence has no associated output. For a non-sequence variable
``X``, at timestep ``t``, the inner function will receive as input
the variable ``X``. These variables are used through the argument
``non_sequences`` of the `aesara.scan` function.

**NITSOT (no input tap, single output tap)** : A NITSOT is an output
variable of the inner function that is not fed back as an input to the
next iteration of the inner function. NITSOTs are typically
encountered in situations where `Scan` is used to perform a 'map'
operation (every element in a tensor is independently altered using a
given operation to produce a new tensor) such as squaring every number
in a vector.

**SITSOT (single input tap, single output tap)** : A SITSOT is an output
variable of the inner function that is fed back as an input to the next
iteration of the inner function. A typical setting where a SITSOT might be
encountered is the case where `Scan` is used to compute the cumulative sum over
the elements of a vector and a SITSOT output is employed to act as an
accumulator.

**MITSOT (multiple input taps, single output tap)** : A MITSOT is an
output variable of the inner function that is fed back as an input to
future iterations of the inner function (either multiple future
iterations or a single one that isn't the immediate next one). For
example, a MITSOT might be used in the case where `Scan` is used to
compute the Fibonacci sequence, one term of the sequence at every
timestep, since every computed term needs to be reused to compute the
two next terms of the sequence.

**MITMOT (multiple input taps, multiple output taps)** : These outputs exist
but they cannot be directly created by the user. They can appear in an Aesara
graph as a result of taking the gradient of the output of a `Scan` with respect
to its inputs: This will result in the creation of a new `Scan` node used to
compute the gradients of the first `Scan` node. If the original `Scan` had SITSOTs
or MITSOTs variables, the new `Scan` will use MITMOTs to compute the gradients
through time for these variables.


To synthesize :

===========================================================  =====================================================  ==========================================================  ===========================================================  =========================================================  ======================================================
Type of `Scan` variables                                       Corresponding outer input                              Corresponding inner input at timestep ``t`` (indexed from 0)  Corresponding inner output at timestep ``t`` (indexed from 0)  Corresponding outer output ``t``                             Corresponding argument of the `aesara.scan` function
===========================================================  =====================================================  ==========================================================  ===========================================================  =========================================================  ======================================================
Sequence                                                     Sequence of elements ``X``                                 Individual sequence element ``X[t]``                            *No corresponding inner output*                              *No corresponding outer output*                            `sequences`
Non-Sequence                                                 Any variable ``X``                                         Variable identical to ``X``                                     *No corresponding inner output*                              *No corresponding outer output*                            `non_sequences`
Non-recurring output (NITSOT)                                *No corresponding outer input*                         *No corresponding inner input*                              Output value at timestep ``t``                                 Concatenation of the values of the output at all timestep  `outputs_info`
Singly-recurrent output (SITSOT)                             Initial value (value at timestep ``-1``)                 Output value at previous timestep (``t-1``)                   Output value at timestep ``t``                                 Concatenation of the values of the output at all timestep  `outputs_info`
Multiply-recurrent output (MITSOT)                           Initial values for the required timesteps where ``t<0``  Output value at previous required timesteps                 Output value at timestep ``t``                                 Concatenation of the values of the output at all timestep  `outputs_info`
Multiply-recurrent multiple outputs (MITMOT)                 Initial values for the required timesteps where ``t<0``  Output value at previous required timesteps                 Output values for current and multiple future timesteps      Concatenation of the values of the output at all timestep  *No corresponding argument*
===========================================================  =====================================================  ==========================================================  ===========================================================  =========================================================  ======================================================


.. _scan_internals_optimizations:

Optimizations
=============

`remove_constants_and_unused_inputs_scan`
-----------------------------------------

This optimization serves two purposes, The first is to remove a `Scan` `Op`'s
unused inputs. The second is to take a `Scan` `Op`'s constant inputs and remove
them, instead injecting the constants directly into the graph or the `Scan`
`Op`'s inner function. This will allow constant folding to happen inside the
inner function.


`PushOutNonSeqScan`
-------------------

This optimizations pushes, out of `Scan`'s inner function and into the outer
function, computation that depends only on non-sequence inputs. Such
computation ends up being done every iteration on the same values so moving
it to the outer function to be executed only once, before the `Scan` `Op`,
reduces the amount of computation that needs to be performed.


`PushOutSeqScan`
----------------

This optimization resembles `PushOutNonSeqScan` but it tries to push, out of
the inner function, the computation that only relies on sequence and
non-sequence inputs. The idea behind this optimization is that, when it is
possible to do so, it is generally more computationally efficient to perform
a single operation on a large tensor rather then perform that same operation
many times on many smaller tensors. In many cases, this optimization can
increase memory usage but, in some specific cases, it can also decrease it.


`PushOutScanOutput`
-------------------

This optimizations attempts to push out some of the computation at the end
of the inner function to the outer function, to be executed after the `Scan`
node. Like `PushOutSeqScan`, this optimization aims to replace many operations
on small tensors by few operations on large tensors. It can also lead to
increased memory usage.


`PushOutDot1`
-------------

This is another optimization that attempts to detect certain patterns of
computation in a `Scan` `Op`'s inner function and move this computation to the
outer graph.


`ScanInplaceOptimizer`
----------------------

This optimization attempts to make `Scan` compute its recurrent outputs inplace
on the input tensors that contain their initial states. This optimization can
improve runtime performance as well as reduce memory usage.


`ScanSaveMem`
-------------

This optimizations attempts to determine if a `Scan` node, during its execution,
for any of its outputs, can get away with allocating a memory buffer that is
large enough to contain some of the computed timesteps of that output but not
all of them.

By default, during the execution of a `Scan` node, memory buffers will be
allocated to store the values computed for every output at every iteration.
However, in some cases, there are outputs for which there is only really a
need to store the most recent ``N`` values, not all of them.

For instance, if a `Scan` node has a SITSOT output (last computed value is
fed back as an input at the next iteration) and only the last timestep of
that output is ever used in the outer function, the `ScanSaveMem` optimization
could determine that there is no need to store all computed timesteps for
that SITSOT output. Only the most recently computed timestep ever needs to
be kept in memory.


`ScanMerge`
-----------

This optimization attempts to fuse distinct `Scan` `Op`s into a single `Scan` `Op`
that performs all the computation. The main advantage of merging `Scan` `Op`\s
together comes from the possibility of both original `Op`\s having some
computation in common. In such a setting, this computation ends up being done
twice. The fused `Scan` `Op`, however, would only need to do it once and could
therefore be more computationally efficient. Also, since every `Scan` node
involves a certain overhead, at runtime, reducing the number of `Scan` nodes in
the graph can improve performance.


`scan_merge_inouts`
-------------------

This optimization attempts to merge a `Scan` `Op`'s identical outer inputs as well
as merge its identical outer outputs (outputs that perform the same
computation on the same inputs). This can reduce the amount of computation as
well as result in a simpler graph for both the inner function and the outer
function.


Helper classes and functions
============================

Because of the complexity involved in dealing with `Scan`, a large number of
helper classes and functions have been developed over time to implement
operations commonly needed when dealing with the `Scan` `Op`. The `Scan` `Op`
itself defines a large number of them and others can be found in the file
``utils.py``. This sections aims to point out the most useful ones sorted
by usage.


Accessing/manipulating `Scan`'s inputs and outputs by type
----------------------------------------------------------

Declared in ``utils.py``, the class `ScanArgs` handles the
parsing of the inputs and outputs (both inner and outer) to a format
that is easier to analyze and manipulate. Without this class,
analyzing `Scan`'s inputs and outputs can require convoluted logic
which make for code that is hard to read and to maintain. Because of
this, you should favor using `ScanArgs` when it is practical and
appropriate to do so.

The `Scan` `Op` extends `ScanPropertiesMixin`, which defines a few helper
methods for this purpose, such as `inner_nitsot_outs` or `mitmot_out_taps`, but
they are often poorly documented and easy to misuse. These should be used with
great care.


Navigating between outer inputs/outputs and inner inputs/outputs
----------------------------------------------------------------

Navigation between these four sets of variables can be done in two ways,
depending on the type of navigation that is required.

If the goal is to navigate between variables that are associated with the same
states (e.g. going from an outer sequence input to the corresponding inner
sequence input, going from an inner output associated with a recurrent state
to the inner input(s) associated with that same recurrent state, etc.), then
the `get_oinp_iinp_iout_oout_mappings_mappings` method of the `Scan` `Op` can be used.

This method returns a dictionary with 12 key/value pairs. The keys are listed
below :

*   "outer_inp_from_outer_out"
*   "inner_inp_from_outer_out"
*   "inner_out_from_outer_out"
*   "inner_inp_from_outer_inp"
*   "inner_out_from_outer_inp"
*   "outer_out_from_outer_inp"
*   "outer_inp_from_inner_inp"
*   "inner_out_from_inner_inp"
*   "outer_out_from_inner_inp"
*   "outer_inp_from_inner_out"
*   "inner_inp_from_inner_out"
*   "outer_out_from_inner_out"

Every corresponding value is a dictionary detailing a mapping from one set of
variables to another. For each of those dictionaries the keys are indices of
variables in one set and the values are the indices of the corresponding
variables in another set. For mappings to outer variables, the values are
individual indices or ``-1`` if there is not corresponding outer variable.
For mappings to inner variables, the values are list of indices because
multiple inner variables may be associated with the same state.

If the goal is to navigate between variables that are *connected* (meaning that
one of them is used to compute the other), the method `Scan.connection_pattern`
can be used.  The method `Scan.connection_pattern` returns a list of lists
detailing, for every pair of outer input and outer output whether they are
connected or not.
