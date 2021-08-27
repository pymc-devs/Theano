# from collections.abc import Mapping

from cons.core import ConsError, _car, _cdr
from etuples import apply, etuple, rands, rator
from etuples.core import ExpressionTuple
from kanren import run
from unification import var, variables

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.graph.opt import LocalOptimizer


# from unification import Var, reify, unify
# from unification.core import _reify, _unify

rator.add((Variable,), lambda x: x.owner and x.owner.op)


def car_Variable(x):
    if x.owner:
        return x.owner.op
    else:
        raise ConsError("Not a cons pair.")


_car.add((Variable,), car_Variable)

rands.add((Variable,), lambda x: x.owner and x.owner.inputs)


def cdr_Variable(x):
    if x.owner:
        x_e = etuple(_car(x), *x.owner.inputs, evaled_obj=x)
    else:
        raise ConsError("Not a cons pair.")

    return x_e[1:]


_cdr.add((Variable,), cdr_Variable)


def apply_Op_ExpressionTuple(op, etuple_arg):
    res = op.make_node(*etuple_arg)

    if hasattr(res, "default_output"):
        return res.default_output()
    else:
        return res


apply.add((Op, ExpressionTuple), apply_Op_ExpressionTuple)


def eval_and_reify(x):
    r"""Get Aesara objects from combinations of `etuple`\s."""
    res = x

    # Create base objects from the resulting meta object
    if isinstance(res, ExpressionTuple):
        res = res.evaled_obj

    return res


# def unify_Variable(u, v, s):
#     if type(u) != type(v):
#         return False
#
#     if getattr(u, "__all_props__", False):
#         s = unify(
#             [getattr(u, slot) for slot in u.__all_props__],
#             [getattr(v, slot) for slot in v.__all_props__],
#             s,
#         )
#     elif u != v:
#         return False
#     if s:
#         # If these two meta objects unified, and one has a logic
#         # variable as its base object, consider the unknown base
#         # object unified by the other's base object (if any).
#         # This way, the original base objects can be recovered during
#         # reification (preserving base object equality and such).
#         if isinstance(u, Var) and v:
#             s[u] = v
#         elif isinstance(v, Var) and u:
#             s[v] = u
#     return s
#
#
# _unify.add((Variable, Variable, Mapping), unify_Variable)
#
# _reify.add((Variable, Mapping), _reify_Variable)


class KanrenRelationSub(LocalOptimizer):
    r"""A local optimizer that uses miniKanren goals to match and replace terms in a Aesara `FunctionGraph`.

    TODO: Only uses *one* miniKanren `run` result (chosen by a configurable
    filter function).  We might want an option to produce multiple graphs, but
    I imagine that would involve an entirely different optimizer type.

    """

    reentrant = True

    def __init__(
        self,
        kanren_relation,
        relation_lvars=None,
        results_filter=lambda x: next(x, None),
        node_filter=lambda x: False,
    ):
        r"""Create a `KanrenRelationSub`.

        Parameters
        ----------
        kanren_relation: kanren.Relation or goal
            The miniKanren relation store or goal to use.  Custom goals should
            take an input and output argument, respectively.
        relation_lvars: Iterable
            A collection of terms to be considered logic variables by miniKanren
            (i.e. Aesara terms used as "unknowns" in `kanren_relation`).
        results_filter: function
            A function that returns a single result from a stream of
            miniKanren results.  The default function returns the first result.
        node_filter: function
            A function taking a single node as an argument that returns `True`
            when the node should be skipped.
        """
        self.kanren_relation = kanren_relation
        self.relation_lvars = relation_lvars or []
        self.results_filter = results_filter
        self.node_filter = node_filter
        super().__init__()

    def adjust_outputs(self, node, new_node, old_node=None):
        r"""Make adjustments for multiple outputs.

        This handles (some) nodes with multiple outputs by returning a list
        with the appropriate length and containing the new node (at the correct
        index if `default_output` is available and correct, or 0--and it
        happens to be the correct one).

        TODO: We should be able to get the correct index from the something
        like `node.outputs.index(old_node)`, but we don't exactly have
        `old_node` unless the miniKanren results give it to us.

        """
        res = list(node.outputs)
        try:
            new_node_idx = res.index(old_node)
        except ValueError:
            # Guesstimate it
            new_node_idx = getattr(node.op, "default_output", 0) or 0

        res[new_node_idx] = new_node
        return res

    def transform(self, node):
        if not isinstance(node, Apply):
            return False

        if self.node_filter(node):
            return False

        try:
            input_expr = node.default_output()
        except AttributeError:
            input_expr = node.outputs

        with variables(*self.relation_lvars):
            q = var()
            kanren_results = run(None, q, self.kanren_relation(input_expr, q))

        chosen_res = self.results_filter(kanren_results)

        if chosen_res:
            if isinstance(chosen_res, ExpressionTuple):
                chosen_res = eval_and_reify(chosen_res)

            if isinstance(chosen_res, dict):
                chosen_res = list(chosen_res.items())

            if isinstance(chosen_res, list):
                # We got a dictionary of replacements
                new_node = {eval_and_reify(k): eval_and_reify(v) for k, v in chosen_res}

                assert all(k in node.fgraph.variables for k in new_node.keys())
            elif isinstance(chosen_res, Variable):
                # Attempt to automatically format the output for multi-output
                # `Apply` nodes.
                new_node = self.adjust_outputs(node, eval_and_reify(chosen_res))
            else:
                raise ValueError(
                    "Unsupported FunctionGraph replacement variable type: {chosen_res}"
                )  # pragma: no cover

            return new_node
        else:
            return False
