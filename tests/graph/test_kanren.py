from etuples import etuple, etuplize
from kanren import eq
from kanren.core import lall
from unification import var

import aesara
import aesara.tensor as at
from aesara.graph.basic import graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.graph.kanren import KanrenRelationSub
from aesara.graph.opt import EquilibriumOptimizer, optimize_graph


@aesara.change_flags(compute_test_value="ignore", cxx="", mode="FAST_COMPILE")
def test_kanren_opt():
    """Make sure we can run miniKanren "optimizations" over a graph until a fixed-point/normal-form is reached."""
    x_tt = at.vector("x")
    c_tt = at.vector("c")
    d_tt = at.vector("c")
    A_tt = at.matrix("A")
    B_tt = at.matrix("B")

    Z_tt = A_tt.dot(x_tt + B_tt.dot(c_tt + d_tt))

    fgraph = FunctionGraph(graph_inputs([Z_tt]), [Z_tt], clone=True)

    assert isinstance(fgraph.outputs[0].owner.op, at.Dot)

    def distributes(in_lv, out_lv):
        return lall(
            # lhs == A * (x + b)
            eq(
                etuple(at.dot, var("A"), etuple(at.add, var("x"), var("b"))),
                etuplize(in_lv),
            ),
            # rhs == A * x + A * b
            eq(
                etuple(
                    at.add,
                    etuple(at.dot, var("A"), var("x")),
                    etuple(at.dot, var("A"), var("b")),
                ),
                out_lv,
            ),
        )

    distribute_opt = EquilibriumOptimizer(
        [KanrenRelationSub(distributes)], max_use_ratio=10
    )

    fgraph_opt = optimize_graph(fgraph, distribute_opt, return_graph=False)

    assert fgraph_opt.owner.op == at.add
    assert isinstance(fgraph_opt.owner.inputs[0].owner.op, at.Dot)
    # TODO: Something wrong with `etuple` caching?
    # assert fgraph_opt.owner.inputs[0].owner.inputs[0] == A_tt
    assert fgraph_opt.owner.inputs[0].owner.inputs[0].name == "A"
    assert fgraph_opt.owner.inputs[1].owner.op == at.add
    assert isinstance(fgraph_opt.owner.inputs[1].owner.inputs[0].owner.op, at.Dot)
    assert isinstance(fgraph_opt.owner.inputs[1].owner.inputs[1].owner.op, at.Dot)
