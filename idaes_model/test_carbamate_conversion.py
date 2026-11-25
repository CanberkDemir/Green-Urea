"""Regression checks for trace-reactant carbamate condenser feeds."""
import unittest

from pyomo.environ import Block, ConcreteModel, Var, value, units as u
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock

from reactions.urea_reactions import add_carbeq_conversion


class CarbamateConversionTests(unittest.TestCase):
    def test_extent_is_nonnegative_and_limited_by_both_reactants(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.reactor = Block()
        reactor = m.fs.reactor
        reactor.control_volume = cv = Block()
        cv.properties_in = Block(m.fs.time)
        state = cv.properties_in[0]
        state.flow_mol = Var(initialize=1, units=u.mol / u.s)
        state.mole_frac_comp = Var(["NH3", "CO2"], initialize=0.1)
        cv.rate_reaction_extent = Var(m.fs.time, ["CARBF"], initialize=0,
                                     units=u.mol / u.s)
        # The production helper only needs the unit's flowsheet time.
        object.__setattr__(reactor, "flowsheet", lambda: m.fs)
        add_carbeq_conversion(reactor)
        assert_units_consistent(m)
        for co2, nh3 in [(0, 0), (0, 0.2), (1e-12, 2e-12),
                         (0.1, 1e-12), (0.1, 0.4), (0.3, 0.2)]:
            with self.subTest(co2=co2, nh3=nh3):
                state.mole_frac_comp["CO2"].set_value(co2)
                state.mole_frac_comp["NH3"].set_value(nh3)
                extent = value(reactor.carbeq_conv.expr.arg(1))
                self.assertGreaterEqual(extent, 0)
                self.assertLessEqual(extent, 0.99 * min(co2, nh3 / 2) + 1e-16)


if __name__ == "__main__":
    unittest.main()
