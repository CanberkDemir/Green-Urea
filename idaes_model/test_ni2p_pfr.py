"""Conservation and transfer limits for the provisional Ni2P PFR."""

import unittest
from dataclasses import replace

from idaes_model.ni2p_pfr import Inputs, run


class Ni2PPFRTests(unittest.TestCase):
    def test_reference_bed_inventory_and_balances(self):
        r = run(Inputs())
        self.assertAlmostEqual(r.supported_catalyst_kg, 4712.38898, places=3)
        self.assertAlmostEqual(r.active_ni2p_kg, r.supported_catalyst_kg * 0.1239)
        self.assertLess(abs(r.h2_balance_error_mol_s), 1e-10)
        self.assertLess(abs(r.nitrogen_balance_error_mol_s), 1e-10)
        self.assertGreater(r.nitrate_conversion, 0)

    def test_no_hydrogen_transfer_means_no_reaction(self):
        r = run(Inputs(kla_s=0))
        self.assertEqual(r.nitrate_conversion, 0)
        self.assertEqual(r.h2_consumed_kgph, 0)

    def test_transfer_and_activity_sensitivity(self):
        p = Inputs()
        slow = run(replace(p, kla_s=0.001))
        fast = run(replace(p, kla_s=0.01))
        inactive = run(replace(p, activity_factor=0))
        self.assertGreater(fast.nitrate_conversion, slow.nitrate_conversion)
        self.assertEqual(inactive.nitrate_conversion, 0)

    def test_hydrogen_starvation_and_selectivity(self):
        p = Inputs(h2_kgph=0.05, kla_s=0.1, ksa_s=0.1)
        r = run(p)
        self.assertGreaterEqual(r.h2_gas_out_kgph, 0)
        self.assertLessEqual(r.h2_consumed_kgph, p.h2_kgph + 1e-8)
        self.assertLess(abs(r.h2_balance_error_mol_s), 1e-10)
        self.assertLess(abs(r.nitrogen_balance_error_mol_s), 1e-10)


if __name__ == "__main__":
    unittest.main()
