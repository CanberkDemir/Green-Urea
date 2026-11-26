"""Conservation and physical-limit checks for the separate H2 transport model."""

import unittest
from dataclasses import replace

from idaes_model.h2_transfer_reactor import Inputs, MW_H2, MW_NO3, h2_solubility_mol_m3, run


class H2TransferTests(unittest.TestCase):
    def test_henry_pressure_temperature_and_salt(self):
        pure = h2_solubility_mol_m3(1, 298.15, 0, 0.1)
        self.assertAlmostEqual(pure, 0.78)
        self.assertAlmostEqual(h2_solubility_mol_m3(27, 298.15, 0, 0.1), 27 * pure)
        self.assertLess(h2_solubility_mol_m3(27, 333.15, 2, 0.1),
                        h2_solubility_mol_m3(27, 298.15, 0, 0.1))

    def test_transfer_only_conserves_h2_and_saturates(self):
        p = Inputs()
        r = run(p)
        feed = p.h2_kgph / MW_H2 / 3600
        dissolved = r.h2_liquid_out_mol_m3 * r.liquid_m3_s
        self.assertAlmostEqual(feed, r.h2_gas_out_mol_s + dissolved, places=8)
        self.assertAlmostEqual(r.h2_liquid_out_mol_m3, r.c_star_in_mol_m3, places=4)
        self.assertEqual(r.h2_reacted_mol_s, 0)

    def test_no_transfer_no_reaction(self):
        p = Inputs(kla_s=0, k_no3_ref=0.01, k_no2_ref=0.01)
        r = run(p)
        self.assertEqual(r.h2_transferred_mol_s, 0)
        self.assertEqual(r.h2_reacted_mol_s, 0)
        self.assertEqual(r.nh3_out_mol_m3, 0)

    def test_reaction_conserves_h2_and_nitrogen(self):
        p = Inputs(k_no3_ref=0.01, k_no2_ref=0.01)
        r = run(p)
        h2_in = p.h2_kgph / MW_H2 / 3600
        h2_out = r.h2_gas_out_mol_s + r.h2_liquid_out_mol_m3 * r.liquid_m3_s
        self.assertAlmostEqual(h2_in, h2_out + r.h2_reacted_mol_s, places=7)
        n_in = p.nitrate_kgph / MW_NO3 / 3600 / r.liquid_m3_s
        n_out = r.nitrate_out_mol_m3 + r.nitrite_out_mol_m3 + r.nh3_out_mol_m3 + 2*r.n2_out_mol_m3
        self.assertAlmostEqual(n_in, n_out, places=7)
        self.assertGreater(r.nh3_out_mol_m3, 0)
        self.assertGreater(r.h2_gas_out_mol_s, 0)

    def test_transfer_coefficient_matters_for_reaction(self):
        p = Inputs(k_no3_ref=0.01, k_no2_ref=0.01)
        low = run(replace(p, kla_s=0.0002))
        high = run(replace(p, kla_s=0.01))
        self.assertGreater(high.h2_reacted_mol_s, low.h2_reacted_mol_s)

    def test_concentrated_brine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "single-phase brine"):
            run(Inputs(water_kgph=1400))


if __name__ == "__main__":
    unittest.main()
