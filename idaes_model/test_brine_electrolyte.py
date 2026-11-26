"""A5 Pitzer model against published activity data; HCl/NaOH dosing."""

import math
import unittest

from idaes_model import brine_electrolyte as BE
from idaes_model.ni2p_pfr import Inputs, run


def g_pm(m, c="Na+", a="Cl-"):
    g = BE.pitzer({c: m, a: m}).gamma
    return math.sqrt(g[c] * g[a])


class PitzerTests(unittest.TestCase):
    def test_nacl_literature(self):
        # Robinson & Stokes / Pitzer & Mayorga, 25 C
        for m, phi, gam in ((0.1, 0.932, 0.778), (1.0, 0.936, 0.657),
                            (3.0, 1.045, 0.714), (6.0, 1.271, 0.986)):
            a = BE.pitzer({"Na+": m, "Cl-": m})
            self.assertAlmostEqual(a.osmotic_coefficient, phi, delta=0.005)
            self.assertAlmostEqual(g_pm(m), gam, delta=0.005)

    def test_other_salts(self):
        self.assertAlmostEqual(g_pm(1.0, "H+"), 0.809, delta=0.01)
        self.assertAlmostEqual(g_pm(2.0, "NH4+"), 0.575, delta=0.01)

    def test_saturated_nacl_bpe(self):
        bpe = BE.boiling_point_elevation_k({"Na+": 6.6, "Cl-": 6.6}, 1.01325)
        self.assertGreater(bpe, 8.0)
        self.assertLess(bpe, 9.5)

    def test_dosing_signs(self):
        w, n = BE.kgph_to_mol(BE.FEED_KGPH)
        self.assertAlmostEqual(BE.acid_base_dose(w, n, 7.0, 333.15), 0.0, delta=0.5)
        self.assertGreater(BE.acid_base_dose(w, n, 2.0, 333.15), 0)
        self.assertLess(BE.acid_base_dose(w, n, 11.0, 333.15), 0)

    def test_pfr_reports_hcl(self):
        r = run(Inputs())
        self.assertGreater(r.hcl_ph_set_kgph, 0)
        self.assertAlmostEqual(r.hcl_reaction_kgph,
                               r.acid_equivalents_mol_s * 3600 * 0.036461, places=6)


if __name__ == "__main__":
    unittest.main()
