import unittest
import numpy as np
from sympy.physics.wigner import clebsch_gordan as sympy_cg
from sympy import Rational
from subhkl.commands import compute_su2_clebsch_gordan

class TestSymPyWignerConventions(unittest.TestCase):
    def test_spin_half_singlet_coupling(self):
        """ Verifies j1=1/2, j2=1/2 coupling to the l=0 scalar singlet. """
        cg_mandi = compute_su2_clebsch_gordan(0.5, 0.5, 0.0)
        self.assertEqual(cg_mandi.shape, (2, 2, 1))
        
        # Anti-symmetric singlet states: |1/2, -1/2> - |-1/2, 1/2> / sqrt(2)
        expected_up_down = float(sympy_cg(Rational(1,2), Rational(1,2), 0, Rational(1,2), Rational(-1,2), 0).evalf())
        expected_down_up = float(sympy_cg(Rational(1,2), Rational(1,2), 0, Rational(-1,2), Rational(1,2), 0).evalf())
        
        self.assertAlmostEqual(cg_mandi[1, 0, 0], expected_up_down, places=6)
        self.assertAlmostEqual(cg_mandi[0, 1, 0], expected_down_up, places=6)

    def test_spin_half_triplet_coupling(self):
        """ Verifies j1=1/2, j2=1/2 coupling to the l=1 vector triplet (m3=0 channel). """
        cg_mandi = compute_su2_clebsch_gordan(0.5, 0.5, 1.0)
        self.assertEqual(cg_mandi.shape, (2, 2, 3))
        
        # Symmetric triplet channel: m3 = 0 -> index 1
        expected_up_down = float(sympy_cg(Rational(1,2), Rational(1,2), 1, Rational(1,2), Rational(-1,2), 0).evalf())
        self.assertAlmostEqual(cg_mandi[1, 0, 1], expected_up_down, places=6)

if __name__ == '__main__':
    unittest.main()
