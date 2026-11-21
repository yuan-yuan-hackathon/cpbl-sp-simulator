import unittest
import numpy as np
from validate_pitch_runs import count_branches,absorb,BRANCH

class CountConversionTests(unittest.TestCase):
 def test_probability_mass_and_consistency(self):
  rng=np.random.default_rng(184);q=rng.dirichlet(np.ones(10),size=(40,4,3));future=count_branches(q)
  self.assertLess(np.max(np.abs(future.sum(-1)-1)),1e-12)
  for b in range(4):
   for s in range(3):self.assertLess(np.max(np.abs(absorb(q[:,b,s],future,b,s)-future[:,b,s])),1e-12)
 def test_strike_ball_foul_and_hit_boundaries(self):
  q=np.broadcast_to(np.array([.3,.3,.2,.01,.1,.04,.02,.005,.015,.01]),(4,3,10));future=count_branches(q);one=np.eye(10)
  np.testing.assert_array_equal(absorb(one[1],future,1,2),np.eye(8)[0])
  np.testing.assert_array_equal(absorb(one[0],future,3,1),np.eye(8)[2])
  np.testing.assert_array_equal(absorb(one[2],future,1,2),future[1,2])
  np.testing.assert_array_equal(absorb(one[8],future,0,0),np.eye(8)[6])
 def test_runs_direction_and_accounting(self):
  q=np.broadcast_to(np.array([.3,.3,.2,.01,.1,.04,.02,.005,.015,.01]),(4,3,10));future=count_branches(q);v=np.array([0,0,.4,1,1.5,1.8,2,1]);e0=future[0,0]@v;e1=absorb(np.eye(10)[8],future,0,0)@v
  self.assertLess(e0-e1,0)
  e2=1.2;value=0.;pitcher=e0-e1;batter=e2-e1;fielding=e2-value
  self.assertAlmostEqual(batter-pitcher-fielding,value-e0)

if __name__=='__main__':unittest.main()
