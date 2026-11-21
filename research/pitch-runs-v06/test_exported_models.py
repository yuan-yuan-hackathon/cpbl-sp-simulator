from pathlib import Path
import json,unittest,joblib,numpy as np
ROOT=Path('results/v06')
def infer(m,x):
 scores=np.array(m['baseline'],float)
 for iteration in m['trees']:
  for k,tree in enumerate(iteration):
   i=0
   while not tree[i][3]:
    f,t,missing,_,_,left,right=tree[i];v=x[f];i=left if (bool(missing) if np.isnan(v) else v<=float(t)) else right
   scores[k]+=tree[i][4]
 mass=np.exp(scores-scores.max());return mass/mass.sum()
class Parity(unittest.TestCase):
 def test_all_classes_and_missing_branches(self):
  maximum=0.
  for group in ['baseline','pitch']:
   m=json.loads((ROOT/(group+'.json')).read_text());bundle=joblib.load(ROOT/(group+'.joblib'));rng=np.random.default_rng(184);x=rng.normal(0,1,(20,len(m['features'])));x[0,:]=np.nan;x[1,::2]=np.nan
   expected=bundle['model'].predict_proba(x);actual=np.array([infer(m,row) for row in x]);err=np.max(np.abs(expected-actual));self.assertLess(err,1e-12);maximum=max(maximum,float(err))
  (ROOT/'export-parity.json').write_text(json.dumps({'models':2,'rows_per_model':20,'classes':10,'includes_all_missing_and_partial_missing':True,'maximum_probability_error':maximum,'status':'passed'},indent=2))
if __name__=='__main__':unittest.main()
