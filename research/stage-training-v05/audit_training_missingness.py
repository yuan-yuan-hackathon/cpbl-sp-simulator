"""Sensitivity audit with fixed hyperparameters; not another test-driven search."""
from train_stage_models import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    d=pd.read_csv(a.data/'contact_training_rows.csv.gz');d=d[d.strict_contact_training_eligible.eq(True)&d.pa_event_type_label.isin(OUT_EVENTS|SAFE_EVENTS)].copy()
    y=d.pa_event_type_label.isin(OUT_EVENTS).astype(int).to_numpy();tr,va,te=masks(d)
    complete=d[LANDING].notna().all(axis=1).to_numpy()
    missing_report={s:{'all_n':int(mask.sum()),'complete_n':int((mask&complete).sum()),'out_rate_complete':float(y[mask&complete].mean()),'out_rate_missing':float(y[mask&~complete].mean())} for s,mask in zip(['train','validation','test'],[tr,va,te])}
    records={}
    for name,cols in [('contact_only',CONTACT),('complete_landing',LANDING)]:
        m=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,learning_rate=.05,l2_regularization=5,min_samples_leaf=40,early_stopping=False,random_state=SEED)
        m.fit(d[cols].to_numpy(float)[tr&complete],y[tr&complete]);pred=m.predict_proba(d[cols].to_numpy(float)[te&complete])[:,1]
        records[name]=binary_metrics(y[te&complete],pred)
        joblib.dump({'model':m,'features':cols,'scope':'complete-case sensitivity only'},a.output/(name+'_sensitivity.joblib'))
    # Missingness-only shortcut: constant by availability pattern trained on train only.
    rates={v:float((y[tr&(complete==v)].sum()+20*y[tr].mean())/((tr&(complete==v)).sum()+20)) for v in [False,True]}
    pm=np.array([rates[bool(v)] for v in complete[te]])
    report={'missingness_by_split':missing_report,'missingness_only_test':binary_metrics(y[te],pm),'fixed_hyperparameter_complete_case_test':records,'interpretation':'Complete-case cohort is selective. Outcome-dependent recording/missingness can inflate all-row accuracy; no causal or production approval implied.'}
    dump(a.output/'missingness_sensitivity.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

