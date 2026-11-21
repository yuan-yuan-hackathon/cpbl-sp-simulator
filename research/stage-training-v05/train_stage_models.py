"""Offline research training. Never updates production or calls network services.
Targets and feature availability are separated; development test is reused.
"""
from pathlib import Path
import argparse, json, hashlib, warnings, os
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error, mean_squared_error
import joblib

SEED=184
OUT_EVENTS={'GO','FO','LO','PO','AO','SF','SH','DP','GIDP'}
SAFE_EVENTS={'1B','2B','3B','IH','E','FC'}
BASE=['pa_outs_before_audited','pa_bases_before_audited','balls_before','strikes_before','history_exit','history_angle','history_direction','history_n']
LOCATION=BASE+['plate_x_m','plate_z_m']
PITCH=LOCATION+['velocity_kph','spin_rpm','release_height_m','release_side_m','extension_m']
CONTACT=['pa_outs_before_audited','pa_bases_before_audited','exit_speed_kph_label','launch_angle_deg_label','launch_direction_deg_label']
LANDING=CONTACT+['landing_bearing_deg_label','landing_distance_m_label','hang_time_s_label']

def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def masks(d): return [d.split.eq(x).to_numpy() for x in ['train','validation','test']]

def calibration(y,p):
    rows=[]
    for i in range(10):
        m=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        if m.any(): rows.append({'bin':i,'n':int(m.sum()),'predicted':float(p[m].mean()),'observed':float(y[m].mean())})
    return rows

def binary_metrics(y,p):
    p=np.clip(p,1e-9,1-1e-9)
    rows=calibration(y,p)
    return {'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),
            'n':len(y),'mean_prediction':float(p.mean()),'observed_rate':float(y.mean()),
            'ece_10_equal_width':float(sum(r['n']*abs(r['predicted']-r['observed']) for r in rows)/len(y)),
            'calibration':rows}

def paired_ci(y,baseline,model,games):
    """Bootstrap entire games; per-event weighting rather than equal game means."""
    b=np.clip(baseline,1e-9,1-1e-9); p=np.clip(model,1e-9,1-1e-9)
    delta=-(y*np.log(p)+(1-y)*np.log(1-p))+(y*np.log(b)+(1-y)*np.log(1-b))
    agg=pd.DataFrame({'g':games,'delta':delta}).groupby('g').delta.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(2000):
        a=agg[rng.integers(0,len(agg),len(agg))]; vals.append(a[:,0].sum()/a[:,1].sum())
    return {'model_minus_baseline_log_loss':float(delta.mean()),'game_cluster_bootstrap_95ci':np.quantile(vals,[.025,.975]).tolist(),'replicates':2000,'games':len(agg)}

def history_features(d):
    """Training rows see earlier train days only; val/test use frozen June history."""
    targets=['exit_speed_kph_label','launch_angle_deg_label','launch_direction_deg_label']
    train=d[d.split.eq('train')].copy(); prior=train[targets].mean().to_numpy()
    sums={}; counts={}; features={}
    for day,g in train.groupby('day',sort=True):
        for idx,row in g.iterrows():
            key=row.batter_id; n=counts.get(key,0); total=sums.get(key,np.zeros(3))
            features[idx]=list((total+30*prior)/(n+30))+[n]
        for key,gg in g.groupby('batter_id'):
            sums[key]=sums.get(key,np.zeros(3))+gg[targets].sum().to_numpy(); counts[key]=counts.get(key,0)+len(gg)
    for idx,row in d[~d.split.eq('train')].iterrows():
        n=counts.get(row.batter_id,0); features[idx]=list((sums.get(row.batter_id,np.zeros(3))+30*prior)/(n+30))+[n]
    for j,c in enumerate(['history_exit','history_angle','history_direction','history_n']): d[c]=[features[i][j] for i in d.index]
    return {'shrinkage_pseudo_contacts':30,'global_means_train':prior.tolist(),'policy':'earlier train days for train; train-cutoff-frozen batter history for validation/test','caveat':'global shrinkage prior estimated on entire training block; no validation/test labels used'}

def fit_binary(name,d,y,groups,out):
    tr,va,te=masks(d); yt=y[tr]; yv=y[va]; ys=y[te]
    candidates={}; fitted={}
    for feature_name,cols in groups.items():
        x=d[cols].to_numpy(float)
        for kind in ['logistic','hgb7','hgb15']:
            if kind=='logistic':
                estimator=make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED))
            else:
                estimator=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=int(kind[3:]),learning_rate=.05,l2_regularization=5,min_samples_leaf=40,early_stopping=False,random_state=SEED)
            estimator.fit(x[tr],yt)
            pv=estimator.predict_proba(x[va])[:,1]
            key=feature_name+'_'+kind; candidates[key]={'features':cols,'validation':binary_metrics(yv,pv)}; fitted[key]=estimator
    best=min(candidates,key=lambda k:candidates[k]['validation']['log_loss'])
    baseline=float(yt.mean()); constant=np.full(te.sum(),baseline)
    report={'selected_by':'validation Log Loss only; one locked test evaluation after all choices','selected':best,'candidates':candidates,
            'constant_train_rate':baseline,'test_constant':binary_metrics(ys,constant),'test_models':{}}
    predictions=d.loc[te,['pitch_id','game_id','day']].copy(); predictions['target']=ys
    for key in [best]+[k for k in candidates if k.endswith('_logistic') and k!=best]:
        cols=candidates[key]['features']; pred=fitted[key].predict_proba(d[cols].to_numpy(float)[te])[:,1]
        report['test_models'][key]=binary_metrics(ys,pred); predictions[key]=pred
        if key==best:
            report['paired_constant']=paired_ci(ys,constant,pred,d.loc[te,'game_id'].to_numpy())
    # Compare selected full model with the strongest validation-selected model in baseline feature group.
    baseline_group=next(iter(groups)); bk=min((k for k in candidates if k.startswith(baseline_group+'_')),key=lambda k:candidates[k]['validation']['log_loss'])
    bp=fitted[bk].predict_proba(d[candidates[bk]['features']].to_numpy(float)[te])[:,1]
    pred=predictions[best].to_numpy(); report['baseline_feature_model']=bk; report['test_feature_baseline']=binary_metrics(ys,bp)
    report['paired_feature_baseline']=paired_ci(ys,bp,pred,d.loc[te,'game_id'].to_numpy())
    report['promotion_gate']=bool(report['paired_feature_baseline']['game_cluster_bootstrap_95ci'][1]<0 and report['test_models'][best]['brier']<report['test_feature_baseline']['brier'])
    joblib.dump({'model':fitted[best],'features':candidates[best]['features'],'task':name,'seed':SEED},out/(name+'.joblib'))
    predictions.to_csv(out/(name+'_development_test_predictions.csv.gz'),index=False)
    dump(out/(name+'_metrics.json'),report)
    return report

def fit_regression(d,out):
    tr,va,te=masks(d); results={}
    for target in ['exit_speed_kph_label','launch_angle_deg_label','launch_direction_deg_label']:
        y=d[target].to_numpy(float); choices={}; models={}
        for group,cols in [('baseline',BASE),('location',LOCATION),('full_pitch',PITCH)]:
            x=d[cols].to_numpy(float)
            for leaves in [7,15]:
                key=group+'_'+str(leaves); m=HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=leaves,learning_rate=.05,l2_regularization=5,min_samples_leaf=40,early_stopping=False,random_state=SEED)
                m.fit(x[tr],y[tr]); pv=m.predict(x[va]); choices[key]={'mae':float(mean_absolute_error(y[va],pv)),'features':cols}; models[key]=m
        key=min(choices,key=lambda k:choices[k]['mae']); cols=choices[key]['features']; pred=models[key].predict(d[cols].to_numpy(float)[te]); constant=np.full(te.sum(),y[tr].mean())
        results[target]={'selected':key,'validation_candidates':choices,'test_n':int(te.sum()),'test_mae':float(mean_absolute_error(y[te],pred)),'test_rmse':float(np.sqrt(mean_squared_error(y[te],pred))),'train_mean_baseline_mae':float(mean_absolute_error(y[te],constant)),'batter_history_baseline_mae':float(mean_absolute_error(y[te],d.loc[te,{'exit_speed_kph_label':'history_exit','launch_angle_deg_label':'history_angle','launch_direction_deg_label':'history_direction'}[target]]))}
        joblib.dump({'model':models[key],'features':cols,'target':target},out/(target+'_regressor.joblib'))
    dump(out/'contact_quality_regression_metrics.json',results);return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.data/'contact_training_rows.csv.gz',dtype={'batter_id':str,'pitcher_id':str}); raw_n=len(d)
    d=d[d.strict_contact_training_eligible.eq(True)].copy();assert d.pitch_id.is_unique
    expected=np.where(d.day<'2026-07-01','train',np.where(d.day<'2026-08-23','validation','test'));assert (d.split.to_numpy()==expected).all()
    sets=[set(d.loc[d.split.eq(s),'game_id']) for s in ['train','validation','test']];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    for c in set(BASE+PITCH+CONTACT+LANDING)-{'history_exit','history_angle','history_direction','history_n'}:d[c]=pd.to_numeric(d[c],errors='coerce')
    history=history_features(d)
    metadata={'version':'stage-training-v05-research','source_sha256':hashlib.sha256((args.data/'contact_training_rows.csv.gz').read_bytes()).hexdigest(),'rows_raw':raw_n,'rows_strict':len(d),'event_counts':d.pa_event_type_label.value_counts().to_dict(),'history':history,
      'splits':{s:{'n':int(d.split.eq(s).sum()),'games':int(d.loc[d.split.eq(s),'game_id'].nunique()),'start':str(d.loc[d.split.eq(s),'day'].min()),'end':str(d.loc[d.split.eq(s),'day'].max())} for s in ['train','validation','test']},
      'limitations':['Already-used development test, not a fresh held-out test.','PreExeDate scheduled date; suspended games actual chronology unaudited.','All pitch-to-contact models conditional on observed eligible fair contact; not unconditional pitch danger or pitcher causal responsibility.','No full run-value or player-specific fielder/baserunner model trained.','Missing TrackMan nonrandom; metrics only represent tracked eligible cohort.','Landing fields are post-contact; excluded from pitch models, may reflect outcome-dependent measurement.']}
    dump(out/'training_manifest.json',metadata);print(json.dumps({'cohort':metadata['splits'],'events':metadata['event_counts']},ensure_ascii=False),flush=True)
    quality=fit_binary('hard_contact_conditional',d,(d.exit_speed_kph_label.to_numpy()>=150).astype(int),{'baseline':BASE,'location':LOCATION,'full_pitch':PITCH},out)
    print('HARD_CONTACT_DONE '+json.dumps({'selected':quality['selected'],'test':quality['test_models'][quality['selected']],'promotion_gate':quality['promotion_gate']},ensure_ascii=False),flush=True)
    reg=fit_regression(d,out);print('REGRESSION_DONE',flush=True)
    keep=d.pa_event_type_label.isin(OUT_EVENTS|SAFE_EVENTS); f=d[keep].copy(); y=f.pa_event_type_label.isin(OUT_EVENTS).astype(int).to_numpy()
    defense=fit_binary('contact_out_branch',f,y,{'state_only':['pa_outs_before_audited','pa_bases_before_audited'],'contact':CONTACT,'contact_landing':LANDING},out)
    dump(out/'summary.json',{'manifest':metadata,'hard_contact':quality,'regression':reg,'defense':defense,'hard_contact_definition':'EV >= 150 km/h; operational threshold, not a universal baseball quality definition','defense_label':'out event branch, not verified individual batter-out or fielder responsibility','production_replacement':False})
    print('COMPLETE '+json.dumps({'hard_contact_best':quality['selected'],'defense_best':defense['selected'],'defense_test':defense['test_models'][defense['selected']],'production_replacement':False},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

