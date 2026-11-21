"""v0.6 verification: per-pitch response -> count recursion -> PA branches -> runs.
Train <= Jun30; validation Jul1-Aug22; reused development test >= Aug23.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,hashlib,platform,argparse
import numpy as np,pandas as pd,sklearn,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
from threadpoolctl import threadpool_limits
threadpool_limits(4)
CLASSES=['BALL','STRIKE','FOUL','HBP','CONTACT_OUT','1B','2B','3B','HR','OTHER']
BRANCH=['K','CONTACT_OUT','WALK','1B','2B','3B','HR','OTHER']
MAP={'SO':'K','SO_REACHED':'K','GO':'CONTACT_OUT','FO':'CONTACT_OUT','LO':'CONTACT_OUT','PO':'CONTACT_OUT','AO':'CONTACT_OUT','GIDP':'CONTACT_OUT','DP':'CONTACT_OUT','SH':'CONTACT_OUT','SF':'CONTACT_OUT','uBB':'WALK','IBB':'WALK','HBP':'WALK','1B':'1B','IH':'1B','2B':'2B','3B':'3B','HR':'HR','IHR':'HR','E':'OTHER','FC':'OTHER','SH_E':'OTHER','SF_E':'OTHER'}
SEED=184
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def absorb(q,future,b,s,solve_loop=False):
 eye=np.eye(8);r=q[...,0,None]*(future[...,b+1,s,:] if b<3 else eye[2])+q[...,1,None]*(future[...,b,s+1,:] if s<2 else eye[0])+q[...,3,None]*eye[2]
 if s<2:r+=q[...,2,None]*future[...,b,s+1,:]
 elif not solve_loop:r+=q[...,2,None]*future[...,b,s,:]
 for i,k in enumerate(CLASSES[4:],4):r+=q[...,i,None]*eye[BRANCH.index(k)]
 if solve_loop and s==2:r/=(1-q[...,2,None])
 return r
def count_branches(q):
 t=np.zeros(q.shape[:-3]+(4,3,8))
 for b in range(3,-1,-1):
  for s in range(2,-1,-1):t[...,b,s,:]=absorb(q[...,b,s,:],t,b,s,True)
 return t
def metrics(y,p):
 p=np.clip(p,1e-12,1);p/=p.sum(1,keepdims=True);rows=[];confidence=p.max(1);correct=(p.argmax(1)==y)
 for i in range(10):
  m=(confidence>=i/10)&(confidence<((i+1)/10) if i<9 else confidence<=1)
  if m.any():rows.append({'n':int(m.sum()),'predicted_confidence':float(confidence[m].mean()),'actual_accuracy':float(correct[m].mean())})
 per_class=[]
 for i,k in enumerate(CLASSES):
  bins=[]
  for j in range(10):
   m=(p[:,i]>=j/10)&(p[:,i]<((j+1)/10) if j<9 else p[:,i]<=1)
   if m.any():bins.append({'n':int(m.sum()),'predicted':float(p[m,i].mean()),'actual':float((y[m]==i).mean())})
  per_class.append({'class':k,'mean_prediction':float(p[:,i].mean()),'actual_rate':float((y==i).mean()),'bins':bins})
 return {'n':len(y),'log_loss':float(log_loss(y,p,labels=np.arange(10))),'multiclass_brier_sum':float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))),'top_label_calibration':rows,'class_calibration':per_class}
def cluster_ci(delta,games):
 agg=pd.DataFrame({'game':games,'delta':delta}).groupby('game').delta.agg(['sum','count']).to_numpy();rng=np.random.default_rng(SEED);vals=[]
 for _ in range(2000):
  z=agg[rng.integers(0,len(agg),len(agg))];vals.append(z[:,0].sum()/z[:,1].sum())
 return {'difference':float(np.mean(delta)),'ci95':np.quantile(vals,[.025,.975]).tolist(),'games':len(agg),'replicates':2000,'unit':'game cluster, event-weighted'}
def export(m,features):
 trees=[]
 for iteration in m._predictors:
  trees.append([[[int(n['feature_idx']),float(n['num_threshold']) if np.isfinite(n['num_threshold']) else 'Infinity',int(n['missing_go_to_left']),int(n['is_leaf']),float(n['value']),int(n['left']),int(n['right'])] for n in t.nodes] for t in iteration])
 return {'format':'multiclass-hgb-v1','classes':CLASSES,'features':features,'baseline':m._baseline_prediction.ravel().tolist(),'trees':trees}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('data/CPBL_2026_predict_data/data'));ap.add_argument('--output',type=Path,default=Path('results/v06'));a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(a.data/'pitch_training_rows.csv.gz',dtype={'batter_id':str,'pitcher_id':str});d=raw[raw.strict_pitch_training_eligible.eq(True)&raw.pa_re_training_eligible.eq(1)].copy()
 same=d.pa_outs_before_audited.eq(d.outs_before_source)&d.pa_bases_before_audited.eq(d.bases_before_source);d=d[same].copy();mapped=d.pa_event_type_label.map(MAP);label=pd.Series(None,index=d.index,dtype=object)
 label[d.pitch_call_label.isin(['BallCalled','BallinDirt','BallIntentional'])]='BALL';label[d.pitch_call_label.isin(['StrikeCalled','StrikeSwinging'])]='STRIKE';label[d.pitch_call_label.eq('FoulBallNotFieldable')]='FOUL';label[d.pitch_call_label.eq('HitByPitch')]='HBP'
 terminal=d.source_log_index.eq(d.pa_terminal_source_index);inplay=d.pitch_call_label.eq('InPlay')&terminal&mapped.isin(CLASSES[4:]);label[inplay]=mapped[inplay];d['target']=label;d=d[d.target.notna()&d.balls_before.between(0,3)&d.strikes_before.between(0,2)].copy();assert d.pitch_id.is_unique
 expected=np.where(d.day<'2026-07-01','train',np.where(d.day<'2026-08-23','validation','test'));assert (d.split.to_numpy()==expected).all()
 masks=[d.split.eq(k).to_numpy() for k in ['train','validation','test']];tr,va,te=masks;sets=[set(d.loc[m,'game_id']) for m in masks];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
 c=pd.read_csv(a.data/'contact_training_rows.csv.gz',dtype={'batter_id':str});c=c[c.strict_contact_training_eligible.eq(True)&c.split.eq('train')];cc=['exit_speed_kph_label','launch_angle_deg_label','launch_direction_deg_label'];prior=c[cc].mean().to_numpy();bs={};bn={};ps={};pn={};h={}
 global_pitch=np.bincount([CLASSES.index(k) for k in d.loc[tr,'target']],minlength=10).astype(float);global_pitch/=global_pitch.sum()
 for day,g in d.groupby('day',sort=True):
  for idx,row in g.iterrows():
   nb=bn.get(row.batter_id,0);npit=pn.get(row.pitcher_id,0);h[idx]=list((bs.get(row.batter_id,np.zeros(3))+30*prior)/(nb+30))+[nb]+list((ps.get(row.pitcher_id,np.zeros(10))+100*global_pitch)/(npit+100))
  if day<'2026-07-01':
   for key,gg in c[c.day.eq(day)].groupby('batter_id'):bs[key]=bs.get(key,np.zeros(3))+gg[cc].sum().to_numpy();bn[key]=bn.get(key,0)+len(gg)
   for key,gg in g.groupby('pitcher_id'):ps[key]=ps.get(key,np.zeros(10))+np.bincount([CLASSES.index(k) for k in gg.target],minlength=10);pn[key]=pn.get(key,0)+len(gg)
 hist=['history_exit','history_angle','history_direction','history_n']+['pitcher_rate_'+k for k in CLASSES]
 for j,k in enumerate(hist):d[k]=[h[idx][j] for idx in d.index]
 base=['pa_outs_before_audited','pa_bases_before_audited','balls_before','strikes_before']+hist;physical=['plate_x_m','plate_z_m','velocity_kph','spin_rpm','release_height_m','release_side_m','extension_m'];full=base+physical
 assert not set(base+full)&{'pitch_call_label','target','pa_event_type_label','exit_speed_kph_label','launch_angle_deg_label','landing_distance_m_label','balls_after_raw','strikes_after_raw','pa_terminal_source_index'}
 y=np.array([CLASSES.index(k) for k in d.target]);models={};choices={};candidates={};testp={}
 for group,cols in [('baseline',base),('pitch',full)]:
  candidates[group]={}
  for leaves in [7,15]:
   m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=leaves,learning_rate=.05,l2_regularization=8,min_samples_leaf=60,early_stopping=False,random_state=SEED);m.fit(d[cols].to_numpy(float)[tr],y[tr]);assert np.array_equal(m.classes_,np.arange(10))
   score=metrics(y[va],m.predict_proba(d[cols].to_numpy(float)[va]));candidates[group][str(leaves)]={'validation_log_loss':score['log_loss'],'validation_brier':score['multiclass_brier_sum']}
   if group not in models or score['log_loss']<choices[group]['validation_log_loss']:models[group]=m;choices[group]={'leaves':leaves,'validation_log_loss':score['log_loss'],'features':cols}
  testp[group]=models[group].predict_proba(d[cols].to_numpy(float)[te]);joblib.dump({'model':models[group],'features':cols},out/(group+'.joblib'));dump(out/(group+'.json'),export(models[group],cols));print('FIT '+group+' '+str(choices[group]),flush=True)
 profile={'batter_prior':prior.tolist()+[0],'pitcher_prior':global_pitch.tolist(),'batters':{str(int(k)):list((v+30*prior)/(bn[k]+30))+[bn[k]] for k,v in bs.items()},'pitchers':{str(int(k)):list((v+100*global_pitch)/(pn[k]+100)) for k,v in ps.items()},'cutoff':'2026-06-30'};dump(out/'profiles.json',profile)
 report={'version':'v06-refit-linux','python':platform.python_version(),'sklearn':sklearn.__version__,'source_sha256':hashlib.sha256((a.data/'pitch_training_rows.csv.gz').read_bytes()).hexdigest(),'classes':CLASSES,'choices':choices,'validation_candidates':candidates,'splits':{k:{'n':int(m.sum()),'games':int(d.loc[m,'game_id'].nunique()),'start':str(d.loc[m,'day'].min()),'end':str(d.loc[m,'day'].max())} for k,m in zip(['train','validation','test'],masks)},'excluded_from_raw':len(raw)-len(d),'test':{g:metrics(y[te],p) for g,p in testp.items()}}
 ll=lambda p:-np.log(np.clip(p[np.arange(te.sum()),y[te]],1e-12,1));report['pitch_probability_difference']=cluster_ci(ll(testp['pitch'])-ll(testp['baseline']),d.loc[te,'game_id'].to_numpy());dump(out/'checkpoint-probability.json',report)
 ev=pd.read_csv(a.data/'audited_events_snapshot.csv.gz');ev=ev[ev.event_scope.eq('PA')&ev.RE_training_eligible.eq(1)].copy();ev['state']=ev.outs_before*8+ev.bases_before;ev['branch']=ev.event_type.map(MAP);ev=ev[ev.branch.notna()].copy();ev['after_state']=np.where(ev.outs_after>=3,24,ev.outs_after*8+ev.bases_after);train_ev=ev[ev.date.str[:10]<'2026-07-01'].copy()
 re24=train_ev.groupby('state').runs_remaining_from_event_start.mean().reindex(range(24)).fillna(train_ev.runs_remaining_from_event_start.mean()).to_numpy();re25=np.r_[re24,0];train_ev['value']=train_ev.runs_scored+re25[train_ev.after_state.astype(int)];global_value=train_ev.groupby('branch').value.mean();value=np.zeros((24,8))
 for state in range(24):
  for k,branch in enumerate(BRANCH):
   rows=train_ev[(train_ev.state==state)&(train_ev.branch==branch)];value[state,k]=rows.value.mean() if len(rows) else global_value[branch]
 actual=ev.set_index('play_id');assert actual.index.is_unique
 test=d[te&d.balls_before.eq(0).to_numpy()&d.strikes_before.eq(0).to_numpy()].drop_duplicates('record_id').copy();ids='cpbl:'+test.game_id+':log:'+test.pa_terminal_source_index.astype(int).astype(str);test=test[ids.isin(actual.index)].copy();ids='cpbl:'+test.game_id+':log:'+test.pa_terminal_source_index.astype(int).astype(str)
 bx=np.repeat(test[base].to_numpy(float),12,axis=0);bx[:,base.index('balls_before')]=np.tile(np.repeat(np.arange(4),3),len(test));bx[:,base.index('strikes_before')]=np.tile(np.tile(np.arange(3),4),len(test));qb=models['baseline'].predict_proba(bx).reshape(len(test),4,3,10);future=count_branches(qb);qp=models['pitch'].predict_proba(test[full].to_numpy(float));before=future[:,0,0,:];after=absorb(qp,future,0,0)
 assert np.max(np.abs(future.sum(-1)-1))<1e-8;assert np.max(np.abs(after.sum(-1)-1))<1e-8
 state=(test.pa_outs_before_audited*8+test.pa_bases_before_audited).astype(int).to_numpy();v=value[state];e0=np.sum(before*v,axis=1);e1=np.sum(after*v,axis=1);target=actual.loc[ids,'runs_remaining_from_event_start'].to_numpy(float);games=test.game_id.to_numpy();rs={}
 for name,pred in [('RE24',re24[state]),('count_baseline',e0),('observed_pitch',e1)]:rs[name]={'n':len(target),'rmse':float(np.sqrt(mean_squared_error(target,pred))),'mae':float(mean_absolute_error(target,pred))}
 report['run_value_test']={'target':'actual remaining half-inning runs at tracked eligible 0-0 PA starts','period':'2026-08-23 to 2026-10-06','metrics':rs,'pitch_vs_count_mse':cluster_ci((e1-target)**2-(e0-target)**2,games),'pitch_vs_RE24_mse':cluster_ci((e1-target)**2-(re24[state]-target)**2,games),'mean_pitcher_runs':float((e0-e1).mean()),'value_tables':'TRAIN only; league continuation, no next-batter model in this evaluation'}
 first=test.iloc[0];x={k:(None if pd.isna(first[k]) else float(first[k])) for k in full};fixture={'input':x,'baseline':models['baseline'].predict_proba(pd.DataFrame([first])[base].to_numpy(float))[0].tolist(),'pitch':qp[0].tolist(),'branch_before':before[0].tolist(),'branch_after':after[0].tolist()};dump(out/'parity-fixture.json',fixture)
 table=pd.DataFrame({'pitch_id':test.pitch_id,'game_id':games,'day':test.day,'actual_remaining_runs':target,'E0':e0,'E1':e1,'pitcher_runs':e0-e1});table.to_csv(out/'run_predictions.csv.gz',index=False)
 examples=table.loc[(table.E1-table.E0).abs().nlargest(8).index].to_dict('records');report['largest_observed_shifts']=examples
 report['limitations']=['Refit in a new environment; previous Windows fitted weights unavailable, not claiming exact previous-weight reproduction.','Reused development test; scheduled date used, resumed-game chronology unaudited.','PA-start context kept only when matching observable source state; runners are not independently attributed.','Future pitches use count/context probabilities; no full physical pitch-sequence model.','Information-stage attribution is predictive, not causal pitcher responsibility.','Run-value test uses TRAIN league continuation; live next-batter extension is not covered by these scores.','No held-out recalibration; missingness and TrackMan coverage nonrandom.','Branch values pool multiple advancement outcomes and can be sparse.']
 report['checks']={'unique_pitch_ids':True,'disjoint_games':True,'forbidden_features_excluded':True,'max_abs_absorption_mass_error':float(np.max(np.abs(future.sum(-1)-1))),'all_run_predictions_finite':bool(np.isfinite(e0).all() and np.isfinite(e1).all())}
 dump(out/'metrics.json',report);dump(out/'value-tables.json',{'branches':BRANCH,'values':value.tolist(),'RE24':re24.tolist()});print('COMPLETE '+json.dumps({'probability':{g:{k:v for k,v in report['test'][g].items() if k in ['n','log_loss','multiclass_brier_sum']} for g in testp},'runs':report['run_value_test']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
