"""Second task: probability of a >=150km/h fair contact on a tracked pitch.
Noncontact pitches are negatives, not silently dropped. Missing contact labels excluded.
"""
from train_stage_models import *
def main():
    import re
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--site-source',type=Path,default=Path('index.html'));ap.add_argument('--reuse-fitted',action='store_true');args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.data/'pitch_training_rows.csv.gz',dtype={'batter_id':str,'pitcher_id':str})
    c=pd.read_csv(args.data/'contact_training_rows.csv.gz',dtype={'batter_id':str,'pitcher_id':str});c=c[c.strict_contact_training_eligible.eq(True)].copy()
    raw_n=len(d); valid_contact=set(c.pitch_id);d=d[d.strict_pitch_training_eligible.eq(True)&(~d.in_play_label.eq(True)|d.pitch_id.isin(valid_contact))].copy()
    assert d.pitch_id.is_unique
    ctrain=c[c.split.eq('train')];prior=ctrain[['exit_speed_kph_label','launch_angle_deg_label','launch_direction_deg_label']].mean().to_numpy();sums={};counts={};features={}
    for day,g in d.groupby('day',sort=True):
        for idx,row in g.iterrows():
            n=counts.get(row.batter_id,0);features[idx]=list((sums.get(row.batter_id,np.zeros(3))+30*prior)/(n+30))+[n]
        if day<'2026-07-01':
            for key,gg in ctrain[ctrain.day.eq(day)].groupby('batter_id'):
                sums[key]=sums.get(key,np.zeros(3))+gg[['exit_speed_kph_label','launch_angle_deg_label','launch_direction_deg_label']].sum().to_numpy();counts[key]=counts.get(key,0)+len(gg)
    for j,col in enumerate(['history_exit','history_angle','history_direction','history_n']):d[col]=[features[i][j] for i in d.index]
    ev=c.set_index('pitch_id').exit_speed_kph_label
    y=(d.pitch_id.map(ev).ge(150)&d.in_play_label.eq(True)).astype(int).to_numpy()
    tr,va,te=masks(d)
    report=json.loads((out/'hard_fair_contact_per_pitch_metrics.json').read_text(encoding='utf8')) if args.reuse_fitted else fit_binary('hard_fair_contact_per_pitch',d,y,{'baseline':BASE,'location':LOCATION,'full_pitch':PITCH},out)
    report['cohort']={s:{'n':int(d.split.eq(s).sum()),'games':int(d.loc[d.split.eq(s),'game_id'].nunique()),'start':d.loc[d.split.eq(s),'day'].min(),'end':d.loc[d.split.eq(s),'day'].max()} for s in ['train','validation','test']}
    report['raw_pitches']=raw_n;report['eligible_pitches']=len(d);report['definition']='P(fair terminal contact with EV>=150 km/h on this tracked eligible pitch). Not overall pitch run value; missing InPlay target rows excluded.'
    report['features_excluded']='No pitch_call, in_play, actual EV, actual launch angle, terminal action or result enters X.'
    dump(out/'hard_fair_contact_per_pitch_metrics.json',report)
    # Rebuild the old-style four-dimensional cell baseline using TRAIN ONLY.
    html=args.site_source.read_text(encoding='utf8');payload=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S).group(1));landing=payload['landing']
    f=c[c.pa_event_type_label.isin(OUT_EVENTS|SAFE_EVENTS)].copy();yf=f.pa_event_type_label.isin(OUT_EVENTS).astype(int).to_numpy();ft,fv,fs=masks(f)
    fields=['launch_angle_deg_label','exit_speed_kph_label','landing_distance_m_label','landing_bearing_deg_label'];edges=[landing[k] for k in ['angle_edges','speed_edges','distance_edges','bearing_edges']]
    keys=[]
    for vals in f[fields].to_numpy(float):
        key=[]
        for v,e in zip(vals,edges):
            i=int(np.searchsorted(e,v,side='right')-1)
            if v==e[-1]:i=len(e)-2
            key.append(i if np.isfinite(v) and 0<=i<len(e)-1 else -1)
        keys.append(tuple(key))
    base=float(yf[ft].mean());stats={}
    for key,yv in zip(np.array(keys,dtype=object)[ft].tolist(),yf[ft]):
        key=tuple(key)
        if -1 in key:continue
        a,b=stats.get(key,(0,0));stats[key]=(a+int(yv),b+1)
    def predict(alpha):
        return np.array([(stats.get(k,(0,0))[0]+alpha*base)/(stats.get(k,(0,0))[1]+alpha) if -1 not in k else base for k in keys])
    alphas=[20,100];alpha=min(alphas,key=lambda a:log_loss(yf[fv],predict(a)[fv]));pg=predict(alpha)[fs]
    bundle=joblib.load(out/'contact_out_branch.joblib');pm=bundle['model'].predict_proba(f[bundle['features']].to_numpy(float)[fs])[:,1]
    grid={'baseline':'train-only four-dimensional grid, same edges as current site, smoothing chosen on validation','selected_alpha':alpha,'test_grid':binary_metrics(yf[fs],pg),'test_trained':binary_metrics(yf[fs],pm),'paired':paired_ci(yf[fs],pg,pm,f.loc[fs,'game_id'].to_numpy()),'caveat':'Not an evaluation of current site grid weights, which lack a verified training-only provenance.'};dump(out/'defense_grid_comparison.json',grid)
    # Portable tree artifacts, parity-checked against sklearn. No production replacement.
    exports=[]
    for path in out.glob('*.joblib'):
        bundle=joblib.load(path);m=bundle['model']
        if not hasattr(m,'_predictors'):continue
        trees=[]
        for iteration in m._predictors:
            assert len(iteration)==1
            nodes=iteration[0].nodes
            assert not nodes['is_categorical'].any()
            trees.append([[int(n['feature_idx']),float(n['num_threshold']) if np.isfinite(n['num_threshold']) else 'Infinity',int(n['missing_go_to_left']),int(n['is_leaf']),float(n['value']),int(n['left']),int(n['right'])] for n in nodes])
        obj={'format':'numeric-hgb-v1','features':bundle['features'],'baseline':float(m._baseline_prediction.ravel()[0]),'link':'sigmoid' if hasattr(m,'predict_proba') else 'identity','trees':trees,'status':'research; not production-approved'}
        dest=out/(path.stem+'.portable.json');dump(dest,obj)
        def manual(x):
            a=obj['baseline']
            for tree in trees:
                k=0
                while not tree[k][3]:
                    feat,thr,missing,_,_,left,right=tree[k];v=x[feat];thr=float(thr);k=left if (bool(missing) if np.isnan(v) else v<=thr) else right
                a+=tree[k][4]
            return 1/(1+np.exp(-a)) if obj['link']=='sigmoid' else a
        data=f if path.stem=='contact_out_branch' else (d if path.stem=='hard_fair_contact_per_pitch' else c)
        # Regression and conditional classifiers used frozen history; reuse X with explicit finite history values for parity only.
        for col in ['history_exit','history_angle','history_direction','history_n']:
            if col not in data:data[col]=0.
        x=data[bundle['features']].to_numpy(float)[:100];pred=m.predict_proba(x)[:,1] if obj['link']=='sigmoid' else m.predict(x)
        err=float(np.max(np.abs(pred-np.array([manual(row) for row in x]))));assert err<1e-10
        exports.append({'file':dest.name,'max_parity_error':err,'bytes':dest.stat().st_size})
    dump(out/'portable_export_checks.json',exports)
    print(json.dumps({'per_pitch':{'selected':report['selected'],'test':report['test_models'][report['selected']],'cohort':report['cohort'],'ci':report['paired_feature_baseline']},'defense_vs_grid':grid,'exports':exports},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

