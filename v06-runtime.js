'use strict';
function v06Probs(m,x){
 const scores=m.baseline.slice();
 for(const iteration of m.trees)for(let k=0;k<iteration.length;k++){
  const t=iteration[k];let i=0;
  while(!t[i][3]){const [f,threshold,missing,,,left,right]=t[i],v=x[m.features[f]];i=(v==null||!Number.isFinite(v)?!!missing:v<=Number(threshold))?left:right;}
  scores[k]+=t[i][4];
 }
 const top=Math.max(...scores),mass=scores.map(v=>Math.exp(v-top)),sum=mass.reduce((a,b)=>a+b,0);return mass.map(v=>v/sum);
}
function v06Out(m,x){
 let score=m.baseline;
 for(const t of m.trees){let i=0;while(!t[i][3]){const [f,threshold,missing,,,left,right]=t[i],v=x[m.features[f]];i=(v==null||!Number.isFinite(v)?!!missing:v<=Number(threshold))?left:right;}score+=t[i][4];}
 return score>=0?1/(1+Math.exp(-score)):Math.exp(score)/(1+Math.exp(score));
}
const V06_BRANCHES=['K','CONTACT_OUT','WALK','1B','2B','3B','HR','OTHER'];
function v06Absorb(q,t,b,s,loop=false){
 const r=Array(8).fill(0),unit=k=>r.map((_,i)=>+(i===k)),add=(p,v)=>v.forEach((n,i)=>r[i]+=p*n);
 add(q[0],b<3?t[b+1][s]:unit(2));add(q[1],s<2?t[b][s+1]:unit(0));add(q[3],unit(2));
 if(s<2)add(q[2],t[b][s+1]);else if(!loop)add(q[2],t[b][s]);
 ['CONTACT_OUT','1B','2B','3B','HR','OTHER'].forEach((k,i)=>r[V06_BRANCHES.indexOf(k)]+=q[i+4]);
 return loop&&s===2?r.map(v=>v/(1-q[2])):r;
}
function v06Count(q){const t=Array.from({length:4},()=>Array.from({length:3},()=>Array(8).fill(0)));for(let b=3;b>=0;b--)for(let s=2;s>=0;s--)t[b][s]=v06Absorb(q[b][s],t,b,s,true);return t;}
function v06Id(v){return String(v).replace(/^cpbl:/,'').replace(/^0+/,'')||'0';}
function v06InferProfiles(bundle,x,bh,ph,balls,strikes){
 const input={...x};['history_exit','history_angle','history_direction','history_n'].forEach((k,i)=>input[k]=bh[i]);bundle.baseline.classes.forEach((k,i)=>input['pitcher_rate_'+k]=ph[i]);
 const q=Array.from({length:4},(_,b)=>Array.from({length:3},(_,s)=>v06Probs(bundle.baseline,{...input,balls_before:b,strikes_before:s}))),future=v06Count(q);
 const now=v06Probs(bundle.pitch,{...input,balls_before:balls,strikes_before:strikes});
 return {before:future[balls][strikes],after:v06Absorb(now,future,balls,strikes)};
}
// Batter-side and event values: league-average current batter, actual pitcher.
function v06Infer(bundle,x,batterId,pitcherId,balls,strikes){
 const profiles=bundle.profiles,pit=profiles.pitchers[v06Id(pitcherId)];
 return {...v06InferProfiles(bundle,x,profiles.batter_prior,pit||profiles.pitcher_prior,balls,strikes),batterFallback:false,batterBaseline:"league_average",pitcherFallback:!pit};
}
// Pitcher-side values: actual current batter, league-average pitcher.
function v06InferPitcherNeutral(bundle,x,batterId,balls,strikes){
 const profiles=bundle.profiles,bat=profiles.batters[v06Id(batterId)];
 return {...v06InferProfiles(bundle,x,bat||profiles.batter_prior,profiles.pitcher_prior,balls,strikes),batterFallback:!bat,pitcherBaseline:"league_average"};
}
async function v06Download(path,hash){
 const r=await fetch(path);if(!r.ok)throw Error('無法下載模型');const data=await r.arrayBuffer();
 const actual=[...new Uint8Array(await crypto.subtle.digest('SHA-256',data))].map(x=>x.toString(16).padStart(2,'0')).join('');if(actual!==hash)throw Error('模型檔校驗失敗');return data;
}
async function v06ZipEntry(buffer,wanted){
 const v=new DataView(buffer),bytes=new Uint8Array(buffer),decoder=new TextDecoder();let end=-1;
 for(let i=bytes.length-22;i>=Math.max(0,bytes.length-65557);i--)if(v.getUint32(i,true)===0x06054b50){end=i;break;}
 if(end<0)throw Error('ZIP格式錯誤');let pos=v.getUint32(end+16,true),count=v.getUint16(end+10,true);
 for(let i=0;i<count;i++){
  if(v.getUint32(pos,true)!==0x02014b50)throw Error('ZIP索引錯誤');
  const method=v.getUint16(pos+10,true),size=v.getUint32(pos+20,true),n=v.getUint16(pos+28,true),e=v.getUint16(pos+30,true),c=v.getUint16(pos+32,true),offset=v.getUint32(pos+42,true),name=decoder.decode(bytes.slice(pos+46,pos+46+n));
  if(name===wanted){const start=offset+30+v.getUint16(offset+26,true)+v.getUint16(offset+28,true),data=bytes.slice(start,start+size);if(method===0)return data;if(method===8)return new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer();throw Error('ZIP壓縮格式不符');}
  pos+=46+n+e+c;
 }
 throw Error('模型項目不存在');
}
async function loadV06(){
 const [a,b]=await Promise.all([
  v06Download('research/pitch-runs-v06/CPBL_pitch_runs_v06_validation.zip','95813a8247a0981e6dcd01c6eaf02e2aca182270bbb84c332ec63524ea0d82a6'),
  v06Download('research/stage-training-v05/CPBL_stage_training_v05.zip','b98af18f0b9ef9259af460e2b9b0a3754a51d632f4217d472ce91fdc0a432cd3')]);
 const [gz,out]=await Promise.all([v06ZipEntry(a,'models.json.gz'),v06ZipEntry(b,'contact_out_branch.portable.json')]);
 const model=await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).json();model.out=JSON.parse(new TextDecoder().decode(out));
 if(model.baseline.format!=='multiclass-hgb-v1'||model.pitch.format!=='multiclass-hgb-v1'||model.out.format!=='numeric-hgb-v1')throw Error('模型版本不符');return model;
}
