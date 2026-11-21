const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(__dirname+'/../index.html','utf8');
for(const match of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g))if(!match[1].includes('application/json'))new vm.Script(match[2]);
const data=JSON.parse(html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1]);
const model=data.model;
const start=html.indexOf('function stageAttribution('),end=html.indexOf('\nfunction render()',start);
const context={};vm.createContext(context);vm.runInContext(html.slice(start,end),context);
let count=0;const classes=model.classes;
for(let s=0;s<24;s++){
 const branch=classes.map((_,k)=>model.runs[s][k]+model.transitions[s][k].reduce((a,t,j)=>a+t*(model.states[j]||0),0));
 const q=classes.map((_,k)=>model.allowed[s][k]?1:0);const n=q.reduce((a,b)=>a+b,0);q.forEach((v,i)=>q[i]=v/n);
 for(const result of classes)for(const pOut of [0,.2,.8,1]){
  const calc=p=>context.stageAttribution({qBefore:q,qAfter:q,branch,classes,result,pOut:p});
  const a=calc(pOut);assert(Number.isFinite(a.pitcher));assert(Math.abs(a.pitcher)<1e-12);
  assert(Math.abs(a.event-(a.value-a.e1))<1e-12);
  if(a.batter!==null){assert(Math.abs(a.error)<1e-12);assert(Math.abs(a.batter-a.fielding-a.event)<1e-12);
   const low=calc(.2),high=calc(.8);
   if(branch[classes.indexOf('CONTACT_OUT')]<=branch[classes.indexOf('1B')]&&branch[classes.indexOf('CONTACT_OUT')]<=branch[classes.indexOf('2B')]&&branch[classes.indexOf('CONTACT_OUT')]<=branch[classes.indexOf('3B')]){
    assert(low.batter>=high.batter-1e-12);assert(low.fielding>=high.fielding-1e-12);
   }
  }else assert(a.fielding===null&&a.e2===null);
  count++;
 }
}
const test={classes:['CONTACT_OUT','1B','2B','3B'],branch:[0,1,2,3],qBefore:[.8,.1,.05,.05],qAfter:[.4,.3,.15,.15],result:'CONTACT_OUT',pOut:.2};
const out=context.stageAttribution(test);assert(out.pitcher<0);assert(out.batter>0);assert(out.fielding>0);assert(Math.abs(out.error)<1e-12);
assert(html.includes('id="stage-attribution-v04"'));assert(html.includes('id="battersp"'));
console.log(JSON.stringify({status:'PASS',scenarios:count,syntax:'all inline scripts parsed',identities:'conservation and event identity',behavior:'difficult outs reward batter and fielding; risky location penalizes pitcher',example:out,accuracy:'NOT VALIDATED'},null,2));

