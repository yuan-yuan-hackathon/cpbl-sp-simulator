const fs=require('fs'),vm=require('vm'),zlib=require('zlib'),assert=require('assert'),cp=require('child_process');
const ctx={};vm.createContext(ctx);vm.runInContext(fs.readFileSync('v06-runtime.js','utf8'),ctx);
const zip='research/pitch-runs-v06/CPBL_pitch_runs_v06_validation.zip';
const read=name=>cp.execFileSync('python3',['-c','import sys,zipfile;sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))',zip,name],{maxBuffer:30e6});
const bundle=JSON.parse(zlib.gunzipSync(read('models.json.gz'))),fixture=JSON.parse(read('parity-fixture.json'));
const ids=Object.keys(bundle.profiles.batters).concat(['missing-batter']),pitcher=Object.keys(bundle.profiles.pitchers)[0];let cases=0;
for(let balls=0;balls<4;balls++)for(let strikes=0;strikes<3;strikes++){
 const ref=ctx.v06Infer(bundle,fixture.input,ids[0],pitcher,balls,strikes);
 for(const id of ids){const got=ctx.v06Infer(bundle,fixture.input,id,pitcher,balls,strikes);
  for(const stage of ['before','after']){assert.deepStrictEqual(got[stage],ref[stage]);assert(Math.abs(got[stage].reduce((a,b)=>a+b,0)-1)<1e-10);}
  assert.equal(got.batterBaseline,'league_average');cases++;
 }
}
assert(bundle.profiles.batter_prior.every(Number.isFinite));
console.log(JSON.stringify({cases,batters:ids.length,counts:12,league_batter_profile:bundle.profiles.batter_prior,result:'PASS'}));
