const fs=require('fs'),vm=require('vm'),zlib=require('zlib'),assert=require('assert');
const ctx={console};vm.createContext(ctx);vm.runInContext(fs.readFileSync('site/v06-runtime.js','utf8'),ctx);
const bundle=JSON.parse(zlib.gunzipSync(fs.readFileSync('results/v06/models.json.gz'))),f=JSON.parse(fs.readFileSync('results/v06/parity-fixture.json'));
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((x,i)=>assert(Math.abs(x-b[i])<1e-12,`${i}: ${x} != ${b[i]}`));};
close(ctx.v06Probs(bundle.baseline,f.input),f.baseline);close(ctx.v06Probs(bundle.pitch,f.input),f.pitch);
const qb=Array.from({length:4},(_,b)=>Array.from({length:3},(_,s)=>ctx.v06Probs(bundle.baseline,{...f.input,balls_before:b,strikes_before:s}))),future=ctx.v06Count(qb);
close(future[0][0],f.branch_before);close(ctx.v06Absorb(ctx.v06Probs(bundle.pitch,f.input),future,0,0),f.branch_after);
const out=JSON.parse(fs.readFileSync('v05/contact_out_branch.portable.json'));assert(Number.isFinite(ctx.v06Out(out,{})));console.log('JavaScript parity, 12-count conversion and missing-input out probability passed');
