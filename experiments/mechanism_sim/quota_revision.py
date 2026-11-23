import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,gzip,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import improve
OUT = Path(os.environ.get('C3DRIVE_QUOTA_RUNS', improve.ROOT / 'runs' / 'mechanism-quota'))

def selector(e,pool,old,candidate,variant,rng):
    before=e.loss(old.predict(pool['X'],pool['c']),pool['Y']);after=e.loss(candidate.predict(pool['X'],pool['c']),pool['Y']);risk=np.maximum(0,after-before)
    z=e.norm(np.column_stack([pool['X'],pool['c']]));ranks=e.nondominated(pool['S'],risk,pool['C']);chosen=[]
    domains=np.unique(pool['source']);floor=int(e.BUDGET*.5)//len(domains)
    # Common deterministic coverage reservation. No test metrics enter selection.
    def cover(ids,need):
        md=((z[ids,None]-z[np.array(chosen)][None])**2).sum(2).min(1) if chosen else np.full(len(ids),np.inf)
        for _ in range(need):
            k=int(md.argmax());pick=int(ids[k]);chosen.append(pick);md=np.minimum(md,((z[ids]-z[pick])**2).sum(1));md[k]=-1
    for dom in domains:
        ids=np.flatnonzero(pool['source']==dom);cover(ids,min(floor,len(ids)))
    reserved=len(chosen)
    if variant=='quota_random':
        ids=np.setdiff1d(np.arange(len(risk)),chosen);chosen.extend(rng.choice(ids,min(e.BUDGET-len(chosen),len(ids)),replace=False).tolist())
    else:
        for rank in range(1,ranks.max()+1):
            ids=np.setdiff1d(np.flatnonzero(ranks==rank),chosen);need=e.BUDGET-len(chosen)
            if len(ids)<=need:chosen.extend(ids.tolist())
            else:cover(ids,need);break
    assert len(chosen)==min(e.BUDGET,len(risk)) and len(set(chosen))==len(chosen)
    assert all(np.sum(pool['source'][chosen]==d)>=min(floor,np.sum(pool['source']==d)) for d in domains)
    selected=np.isin(np.arange(len(risk)),chosen)
    detail={'retention_risk':risk,'old_loss':before,'new_loss':after,'features':z,'ranks':ranks,'selected':selected,'boundary':np.array([],int)}
    return {k:v[chosen].copy() for k,v in pool.items()},detail

def worker(job):
    order,variant,run=job;path=OUT/'runs'/f'{order}_{variant}_{run:02d}';(path/'data').mkdir(parents=True,exist_ok=True)
    dest=path/'result.json.gz'
    if dest.exists():return path.name,'cached'
    e=improve.load_engine();e.SEQUENCE=improve.ORDERS[order];e.TOTAL_BLOCKS=len(e.SEQUENCE)*64
    e.select_memory=lambda pool,old,candidate,v,rng:selector(e,pool,old,candidate,v,rng)
    threshold=json.loads((improve.OLD/'simulation_summary.json').read_text(encoding='utf-8'))['threshold']
    result=e.simulate(run,variant,e.worlds(run),e.initial_registry(),threshold,path)
    doc={'job':job,'sequence':e.SEQUENCE,**{k:result[k] for k in ['stages','blocks','events','attempts','unfinished']}}
    with gzip.open(dest,'wt',encoding='utf-8') as f:json.dump(e.plain(doc),f,separators=(',',':'))
    return path.name,{'score':float(np.mean([r[5] for r in result['stages'] if r[2]==11])),'commits':sum(r['passed'] for r in result['events'])}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    jobs=[('default',v,r) for r in range(1,13) for v in ['quota_risk','quota_random']]+[(o,'quota_risk',r) for r in range(1,13) for o in ['alternating','permuted']]
    plan={'scope':'Controlled synthetic control task only; not vehicle measurements.','seeds':list(range(1,13)),'jobs':jobs,'memory_budget':400,'adapter_slots':4,'quota_fraction':.5,'quota_rule':'Each available source domain reserves min(available, floor(0.5*B/K)) states using common deterministic farthest-first coverage; remaining slots use either 3-D Pareto then boundary k-center, or uniform random. Source-domain metadata is assumed available; no test-based tuning.','fixed':'Same teacher, episode seeds, training solver, gates, detector and per-proposal training budget. Total proposals may differ.','selection':'All prespecified runs retained, irrespective of results. No quota sweep.','engine_sha256':hashlib.sha256((improve.OUT/'engine_extended.py').read_bytes()).hexdigest()}
    (OUT/'experiment_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    start=time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):print(i,len(jobs),f.result(),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
