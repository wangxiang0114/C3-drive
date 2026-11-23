"""Run domain order, observation shift, and replay comparisons."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,importlib.util,json,gzip,time,hashlib,copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('C3DRIVE_EXTENDED_RUNS', ROOT / 'runs' / 'mechanism-extended'))
OLD = Path(os.environ.get('C3DRIVE_BASE_RUNS', ROOT / 'runs' / 'mechanism-sim'))
N=12
ORDERS={
 'default':[0,1,2,3,4,5,6,7,8,2,0,5],
 'alternating':[0,1,6,2,5,3,8,4,7,2,0,5],
 'permuted':[0,5,3,8,1,7,4,6,2,2,0,5],
}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    source=(Path(__file__).parent/'engine.py').read_text(encoding='utf-8')
    source=source.replace("ALLOCATED=DX*2+M*(DP*H+DB*2)","ALLOCATED=DX*2+M*(DP*H+DB*2)\nMERGE_STRATEGY=None\nEVENT_AUDIT=None\nSCENARIO=None")
    source=source.replace("if variant=='random_memory':chosen=rng.choice(len(R),min(BUDGET,len(R)),replace=False).tolist()", """if variant=='random_memory':chosen=rng.choice(len(R),min(BUDGET,len(R)),replace=False).tolist()
    elif variant=='coverage_only':
        md=np.full(len(R),np.inf)
        for k in range(min(BUDGET,len(R))):
            pick=int(rng.integers(len(R))) if k==0 else int(md.argmax())
            chosen.append(pick);md=np.minimum(md,((z-z[pick])**2).sum(1));md[chosen]=-1
    elif variant=='pareto_only':
        for rank in range(1,ranks.max()+1):
            ids=np.flatnonzero(ranks==rank);need=BUDGET-len(chosen)
            if len(ids)<=need:chosen.extend(ids.tolist())
            else:chosen.extend(rng.choice(ids,need,replace=False).tolist());break""")
    source=source.replace("options.sort(key=lambda a:a[3])", """options.sort(key=lambda a:a[3])
        if isinstance(MERGE_STRATEGY, tuple):
            options=[o for o in options if (o[1],o[2])==MERGE_STRATEGY]
        elif MERGE_STRATEGY=='random' and options:
            options=[options[int(rng_for(run,971,block).integers(len(options)))]]
        elif MERGE_STRATEGY=='parameter' and options:
            options.sort(key=lambda a:float(np.sum((old.U[a[1]]-old.U[a[2]])**2)+np.sum((old.V[a[1]]-old.V[a[2]])**2)))
            options=options[:1]
        elif MERGE_STRATEGY=='functional':options=options[:1]""")
    source=source.replace("model=base_model(run,world,registry);memory=empty_memory();rng=rng_for(run,50)","model=base_model(run,world,registry);memory=empty_memory();rng=rng_for(run,50);base_world=copy.deepcopy(world)")
    source=source.replace("phase=block//PHASE_BLOCKS;domain=SEQUENCE[phase];before_hash", """phase=block//PHASE_BLOCKS;domain=SEQUENCE[phase]
        if SCENARIO is not None:
            world=copy.deepcopy(base_world)
            if SCENARIO=='appearance':
                for key in ['e_bias','heading_bias','speed_gain']:world[1][key]=world[0][key]
            elif SCENARIO=='hidden':world[1]['context']=world[0]['context']+.08*(world[1]['context']-world[0]['context'])
            elif SCENARIO=='gradual':
                alpha=min(1.,max(0.,(block-PHASE_BLOCKS+1)/32))
                for key in ['context','e_bias','heading_bias','speed_gain']:world[1][key]=world[0][key]+alpha*(world[1][key]-world[0][key])
        before_hash""")
    source=source.replace("accepted=proposal['accepted'] and bool(finalgate['passed']);old_active", """accepted=proposal['accepted'] and bool(finalgate['passed'])
                if EVENT_AUDIT is not None:EVENT_AUDIT(model,memory,proposal,world,domain,run,block,pending,accepted)
                old_active""")
    source=source.replace("float(np.mean(frame_scores)),commits,merges,model.fingerprint(),memory_fingerprint(memory)])", "float(np.mean(frame_scores)),commits,merges,model.fingerprint(),memory_fingerprint(memory),float(mahal(z,registry['mu'],registry['precision']))])")
    target=OUT/'engine_extended.py';target.write_text(source,encoding='utf-8')
    compile(source,str(target),'exec')
    plan={'seed_count_new_comparisons':N,'orders':ORDERS,'memory_variants':['full','random_memory','coverage_only','pareto_only'],'detector_conditions':['coupled','appearance','hidden','gradual'],'inherited_full_and_random_runs':30,'selection_rule':'Use runs 1 through 12, without outcome-based filtering. Matched one-pair merge comparisons use the first full-pool proposal in each canonical run. Gate counterfactuals use every completed canonical full-system proposal.','frozen_engine_sha256':hashlib.sha256((Path(__file__).parent/'engine.py').read_bytes()).hexdigest(),'extension_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'mechanism_sim'}
    (OUT/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')

def load_engine():
    spec=importlib.util.spec_from_file_location('extended_engine',OUT/'engine_extended.py');e=importlib.util.module_from_spec(spec);sys.modules['extended_engine']=e;spec.loader.exec_module(e);return e

def worker(job):
    kind,name,variant,run=job
    path=OUT/'runs'/f'{kind}_{name}_{variant}_{run:02d}';(path/'data').mkdir(parents=True,exist_ok=True)
    cache=path/'result.json.gz'
    if cache.exists():return str(path.name),'cached'
    e=load_engine();e.SEQUENCE=ORDERS[name] if kind=='order' else [0,1,0,1,0,1,0,1,0,1,0,1];e.TOTAL_BLOCKS=len(e.SEQUENCE)*e.PHASE_BLOCKS
    if kind=='condition':e.SCENARIO=name
    registry=e.initial_registry();h=json.loads((OLD/'simulation_summary.json').read_text(encoding='utf-8'))['threshold']
    audits=[];pairs=[];first=[False]
    def audit(old,memory,proposal,world,domain,run,block,pending,accepted):
        domains=sorted(set([0]+[r['source_domain'] for r in old.routes]));historical=[d for d in domains if d!=domain]
        previous=e.evaluate(old,world,list(range(9)),run,92,block);candidate=e.evaluate(proposal['candidate'],world,list(range(9)),run,92,block)
        drop=max([previous[d][0]-candidate[d][0] for d in historical],default=0.)
        sr_drop=max([previous[d][1]-candidate[d][1] for d in historical],default=0.)
        audits.append({'run':run,'block':block,'accepted':bool(accepted),'merge_proposed':proposal['merged'],'historical_max_drop':drop,'historical_success_drop':sr_drop,'current_gain':candidate[domain][0]-previous[domain][0],'all_domain_gain':np.mean([candidate[d][0]-previous[d][0] for d in range(9)]),'old_hash':old.fingerprint(),'candidate_hash':proposal['candidate'].fingerprint(),'memory_count':len(memory['id'])})
        # Preserve rejected candidates so gate counterfactuals remain auditable.
        if not accepted:
            np.savez_compressed(path/'data'/f'rejected_candidate_{block}.npz',base=old.base,U=old.U,old_V=old.V,candidate_V=proposal['candidate'].V,old_route_mu=np.array([r['mu'] for r in old.routes]),candidate_route_mu=np.array([r['mu'] for r in proposal['candidate'].routes]),old_route_slot=np.array([r['slot'] for r in old.routes]),candidate_route_slot=np.array([r['slot'] for r in proposal['candidate'].routes]))
        if old.occupied.sum()==e.M and not first[0]:
            first[0]=True;samples,reg=e.collect(old,world,domain,run,pending['trigger_block'])
            for strategy in ['functional','parameter','random']:
                e.MERGE_STRATEGY=strategy
                branch=e.propose(old,memory,samples,reg,world,domain,run,pending['trigger_block'],'full')
                fg,_,_=e.gate_metrics(old,branch['candidate'],world,domain,run,block)
                tested=e.evaluate(branch['candidate'],world,list(range(9)),run,92,block)
                hd=max([previous[d][0]-tested[d][0] for d in historical],default=0.)
                pairs.append({'run':run,'block':block,'strategy':strategy,'pair_i':branch['pair_i'],'pair_j':branch['pair_j'],'old_hash':old.fingerprint(),'historical_max_drop':hd,'current_gain':tested[domain][0]-previous[domain][0],'gate_passed':bool(branch['accepted'] and fg['passed']),'attempts':len(branch['attempt_logs'])})
            e.MERGE_STRATEGY=None
    if kind=='order' and name=='default' and variant=='full':e.EVENT_AUDIT=audit
    result=e.simulate(run,variant,e.worlds(run),registry,h,path)
    # Raw logs retain all model/memory fingerprints and the extra frozen-reference block statistic.
    doc={'job':job,'sequence':e.SEQUENCE,'stages':result['stages'],'blocks':result['blocks'],'events':result['events'],'attempts':result['attempts'],'audits':audits,'pairs':pairs,'unfinished':result['unfinished']}
    with gzip.open(cache,'wt',encoding='utf-8') as f:json.dump(e.plain(doc),f,separators=(',',':'))
    return path.name,{'events':len(result['events']),'audits':len(audits),'pairs':len(pairs)}

def calibrate_single():
    target=OUT/'single_block_calibration.npz'
    if target.exists():return
    e=load_engine();reg=e.initial_registry();maxima=[];held=[]
    for stream,n,dest in [(81,600,maxima),(82,300,held)]:
        for group in range(0,n,20):
            rng=e.rng_for(0,stream,group);c=e.norm(np.array([3,0,0,0,0,0,0,0])+.4*e.ar(rng,20,e.TOTAL_BLOCKS*e.L));z=c.reshape(20,e.TOTAL_BLOCKS,e.L,e.DC).mean(2)
            dest.extend(e.mahal(z,reg['mu'],reg['precision']).max(1).tolist())
    h=float(np.sort(maxima)[594]);np.savez_compressed(target,episode_maxima=maxima,heldout_maxima=held,threshold=h)
    print('Single-block threshold',h,'heldout alarms',sum(np.array(held)>h),flush=True)

def main():
    prepare();calibrate_single();jobs=[]
    for run in range(1,N+1):
        for variant in ['full','coverage_only','pareto_only']:jobs.append(('order','default',variant,run))
        for name in ['alternating','permuted']:
            for variant in ['full','random_memory']:jobs.append(('order',name,variant,run))
        for name in ['coupled','appearance','hidden','gradual']:jobs.append(('condition',name,'full',run))
    start=time.time();print('Jobs:',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            name,info=f.result();print(i,'/',len(jobs),name,info,'seconds',round(time.time()-start,1),flush=True)
    print('ALL COMPLETED',flush=True)

if __name__=='__main__':main()
