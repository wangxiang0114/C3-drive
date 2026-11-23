"""Long-horizon capacity and memory comparisons.
One-factor capacity/memory sweeps and actual gate-policy interventions.
All prior data remain unchanged; no expected outcome is encoded.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,copy,gzip,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import improve

ROOT=improve.ROOT;OUT = Path(os.environ.get('C3DRIVE_LONG_RUNS', ROOT / 'runs' / 'mechanism-long'))
N=12;SEQUENCE=list(range(15))+[2,0,5]
CONFIGS=[('M2_B400',2,400,'nominal'),('M4_B200',4,200,'nominal'),('M4_B400',4,400,'nominal'),('M4_B800',4,800,'nominal'),('M6_B400',6,400,'nominal'),('M4_B400_strict',4,400,'strict'),('M4_B400_no_gate',4,400,'disabled')]

def worlds(e,run):
    w=e.worlds(run)
    for idx in range(9,15):
        theta=2*np.pi*(idx-9)/6+.23
        context=np.array([3,1.8*np.cos(theta),1.8*np.sin(theta),.6*np.sin(2*theta),.55*np.cos(2*theta),.45*np.sin(3*theta),0,0])
        w.append({'domain':idx,'context':context,'e_bias':float(.35*np.sin(theta+.5)+.15*np.cos(3*theta)),'heading_bias':float(.12*np.cos(theta-.3)),'speed_gain':float(1+.38*np.sin(theta-.8))})
    return w

def load_engine():
    import importlib.util
    spec=importlib.util.spec_from_file_location('boundary_engine',OUT/'engine_boundary.py');e=importlib.util.module_from_spec(spec);sys.modules['boundary_engine']=e;spec.loader.exec_module(e);return e

def prepare():
    (OUT/'runs').mkdir(parents=True,exist_ok=True)
    src=(improve.OUT/'engine_extended.py').read_text(encoding='utf-8')
    src=src.replace('SCENARIO=None','SCENARIO=None\nGATE_DISABLED=False')
    # Change only the explicitly disabled-gate variant; failed/no-option proposals remain invalid.
    src=src.replace("accepted=(gain>=CONFIG['new_domain_min_improvement']", "accepted=GATE_DISABLED or (gain>=CONFIG['new_domain_min_improvement']")
    src=src.replace("accepted=proposal['accepted'] and bool(finalgate['passed'])", "accepted=proposal['accepted'] and (GATE_DISABLED or bool(finalgate['passed']))")
    src=src.replace('list(range(9))','list(range(len(world)))')
    compile(src,'engine_boundary.py','exec');(OUT/'engine_boundary.py').write_text(src,encoding='utf-8')
    plan={'purpose':'Long-horizon capacity comparisons','runs':N,'sequence':SEQUENCE,'domain_count':15,'phase_blocks':64,'frames_per_block':16,'configurations':[{'name':a,'M':b,'B':c,'gate':d} for a,b,c,d in CONFIGS],'gate_nominal':{'history':.06,'clean':.035,'success':.09,'new_gain':.004},'gate_strict':{'history':.03,'clean':.02,'success':.05,'new_gain':.004},'gate_disabled':'First feasible functional candidate is submitted without candidate-validation filtering; invalid/no-pair proposals remain rejected. Other collection, routing and replay operations are identical.','design':'One-factor M sweep at B=400 and B sweep at M=4, not a full-factorial design; no M-by-B interaction claim. All seeds 1..12, same environment and per-proposal fitting budget. Total lifetime update compute can differ.','threshold_calibration':'Recalibrate on 600 clean episodes of 1152 blocks; 300 independent heldout episodes. Identical threshold for all configurations.','resource_scope':'Deployed numeric array payload and scalar metadata, not Python object overhead, process RSS, temporary training arrays, or external evaluation environment/teacher. Base controller 14 scalars; each adapter slot 360 frozen feature weights + 50 trained readout weights.','source_sha256':hashlib.sha256(src.encode()).hexdigest(),'no_claim_of_real_experimental_recovery':True}
    (OUT/'long_experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')

def setup(e,M,B,mode):
    e.M=M;e.BUDGET=B;e.ALLOCATED=e.DX*2+M*(e.DP*e.H+e.DB*2);e.SEQUENCE=SEQUENCE;e.TOTAL_BLOCKS=len(SEQUENCE)*64
    e.CONFIG.update({'slots':M,'replay_budget':B,'allocated_scalar_parameters':e.ALLOCATED,'historical_gate_tolerance':.03 if mode=='strict' else .06,'clean_gate_tolerance':.02 if mode=='strict' else .035,'success_tolerance':.05 if mode=='strict' else .09})
    e.GATE_DISABLED=mode=='disabled'

def numeric_state(e,z,prefix,memcount):
    network=z['base'].nbytes+z['U'].nbytes+z[prefix+'V'].nbytes
    aux=z[prefix+'occupied'].nbytes+z[prefix+'slot_age'].nbytes+3*8
    for k in ['route_mu','route_precision','route_slot','route_sources']:aux+=z[prefix+k].nbytes
    for k in range(len(z[prefix+'route_slot'])):aux+=z[prefix+f'route_cal_{k}'].nbytes
    replay=memcount*(7+8+2+1+1+1+1)*8
    return {'network_bytes':network,'router_detector_bytes':aux,'replay_bytes':replay,'total_bytes':network+aux+replay,'occupied_slots':int(z[prefix+'occupied'].sum()),'routes':len(z[prefix+'route_slot'])}

def worker(job):
    name,M,B,mode,run=job;folder=OUT/'runs'/f'{name}_run{run:02d}';(folder/'data').mkdir(parents=True,exist_ok=True);target=folder/'result.json.gz'
    if target.exists():return folder.name,'cached'
    e=load_engine();setup(e,M,B,mode);world=worlds(e,run);registry=e.initial_registry();h=float(np.load(OUT/'data/threshold_calibration.npz')['threshold'])
    result=e.simulate(run,'full',world,registry,h,folder)
    resources=[]
    with np.load(folder/'data'/f'checkpoint_run{run:02d}_full.npz') as z:
        for phase in range(len(SEQUENCE)):
            rr=next(r for r in result['stages'] if r[2]==phase);c=rr[9];rec=numeric_state(e,z,f'commit{c}_',rr[8]);rec.update({'phase':phase,'M':M,'B':B,'network_scalars':e.ALLOCATED});resources.append(rec)
    doc={'job':job,'sequence':SEQUENCE,'stages':result['stages'],'blocks':result['blocks'],'events':result['events'],'attempts':result['attempts'],'resources':resources,'unfinished':result['unfinished']}
    with gzip.open(target,'wt',encoding='utf-8') as f:json.dump(e.plain(doc),f,separators=(',',':'))
    return folder.name,{'events':len(result['events']),'accepted':sum(ev['passed'] for ev in result['events']),'merges':sum(ev['merge_committed'] for ev in result['events'])}

def main():
    prepare();(OUT/'data').mkdir(exist_ok=True)
    if not (OUT/'data/threshold_calibration.npz').exists():
        e=load_engine();setup(e,4,400,'nominal');h=e.calibrate_threshold(e.initial_registry(),OUT);print('Long-horizon threshold',h,flush=True)
    # A frozen plan is written before any outcomes are viewed.
    jobs=[(*cfg,run) for run in range(1,N+1) for cfg in CONFIGS]
    start=time.time();print('Long boundary jobs',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(fs),1):
            name,stat=f.result();print(i,'/',len(jobs),name,stat,'seconds',round(time.time()-start,1),flush=True)
    print('LONG EXPERIMENT COMPLETED',flush=True)

if __name__=='__main__':main()
