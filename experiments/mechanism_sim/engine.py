"""A reproducible synthetic driving-control system. No CARLA or measured data.

Detector -> collection -> random-feature adapter training / distillation ->
Pareto replay proposal -> finite closed-loop validation -> atomic commit.
This archive includes the original simulation underlying early mechanism plots.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,copy,csv,gzip,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SEED=20261023
M=4;H=24;DX=7;DC=8;DP=DX+DC;DB=H+1;BUDGET=400
L=16;PHASE_BLOCKS=64;SEQUENCE=[0,1,2,3,4,5,6,7,8,2,0,5]
TOTAL_BLOCKS=PHASE_BLOCKS*len(SEQUENCE);RHO=.5;KAPPA=1.1
ALLOCATED=DX*2+M*(DP*H+DB*2)
CONFIG={'seed':SEED,'description':'Synthetic closed-loop lateral and speed control; not a CARLA reproduction','slots':M,'adapter_hidden_units':H,'replay_budget':BUDGET,'sequence':SEQUENCE,'phase_blocks':PHASE_BLOCKS,'frames_per_block':L,'time_step':.1,'episode_steps':56,'test_episodes_per_domain':20,'gate_episodes_per_domain':12,'fit_blocks_initial':512,'calibration_blocks_initial':512,'fit_blocks_new':128,'calibration_blocks_new':128,'threshold_episodes':600,'threshold_episode_length_blocks':TOTAL_BLOCKS,'alpha_episode':.01,'rho':RHO,'kappa':KAPPA,'historical_gate_tolerance':.06,'clean_gate_tolerance':.035,'new_domain_min_improvement':.004,'success_tolerance':.09,'terminal_cross_track_tolerance':.2,'terminal_heading_tolerance':.12,'terminal_speed_error_tolerance':.3,'collector_episodes':32,'collector_steps':128,'training_samples_per_event':512,'collection_blocks':8,'proposal_block_offset':8,'decision_block_offset':12,'rejection_cooldown_blocks':8,'ridge':1e-4,'replay_anchor_weight':.6,'allocated_scalar_parameters':ALLOCATED,'variants':['full','random_memory','replace_oldest','frozen'],'representative_run':1}

def rng_for(*ids):return np.random.default_rng(np.random.SeedSequence([SEED,*map(int,ids)]))
def plain(a):
    if isinstance(a,np.ndarray):return plain(a.tolist())
    if isinstance(a,dict):return {str(k):plain(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)):return [plain(x) for x in a]
    if isinstance(a,(np.integer,np.bool_)):return int(a)
    if isinstance(a,(np.floating,float)):return float(a) if np.isfinite(a) else None
    return a
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def ar(rng,n,steps,dim=DC):
    z=rng.normal(size=(n,steps,dim));last=rng.normal(size=(n,dim))
    for j in range(steps):last=RHO*last+np.sqrt(1-RHO**2)*z[:,j];z[:,j]=last
    return z

def worlds(run):
    # Context is a noisy observed rendering feature, never a domain ID.
    context=np.array([[3,0,0,0,0,0,0,0],[3,.9,0,0,0,0,0,0],[3,1.6,.15,0,0,0,0,0],[3,0,1.1,0,0,0,0,0],[3,.1,1.8,0,0,0,0,0],[3,-1.3,0,.6,0,0,0,0],[3,0,-1.3,-.5,0,0,0,0],[3,1.7,-1.6,1,0,0,0,0],[3,-1.7,1.4,-1,0,0,0,0]],float)
    eb=np.array([0,.12,.18,-.12,-.18,.30,-.30,.40,-.45])
    hb=np.array([0,.02,.04,-.04,-.06,-.07,.09,.13,-.15])
    vg=np.array([1,.95,.9,1.08,1.15,.75,1.3,.65,1.4])
    # Repeats change traffic/noise/training, not the predefined scenario grid.
    return [{'domain':i,'context':context[i],'e_bias':eb[i],'heading_bias':hb[i],'speed_gain':vg[i]} for i in range(9)]

def observation(states,domains,world,noise):
    e,h,v,vr,curv=states.T
    eb=np.array([world[i]['e_bias'] for i in domains]);hb=np.array([world[i]['heading_bias'] for i in domains]);vg=np.array([world[i]['speed_gain'] for i in domains])
    eo=e+eb+.012*noise[:,0];ho=h+hb+.006*noise[:,1];vo=v*vg+.015*noise[:,2]
    return np.column_stack([np.ones(len(states)),eo/.5,ho/.25,(vo-1.3)/.7,(vr-1.3)/.4,curv/.15,vo*curv/.2])
def expert(states):
    e,h,v,vr,curv=states.T
    return np.column_stack([np.clip(v*curv-1.4*e-1.8*h,-1.2,1.2),np.clip(.9*(vr-v),-1,1)])
def dynamics(states,action,noise):
    e,h,v,vr,curv=states.T
    e=e+.1*v*h+.003*noise[:,0]
    h=h+.1*(action[:,0]-v*curv)+.0015*noise[:,1]
    v=np.clip(v+.1*action[:,1]+.004*noise[:,2],.1,3.3)
    return np.column_stack([np.clip(e,-3,3),np.clip(h,-1.5,1.5),v,vr,curv])
def tracking(states):
    e,h,v,vr,_=states.T
    return np.exp(-((e/.38)**2+.4*(h/.20)**2+.25*((v-vr)/.55)**2))
def failed(states):return (np.abs(states[:,0])>1.0)|(np.abs(states[:,1])>.9)|(states[:,2]>2.8)
def severity(states):
    e,h,v,vr,_=states.T
    return np.maximum(0,np.abs(e)-.25)**2+.2*(np.abs(e)>.75)+.1*(np.abs(h)>.4)+.1*(v>vr+.65)
def loss(pred,target):return np.mean(((pred-target)/np.array([.8,.6]))**2,axis=1)
def phi(q,U):return np.column_stack([np.ones(len(q)),np.tanh(q@U)])
def ridge(F,Y,weights=None):
    w=np.ones(len(F))/len(F) if weights is None else weights
    return np.linalg.solve(F.T@(w[:,None]*F)+CONFIG['ridge']*np.eye(F.shape[1]),F.T@(w[:,None]*Y))

class Model:
    def __init__(self,base,U):
        self.base=base;self.U=U;self.V=np.zeros((M,DB,2));self.occupied=np.zeros(M,bool);self.slot_age=np.full(M,-1,int)
        self.routes=[{'slot':-1,'mu':None,'precision':None,'cal':None,'source_domain':0}]
    def clone(self):return copy.deepcopy(self)
    def route(self,context):
        # Minimum observed-context Mahalanobis distance, with no true domain ID.
        distances=np.column_stack([mahal(context,r['mu'],r['precision']) for r in self.routes])
        reg=distances.argmin(1);return np.array([self.routes[i]['slot'] for i in reg]),reg
    def predict(self,X,context,clipped=False):
        slots,_=self.route(context);q=np.column_stack([X,context]);out=X@self.base
        for k in np.unique(slots):
            if k<0:continue
            ids=slots==k;out[ids]+=phi(q[ids],self.U[k])@self.V[k]
        return np.clip(out,[-1.2,-1],[1.2,1]) if clipped else out
    def fingerprint(self):
        h=hashlib.sha256(self.V.tobytes()+self.occupied.tobytes())
        for r in self.routes:h.update(np.array([r['slot']],np.int64).tobytes());h.update(r['mu'].tobytes());h.update(r['cal'].tobytes())
        return h.hexdigest()[:20]

def mahal(z,mu,P):
    x=z-mu;return np.einsum('...i,ij,...j->...',x,P,x)
def fit_context(blocks,calblocks,source_domain):
    mu=blocks.mean(0);S=np.cov(blocks,rowvar=False);S=.85*S+.15*np.trace(S)/DC*np.eye(DC);P=np.linalg.inv(S)
    return {'slot':-1,'mu':mu,'precision':P,'cal':np.sort(mahal(calblocks,mu,P)),'source_domain':int(source_domain)}
def pvalue(score,cal):return (1+len(cal)-np.searchsorted(cal,score,side='left'))/(len(cal)+1)
def initial_registry():
    r=rng_for(0,80);c=norm(np.array([3,0,0,0,0,0,0,0])+ .40*ar(r,1024,L))
    blocks=c.mean(1);return fit_context(blocks[:512],blocks[512:],0)
def calibrate_threshold(registry,out):
    maxima=[];kappa=KAPPA
    for group in range(0,600,20):
        r=rng_for(0,81,group);c=norm(np.array([3,0,0,0,0,0,0,0])+.4*ar(r,20,TOTAL_BLOCKS*L))
        z=c.reshape(20,TOTAL_BLOCKS,L,DC).mean(2);scores=mahal(z,registry['mu'],registry['precision']);p=pvalue(scores,registry['cal']);g=np.zeros(20);mx=np.zeros(20)
        for b in range(TOTAL_BLOCKS):g=np.maximum(0,g-np.log(p[:,b])-kappa);mx=np.maximum(mx,g)
        maxima.extend(mx.tolist())
    order=int(np.ceil((len(maxima)+1)*.99));h=float(np.sort(maxima)[order-1])
    heldout=[]
    for group in range(0,300,20):
        r=rng_for(0,82,group);c=norm(np.array([3,0,0,0,0,0,0,0])+.4*ar(r,20,TOTAL_BLOCKS*L));z=c.reshape(20,TOTAL_BLOCKS,L,DC).mean(2);p=pvalue(mahal(z,registry['mu'],registry['precision']),registry['cal']);g=np.zeros(20);mx=np.zeros(20)
        for b in range(TOTAL_BLOCKS):g=np.maximum(0,g-np.log(p[:,b])-kappa);mx=np.maximum(mx,g)
        heldout.extend(mx.tolist())
    np.savez_compressed(out/'data/threshold_calibration.npz',episode_maxima=maxima,heldout_episode_maxima=heldout,order_index=order,threshold=h,clean_mu=registry['mu'],clean_precision=registry['precision'],clean_cal=registry['cal'])
    return h

def make_states(rng,n):
    return np.column_stack([rng.normal(0,.30,n),rng.normal(0,.12,n),rng.uniform(.8,1.9,n),rng.uniform(1.,1.7,n),rng.uniform(-.12,.12,n)])
def base_model(run,world,registry):
    rng=rng_for(run,1);states=make_states(rng,2048);X=observation(states,np.zeros(len(states),int),world,rng.normal(size=(len(states),3)));Y=expert(states)
    base=ridge(X,Y);U=rng.normal(0,.24,(M,DP,H));model=Model(base,U);model.routes[0]=copy.deepcopy(registry)
    return model

def evaluate(model,world,domains,run,split,stage=0):
    """Finite closed-loop episodes; tests use fixed seeds across checkpoints.
    Gate episodes are independent from training and final test episodes.
    """
    episodes=CONFIG['test_episodes_per_domain'] if split==90 else CONFIG['gate_episodes_per_domain']
    ids=np.repeat(domains,episodes);n=len(ids);rng=rng_for(run,split,stage if split!=90 else 0)
    states=make_states(rng,n);context_noise=ar(rng,n,CONFIG['episode_steps']);sensor=rng.normal(size=(n,CONFIG['episode_steps'],3));process=rng.normal(size=(n,CONFIG['episode_steps'],3))
    means=np.array([world[i]['context'] for i in ids]);sums=np.zeros(n);bad=np.zeros(n,bool)
    for t in range(CONFIG['episode_steps']):
        c=norm(means+.4*context_noise[:,t]);X=observation(states,ids,world,sensor[:,t]);action=model.predict(X,c,True);states=dynamics(states,action,process[:,t]);sums+=tracking(states);bad|=failed(states)
    result={}
    for domain in domains:
        success=(~bad)&(np.abs(states[:,0])<=CONFIG['terminal_cross_track_tolerance'])&(np.abs(states[:,1])<=CONFIG['terminal_heading_tolerance'])&(np.abs(states[:,2]-states[:,3])<=CONFIG['terminal_speed_error_tolerance'])
        mask=ids==domain;result[int(domain)]=(float((sums[mask]/CONFIG['episode_steps']).mean()),float(success[mask].mean()))
    return result

def collect(model,world,domain,run,block):
    """32 shadow episodes in the currently observed environment; teacher labels
    are available only here. Neither future environments nor final tests enter.
    """
    rng=rng_for(run,20,block);n=32;steps=128;states=make_states(rng,n);ids=np.full(n,domain,int)
    cn=ar(rng,n,steps);sensor=rng.normal(size=(n,steps,3));pn=rng.normal(size=(n,steps,3));rows=[];contexts=[]
    for t in range(steps):
        c=norm(world[domain]['context']+.4*cn[:,t]);X=observation(states,ids,world,sensor[:,t]);Y=expert(states);action=model.predict(X,c,True)
        rows.append((X.copy(),c.copy(),Y.copy(),severity(states),failed(states).astype(int)))
        contexts.append(c.copy());action=np.clip(action+rng.normal(0,.045,action.shape),[-1.2,-1],[1.2,1]);states=dynamics(states,action,pn[:,t])
    # Split episodes, not adjacent frames, for context fitting and calibration.
    ctx=np.stack(contexts,axis=1);bl=ctx.reshape(n,8,L,DC).mean(2)
    reg=fit_context(bl[:16].reshape(-1,DC),bl[16:].reshape(-1,DC),domain)
    flat=[np.concatenate([r[k] for r in rows],axis=0) for k in range(5)]
    training_indices=np.flatnonzero(np.tile(np.arange(n)<16,steps))
    choose=rng.choice(training_indices,512,replace=False)
    return {'X':flat[0][choose],'c':flat[1][choose],'Y':flat[2][choose],'S':flat[3][choose],'C':flat[4][choose],'id':run*10000000+block*1000+np.arange(512),'source':np.full(512,domain,int)},reg

def empty_memory():return {'X':np.empty((0,DX)),'c':np.empty((0,DC)),'Y':np.empty((0,2)),'S':np.empty(0),'C':np.empty(0,int),'id':np.empty(0,int),'source':np.empty(0,int)}
def join_memory(a,b):return {k:np.concatenate([a[k],b[k]],axis=0) for k in a}
def memory_fingerprint(memory):return hashlib.sha256(memory['id'].astype(np.int64).tobytes()).hexdigest()[:20]

def ranks_exact(S,R,C):
    n=len(S);order=np.lexsort((-R,-C,-S));rr=np.argsort(-R,kind='stable');positions=np.empty(n,int);positions[rr]=np.arange(1,n+1);trees=np.zeros((2,n+1),int);ranks=np.zeros(n,int)
    # Continuous losses can tie at exactly zero. Identical objective vectors must
    # receive the same rank, so update a tied group only after all its queries.
    start=0
    while start<n:
        stop=start+1;i=order[start]
        while stop<n:
            j=order[stop]
            if S[j]!=S[i] or R[j]!=R[i] or C[j]!=C[i]:break
            stop+=1
        # Use R coordinate compression so equal R shares the same query index.
        group=order[start:stop]
        for i in group:
            best=0;p=int(positions[i])
            for kind in ([1] if C[i] else [0,1]):
                j=p
                while j>0:best=max(best,trees[kind,j]);j-=j&-j
            ranks[i]=best+1
        for i in group:
            j=int(positions[i])
            while j<=n:trees[C[i],j]=max(trees[C[i],j],ranks[i]);j+=j&-j
        start=stop
    return ranks

def nondominated(S,R,C):
    # At N <= 912 an independent pairwise implementation is sufficiently small,
    # handles exact zeros/ties, and avoids order-statistic tie ambiguities.
    v=np.column_stack([C,S,R]);dom=(v[:,None,:]>=v[None,:,:]).all(2)&(v[:,None,:]>v[None,:,:]).any(2)
    remaining=np.ones(len(S),bool);rank=np.zeros(len(S),int);level=1
    while remaining.any():
        ids=np.flatnonzero(remaining);front=ids[~dom[np.ix_(ids,ids)].any(0)];assert len(front)
        rank[front]=level;remaining[front]=False;level+=1
    return rank
def select_memory(pool,old,candidate,variant,rng):
    before=loss(old.predict(pool['X'],pool['c']),pool['Y']);after=loss(candidate.predict(pool['X'],pool['c']),pool['Y']);R=np.maximum(0,after-before)
    z=norm(np.column_stack([pool['X'],pool['c']]))
    ranks=nondominated(pool['S'],R,pool['C']);chosen=[];boundary=np.array([],int)
    if variant=='random_memory':chosen=rng.choice(len(R),min(BUDGET,len(R)),replace=False).tolist()
    else:
        for rank in range(1,ranks.max()+1):
            ids=np.flatnonzero(ranks==rank)
            if len(chosen)+len(ids)<=BUDGET:chosen.extend(ids.tolist())
            else:
                boundary=ids;need=BUDGET-len(chosen)
                md=((z[ids,None]-z[np.array(chosen)][None])**2).sum(2).min(1) if chosen else np.full(len(ids),np.inf)
                for _ in range(need):
                    k=int(md.argmax());pick=int(ids[k]);chosen.append(pick);md=np.minimum(md,((z[ids]-z[pick])**2).sum(1));md[k]=-1
                break
    selected=np.isin(np.arange(len(R)),chosen)
    detail={'retention_risk':R,'old_loss':before,'new_loss':after,'features':z,'ranks':ranks,'selected':selected,'boundary':boundary}
    return {k:v[chosen].copy() for k,v in pool.items()},detail

def train_slot(model,slot,samples,old,memory):
    X=samples['X'];c=samples['c'];Y=samples['Y'];q=np.column_stack([X,c]);F=phi(q,model.U[slot]);target=Y-X@model.base;w=np.ones(len(X))/len(X)
    if len(memory['id']):
        slots,_=model.route(memory['c']);mask=slots==slot
        if mask.any():
            xm=memory['X'][mask];cm=memory['c'][mask];oldtarget=old.predict(xm,cm);yt=.5*memory['Y'][mask]+.5*oldtarget
            F=np.vstack([F,phi(np.column_stack([xm,cm]),model.U[slot])]);target=np.vstack([target,yt-xm@model.base]);w=np.r_[w,np.full(mask.sum(),CONFIG['replay_anchor_weight']/mask.sum())]
    model.V[slot]=ridge(F,target,w)

def gate_metrics(old,candidate,world,current_domain,run,checkpoint):
    domains=sorted(set([0,current_domain]+[r['source_domain'] for r in old.routes]))
    previous=evaluate(old,world,domains,run,91,checkpoint);current=evaluate(candidate,world,domains,run,91,checkpoint)
    historical=[k for k in domains if k!=current_domain]
    maxdrop=max([previous[k][0]-current[k][0] for k in historical],default=0.)
    sr_drop=max([previous[k][1]-current[k][1] for k in historical],default=0.)
    gain=current[current_domain][0]-previous[current_domain][0];clean_drop=previous[0][0]-current[0][0]
    passed=(gain>=CONFIG['new_domain_min_improvement'] and maxdrop<=CONFIG['historical_gate_tolerance'] and clean_drop<=CONFIG['clean_gate_tolerance'] and sr_drop<=CONFIG['success_tolerance'])
    return {'historical_score_drop':maxdrop,'historical_success_drop':sr_drop,'current_gain':gain,'clean_drop':clean_drop,'passed':int(passed)},previous,current

def propose(old,memory,samples,newreg,world,current_domain,run,block,variant):
    pool=join_memory(memory,samples);occupied=np.flatnonzero(old.occupied);options=[]
    if len(occupied)<M:options=[None]
    elif variant=='replace_oldest':
        oldest=int(occupied[np.argmin(old.slot_age[occupied])]);options=[('replace',oldest,oldest,0.)]
    else:
        probes=memory if len(memory['id']) else samples;q=np.column_stack([probes['X'],probes['c']]);probe_slots,probe_regs=old.route(probes['c'])
        for i in occupied:
            for j in occupied:
                if i>=j:continue
                mask=(probe_slots==i)|(probe_slots==j)
                if not np.any(probe_slots==i) or not np.any(probe_slots==j):continue
                di=phi(q[mask],old.U[i])@old.V[i];dj=phi(q[mask],old.U[j])@old.V[j]
                groups=probe_regs[mask];per_group=[loss(di,dj)[groups==k].mean() for k in np.unique(groups)]
                options.append(('merge',int(i),int(j),float(np.mean(per_group))))
        options.sort(key=lambda a:a[3])
        if not options:
            # No represented pair can be distilled. Keep the deployed state.
            proposed_mem,detail=select_memory(pool,old,old,variant,rng_for(run,40,block,0))
            log={'attempt':0,'pair_i':-1,'pair_j':-1,'functional_distance':np.nan,'historical_score_drop':0.,'historical_success_drop':0.,'current_gain':0.,'clean_drop':0.,'passed':0}
            return {'candidate':old.clone(),'memory':proposed_mem,'pool':pool,'detail':detail,'accepted':False,'merged':0,'newslot':-1,'attempt_logs':[log],'old_gate':{},'candidate_gate':{},'pair_i':-1,'pair_j':-1,'functional_distance':np.nan}
    domains=sorted(set([0,current_domain]+[r['source_domain'] for r in old.routes]));previous=evaluate(old,world,domains,run,91,block)
    last=None;attempt_logs=[]
    for attempt,option in enumerate(options,1):
        candidate=old.clone();merged=0;pair_i=pair_j=-1;distance=np.nan
        if option is None:newslot=int(np.flatnonzero(~candidate.occupied)[0])
        elif option[0]=='replace':
            newslot=option[1]
            # Remove stale routing/calibration for the discarded adapter's domains.
            candidate.routes=[r for r in candidate.routes if r['slot']!=newslot]
        else:
            _,i,j,distance=option;pair_i=i;pair_j=j;merged=1;newslot=j
            slots,regs=old.route(memory['c']);mask=(slots==i)|(slots==j)
            # Function-space distillation: teacher outputs on historical inputs.
            xp=memory['X'][mask];cp=memory['c'][mask];teachers=old.predict(xp,cp)-xp@old.base
            if not len(xp):continue
            groups=regs[mask];unique=np.unique(groups);weights=np.array([1/(len(unique)*np.sum(groups==g)) for g in groups])
            candidate.V[i]=ridge(phi(np.column_stack([xp,cp]),candidate.U[i]),teachers,weights)
            for r in candidate.routes:
                if r['slot']==j:r['slot']=i
            candidate.occupied[j]=False
        candidate.occupied[newslot]=True;candidate.V[newslot]=0;candidate.slot_age[newslot]=block
        reg=copy.deepcopy(newreg);reg['slot']=newslot;candidate.routes.append(reg)
        train_slot(candidate,newslot,samples,old,memory)
        proposed_mem,detail=select_memory(pool,old,candidate,variant,rng_for(run,40,block,attempt))
        current=evaluate(candidate,world,domains,run,91,block)
        historical=[k for k in domains if k!=current_domain]
        maxdrop=max([previous[k][0]-current[k][0] for k in historical],default=0.)
        sr_drop=max([previous[k][1]-current[k][1] for k in historical],default=0.)
        gain=current[current_domain][0]-previous[current_domain][0];clean_drop=previous[0][0]-current[0][0]
        accepted=(gain>=CONFIG['new_domain_min_improvement'] and maxdrop<=CONFIG['historical_gate_tolerance'] and clean_drop<=CONFIG['clean_gate_tolerance'] and sr_drop<=CONFIG['success_tolerance'])
        attempt_logs.append({'attempt':attempt,'pair_i':pair_i,'pair_j':pair_j,'functional_distance':distance,'historical_score_drop':maxdrop,'historical_success_drop':sr_drop,'current_gain':gain,'clean_drop':clean_drop,'passed':int(accepted)})
        last={'candidate':candidate,'memory':proposed_mem,'pool':pool,'detail':detail,'accepted':accepted,'merged':merged,'newslot':newslot,'attempt_logs':attempt_logs,'old_gate':previous,'candidate_gate':current,'pair_i':pair_i,'pair_j':pair_j,'functional_distance':distance}
        if accepted:break
    assert last is not None
    return last

def simulate(run,variant,world,registry,threshold,out,raw_writer=None):
    model=base_model(run,world,registry);memory=empty_memory();rng=rng_for(run,50)
    state=make_states(rng,1);ctxnoise=ar(rng,1,TOTAL_BLOCKS*L)[0];sensor=rng.normal(size=(TOTAL_BLOCKS*L,3));process=rng.normal(size=(TOTAL_BLOCKS*L,3))
    g=0.;pending=None;cooldown=0;events=[];attempts=[];blocklog=[];stages=[];checkpoints=[];memory_details=[];eid=0;commits=0;merges=0
    raw_context=[];state_code=0;commit_snapshots=[]
    def capture_commit(index):
        snap={'V':model.V.copy(),'occupied':model.occupied.copy(),'slot_age':model.slot_age.copy(),'route_mu':np.array([r['mu'] for r in model.routes]),'route_precision':np.array([r['precision'] for r in model.routes]),'route_slot':np.array([r['slot'] for r in model.routes]),'route_sources':np.array([r['source_domain'] for r in model.routes]),'memory_ids':memory['id'].copy()}
        for k,r in enumerate(model.routes):snap[f'route_cal_{k}']=r['cal'].copy()
        commit_snapshots.append((index,snap))
    capture_commit(0)
    for block in range(TOTAL_BLOCKS):
        phase=block//PHASE_BLOCKS;domain=SEQUENCE[phase];before_hash=model.fingerprint();before_mem=memory_fingerprint(memory);reset=0;trigger=0;state_code=0
        if pending is not None:
            elapsed=block-pending['trigger_block']-1
            if elapsed<8:state_code=1
            elif elapsed<10:state_code=2
            else:state_code=3
            if elapsed==8:
                samples,reg=collect(model,world,domain,run,pending['trigger_block'])
                pending['proposal']=propose(model,memory,samples,reg,world,domain,run,pending['trigger_block'],variant)
                pending['training_domain']=domain
            if elapsed==12:
                proposal=pending['proposal'];finalgate,_,_=gate_metrics(model,proposal['candidate'],world,domain,run,block)
                accepted=proposal['accepted'] and bool(finalgate['passed']);old_active=int(model.occupied.sum());old_test=None
                # Independent test scores are diagnostic only; never enter gate.
                if accepted:
                    model=proposal['candidate'];memory=proposal['memory'];commits+=1;merges+=proposal['merged'];state_code=4;capture_commit(commits)
                else:cooldown=8;state_code=0
                reset=1;g=0.;eid+=1
                event={'run':run,'variant':variant,'event':eid,'trigger_block':pending['trigger_block'],'trigger_time':pending['trigger_block']+1,'decision_block':block,'decision_time':block,'trigger_phase':pending['trigger_phase'],'trigger_domain':pending['trigger_domain'],'training_domain':pending['training_domain'],'decision_domain':domain,'passed':int(accepted),'merge_committed':int(accepted and proposal['merged']),'merge_proposed':proposal['merged'],'slots_before':old_active,'slots_after':int(model.occupied.sum()),'memory_before':pending['memory_before'],'memory_after':len(memory['id']),'model_before':before_hash,'model_after':model.fingerprint(),'memory_hash_before':before_mem,'memory_hash_after':memory_fingerprint(memory),'registry_after':len(model.routes),'pair_i':proposal['pair_i'],'pair_j':proposal['pair_j'],'functional_distance':proposal['functional_distance'],'proposal_gate_passed':int(proposal['accepted']),'final_gate_passed':finalgate['passed']}
                event.update({k:finalgate[k] for k in ['historical_score_drop','historical_success_drop','current_gain','clean_drop']})
                events.append(event)
                attempts.extend([{'run':run,'variant':variant,'event':eid,**a} for a in proposal['attempt_logs']])
                if variant=='full':
                    pool=proposal['pool'];detail=proposal['detail'];n=len(pool['id'])
                    rows=np.column_stack([np.full(n,run),np.full(n,eid),pool['id'],pool['source'],pool['S'],detail['retention_risk'],pool['C'],detail['ranks'],detail['selected'].astype(int),detail['old_loss'],detail['new_loss'],np.full(n,int(accepted)),detail['features']])
                    memory_details.append(rows)
                pending=None
        # Uninterrupted online control. Policy changes only at a successful commit.
        contexts=[];frame_scores=[]
        for j in range(L):
            n=block*L+j;c0=world[domain]['context']+.4*ctxnoise[n]
            # A transient rendering anomaly does not change the control mapping.
            pulse=(.85*np.exp(-.5*((block+j/L-35.25)/.28)**2)+.65*np.exp(-.5*((block+j/L-(2*PHASE_BLOCKS+22.4))/.23)**2))
            c0=c0.copy();c0[4]+=pulse;c=norm(c0[None]);X=observation(state,np.array([domain]),world,sensor[n:n+1]);action=model.predict(X,c,True)
            diagnostic=float(mahal(c,registry['mu'],registry['precision'])[0]);score=float(tracking(state)[0]);bad=int(failed(state)[0]);contexts.append(c[0]);frame_scores.append(score)
            if raw_writer is not None:raw_writer.writerow([run,n,n/L,phase,domain,*state[0].tolist(),*action[0].tolist(),diagnostic,score,bad,state_code,*c[0].tolist()])
            state=dynamics(state,action,process[n:n+1])
            if failed(state)[0]:state=make_states(rng,1)
        z=np.mean(contexts,axis=0);ps=[float(pvalue(mahal(z,r['mu'],r['precision']),r['cal'])) for r in model.routes];p=max(ps)
        if not reset:g=max(0,g-np.log(p)-KAPPA)
        if variant!='frozen' and pending is None and cooldown<=0 and g>threshold:
            trigger=1;pending={'trigger_block':block,'trigger_phase':phase,'trigger_domain':domain,'memory_before':len(memory['id'])};state_code=1
        if cooldown>0:cooldown-=1
        blocklog.append([run,variant,block,phase,domain,float(np.mean([mahal(np.array(contexts),registry['mu'],registry['precision'])])),p,g,threshold,trigger,reset,state_code,int(model.occupied.sum()),len(memory['id']),len(model.routes),float(np.mean(frame_scores)),commits,merges,model.fingerprint(),memory_fingerprint(memory)])
        assert model.occupied.sum()<=M and len(memory['id'])<=BUDGET
        if not reset:assert model.fingerprint()==before_hash and memory_fingerprint(memory)==before_mem
        if reset and not events[-1]['passed']:assert before_hash==model.fingerprint() and before_mem==memory_fingerprint(memory)
        if (block+1)%PHASE_BLOCKS==0:
            evaluation=evaluate(model,world,list(range(9)),run,90)
            for testdomain,(score,sr) in evaluation.items():stages.append([run,variant,phase,domain,testdomain,score,sr,int(model.occupied.sum()),len(memory['id']),commits,merges,model.fingerprint(),memory_fingerprint(memory)])
            checkpoints.append({'phase':phase,'V':model.V.copy(),'occupied':model.occupied.copy(),'route_mu':np.array([r['mu'] for r in model.routes]),'route_precision':np.array([r['precision'] for r in model.routes]),'route_slot':np.array([r['slot'] for r in model.routes]),'route_sources':np.array([r['source_domain'] for r in model.routes]),'memory_ids':memory['id'].copy()})
    if pending is not None:
        # Unfinished proposals are not committed at the simulation horizon.
        pass
    save={'base':model.base,'U':model.U}
    for index,snap in commit_snapshots:
        for k,v in snap.items():save[f'commit{index}_{k}']=v
    for c in checkpoints:
        phase=c.pop('phase')
        for k,v in c.items():save[f'phase{phase}_{k}']=v
    np.savez_compressed(out/f'data/checkpoint_run{run:02d}_{variant}.npz',**save)
    return {'stages':stages,'blocks':blocklog,'events':events,'attempts':attempts,'memory_details':memory_details,'unfinished':int(pending is not None)}

def csv_write(path,headers,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)

def run_one(run,registry,threshold,out):
    world=worlds(run);results={}
    with gzip.open(out/f'data/online_frames_full_run{run:02d}.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['run','frame','time_block_units','phase','true_domain','cross_track','heading','speed','target_speed','curvature','steering','acceleration','raw_diagnostic','tracking_score','failure','state']+[f'context_{i}' for i in range(8)])
        for variant in CONFIG['variants']:results[variant]=simulate(run,variant,world,registry,threshold,out,writer if variant=='full' else None)
    return run,results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'mechanism-sim')
    a = parser.parse_args()
    if a.runs < 1:
        parser.error('--runs must be at least 1.')
    out = a.output
    if out.exists():
        parser.error(f'Choose a new output directory: {out}')
    (out / 'data').mkdir(parents=True)
    CONFIG['runs']=a.runs;CONFIG['engine_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();CONFIG['heldout_clean_episodes']=300
    (out/'simulation_config.json').write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding='utf-8')
    registry=initial_registry();print('Calibrating detector on independent clean episodes...',flush=True);threshold=calibrate_threshold(registry,out);print('Threshold',threshold,flush=True)
    stages=[];blocks=[];events=[];attempts=[];memories=[];start=time.time()
    unfinished=0
    with ProcessPoolExecutor(max_workers=min(3,a.runs)) as executor:
        futures=[executor.submit(run_one,run,registry,threshold,out) for run in range(1,a.runs+1)]
        for future in as_completed(futures):
            run,results=future.result()
            for variant,result in results.items():
                stages.extend(result['stages']);blocks.extend(result['blocks']);events.extend(result['events']);attempts.extend(result['attempts']);memories.extend(result['memory_details'])
                unfinished+=result['unfinished']
                print(f'Run {run}/{a.runs} {variant}: events={len(result["events"])}, accepted={sum(e["passed"] for e in result["events"])}, merges={sum(e["merge_committed"] for e in result["events"])}, elapsed={time.time()-start:.1f}s',flush=True)
    stages.sort(key=lambda r:(r[0],CONFIG['variants'].index(r[1]),r[2],r[4]));blocks.sort(key=lambda r:(r[0],CONFIG['variants'].index(r[1]),r[2]));events.sort(key=lambda e:(e['run'],CONFIG['variants'].index(e['variant']),e['event']));attempts.sort(key=lambda e:(e['run'],CONFIG['variants'].index(e['variant']),e['event'],e['attempt']))
    csv_write(out/'data/stage_domain_tests.csv',['run','variant','phase','presented_domain','test_domain','tracking_score','success_rate','occupied_slots','memory_count','commits','merges','model_hash','memory_hash'],stages)
    csv_write(out/'data/block_log.csv',['run','variant','block','phase','true_domain','raw_diagnostic_mean','union_p','evidence','threshold','trigger','reset','state','occupied_slots','memory_count','accepted_registry_count','online_tracking_mean','commits','merges','model_hash','memory_hash'],blocks)
    if events:csv_write(out/'data/update_events.csv',list(events[0]),[[plain(e[k]) for k in events[0]] for e in events])
    if attempts:csv_write(out/'data/gate_attempts.csv',list(attempts[0]),[[plain(e[k]) for k in attempts[0]] for e in attempts])
    memory=np.vstack(memories) if memories else np.empty((0,27))
    if len(memory):memory=memory[np.lexsort((memory[:,2],memory[:,1],memory[:,0]))]
    np.savez_compressed(out/'data/memory_candidates.npz',rows=memory)
    csv_write(out/'data/domain_definitions.csv',['domain','e_bias','heading_bias','speed_gain']+[f'context_mean_{i}' for i in range(8)],[[w['domain'],w['e_bias'],w['heading_bias'],w['speed_gain'],*w['context']] for w in worlds(1)])
    summary={'runs':a.runs,'threshold':threshold,'allocated_parameters':ALLOCATED,'stage_test_records':len(stages),'block_records':len(blocks),'memory_candidate_records':len(memory),'online_frames_full':a.runs*TOTAL_BLOCKS*L,'unfinished_proposals_at_horizon':unfinished,'variants':{}}
    for variant in CONFIG['variants']:
        rr=[r for r in stages if r[1]==variant and r[2]==len(SEQUENCE)-1];ev=[e for e in events if e['variant']==variant]
        summary['variants'][variant]={'final_mean_tracking_score':float(np.mean([r[5] for r in rr])),'final_mean_success_rate':float(np.mean([r[6] for r in rr])),'events':len(ev),'commits':sum(e['passed'] for e in ev),'merges':sum(e['merge_committed'] for e in ev),'rejected_updates':sum(not e['passed'] for e in ev)}
    (out/'simulation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'data/stage_arrays.npz',records=np.array([[r[0],CONFIG['variants'].index(r[1]),*r[2:11]] for r in stages],float))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
