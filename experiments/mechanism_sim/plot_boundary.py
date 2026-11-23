from pathlib import Path
import json,gzip,csv,re
import numpy as np
import improve,long_boundary
OUT=long_boundary.OUT;OLD=improve.OLD;SRC=improve.OUT
RECORDS=Path(__file__).with_name('records')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PLOTS=OUT/'figures';PLOTS.mkdir(parents=True,exist_ok=True)
for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_path = Path('C:/Windows/Fonts') / name
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
font = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
plt.rcParams.update({'font.family':font,'font.size':8,'axes.titlesize':8.7,'axes.labelsize':7.8,'xtick.labelsize':6.6,'ytick.labelsize':6.6,'axes.linewidth':.65,'axes.labelpad':2,'lines.linewidth':1.05,'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.pad':2,'ytick.major.pad':2,'legend.fontsize':5.8,'legend.frameon':False,'text.color':'#222222','axes.labelcolor':'#222222','axes.edgecolor':'#444444','legend.borderpad':.25,'legend.labelspacing':.23,'mathtext.fontset':'custom','mathtext.rm':font,'mathtext.it':font + ':italic','mathtext.bf':font + ':bold','mathtext.sf':font,'mathtext.tt':font,'mathtext.cal':font + ':italic','mathtext.fallback':None,'svg.fonttype':'none','pdf.fonttype':42,'savefig.pad_inches':.03})
BLUE,RED,PURPLE,GRAY,GREEN,GOLD='#0072B2','#D55E00','#CC79A7','#7E868D','#009E73','#B87900'
colors=[BLUE,RED,GREEN,GRAY];names=['Full system','Random replay','Slot replacement','Frozen policy'];fonts={};stats={}

def bootstrap(a):
    a=np.asarray(a);r=np.random.default_rng(20261024);idx=r.integers(0,len(a),(6000,len(a)));mu=a.mean(0);lo,hi=np.quantile(a[idx].mean(1),[.025,.975],axis=0);return mu,lo,hi
def error(a):
    mu,lo,hi=bootstrap(a);return mu,np.array([mu-lo,hi-mu])
def clean(ax):ax.grid(False);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
def title(ax,l,t):ax.set_title(f'({l}) {t}',fontweight='bold',loc='center',pad=4)
def make(n,t,size):
    f=plt.figure(figsize=size,layout='constrained');f.set_constrained_layout_pads(w_pad=.025,h_pad=.025,wspace=.035,hspace=.04);f.suptitle(f'Fig. {n}. {t}',fontweight='bold',fontsize=10.6);return f
def save(f,n):
    for ext in ['png','pdf','svg']:f.savefig(PLOTS/f'Fig{n}_BoundaryRevision.{ext}',dpi=600,bbox_inches='tight')
    ff=sorted({x.decode('ascii') for x in re.findall(rb'/BaseFont\s*/([^\s/<>]+)',(PLOTS/f'Fig{n}_BoundaryRevision.pdf').read_bytes())});assert ff and all(font.replace(' ', '') in x for x in ff),ff
    fonts[str(n)]=ff;f.savefig(OUT/f'preview_{n}.png',dpi=140,bbox_inches='tight');plt.close(f)
def readcsv(name):
    with (RECORDS/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def load_doc(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

diag=json.loads((RECORDS/'protocol_diagnostics.json').read_text(encoding='utf-8'));orig=np.load(OLD/'data/plot_arrays.npz');score=orig['tracking_score'];success=orig['success'];n=12
conditions={}
for cond in ['coupled','appearance','hidden','gradual']:
    for run in range(1,13):conditions[cond,run]=load_doc(SRC/'runs'/f'condition_{cond}_full_{run:02d}'/'result.json.gz')
budget_names=[c[0] for c in long_boundary.CONFIGS];long_docs={}
for path in (OUT/'runs').glob('*/result.json.gz'):
    d=load_doc(path);long_docs[d['job'][0],d['job'][-1]]=d
assert len(long_docs)==84,len(long_docs)
ls=np.zeros((12,7,18,15));lr=ls.copy();occupancy=np.zeros((12,7,18));resources={key:occupancy.copy() for key in ['network_bytes','router_detector_bytes','replay_bytes','total_bytes']}
for (name,run),doc in long_docs.items():
    m=budget_names.index(name)
    for r in doc['stages']:ls[run-1,m,r[2],r[4]]=r[5];lr[run-1,m,r[2],r[4]]=r[6];occupancy[run-1,m,r[2]]=r[7]
    for r in doc['resources']:
        for key in resources:resources[key][run-1,m,r['phase']]=r[key]
    for ev in doc['events']:
        if not ev['passed']:assert ev['model_before']==ev['model_after'] and ev['memory_hash_before']==ev['memory_hash_after']
    assert all(r[7]<=doc['job'][1] and r[8]<=doc['job'][2] for r in doc['stages'])
current=np.stack([ls[:,:,p,k] for p,k in enumerate(long_boundary.SEQUENCE)],axis=2)
historical=np.zeros((12,7,18))
for phase in range(1,18):
    domains=sorted(set(long_boundary.SEQUENCE[:phase]));drops=[]
    for domain in domains:
        first=long_boundary.SEQUENCE.index(domain);drops.append(ls[:,:,first:phase,domain].max(2)-ls[:,:,phase,domain])
    historical[:,:,phase]=np.maximum(0,np.stack(drops,axis=-1).max(-1))

# Fig7: original outcomes, explicit stage commits, and no invented explanation of plateaus.
fig=make(7,'Nine-domain adaptation and order sensitivity',(7.4,5.7));gs=fig.add_gridspec(2,3,width_ratios=[4.4,.13,1.9],height_ratios=[3.2,1.5]);a=fig.add_subplot(gs[0,0]);cb=fig.add_subplot(gs[0,1]);b=fig.add_subplot(gs[0,2]);c=fig.add_subplot(gs[1,:2]);d=fig.add_subplot(gs[1,2])
mu=score[:,0].mean(0);sr=success[:,0].mean(0);im=a.imshow(mu,cmap='cividis_r',vmin=0,vmax=1,aspect='auto')
def ink(v):
    rgb=np.array(im.cmap(v)[:3]);lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);lum=lin@np.array([.2126,.7152,.0722]);return 'white' if 1.05/(lum+.05)>(lum+.05)/.05 else '#171717'
for i in range(12):
    for j in range(9):a.text(j,i,f'{mu[i,j]:.2f}/{sr[i,j]:.2f}',fontsize=5.4,ha='center',va='center',color=ink(mu[i,j]))
counts=diag['canonical_phase_counts_30runs'];a.set_yticks(range(12),[f'P{r["phase"]}: D{r["domain"]} [{r["commits"]}]' for r in counts],fontsize=6);a.set_xticks(range(9),[f'D{i}' for i in range(9)]);a.set_xlabel('Heldout test domain');a.set_ylabel('Phase / domain [commits across 30 runs]');title(a,'a','Tracking score / task success (30 seeds)');fig.colorbar(im,cax=cb,label='Tracking score')
for m in range(4):
    v,lo,hi=bootstrap(score[:,m].mean(2));b.plot(v,np.arange(12),color=colors[m],ls=['-','--','-.',':'][m],label=names[m]);b.fill_betweenx(np.arange(12),lo,hi,color=colors[m],alpha=.09)
b.set_ylim(11.5,-.5);b.set_yticks([0,3,6,8,11],['1','4','7','9','12']);b.set_xlabel('All-domain mean');b.set_ylabel('Phase');b.legend(loc='lower left',fontsize=5.6);clean(b);title(b,'b','Stage-wise performance')
for m in range(4):
    v,er=error(score[:,m,-1].min(1));c.errorbar(m,v,yerr=er.reshape(2,1),fmt=['o','s','^','D'][m],color=colors[m],capsize=3,ms=5)
c.set_xticks(range(4),['Full','Random','Replacement','Frozen']);c.set_ylim(0,1.04);c.set_ylabel('Final worst-domain score');clean(c);title(c,'c','Worst-domain robustness (30 seeds)')
orderdiff=[]
for j,name in enumerate(improve.ORDERS):
    ff=[];rr=[]
    for run in range(1,13):
        doc=load_doc(SRC/'runs'/f'order_{name}_full_{run:02d}'/'result.json.gz');ff.append(np.mean([r[5] for r in doc['stages'] if r[2]==11]))
        if name=='default':rr.append(score[run-1,1,-1].mean())
        else:
            doc=load_doc(SRC/'runs'/f'order_{name}_random_memory_{run:02d}'/'result.json.gz');rr.append(np.mean([r[5] for r in doc['stages'] if r[2]==11]))
    v=np.array(ff)-rr;orderdiff.append(v);mean,er=error(v);d.scatter(j+np.linspace(-.09,.09,12),v,s=10,color=BLUE,alpha=.45,edgecolors='none');d.errorbar(j,mean,yerr=er.reshape(2,1),fmt='D',color=BLUE,ecolor='#333333',ms=4,capsize=2.5)
d.axhline(0,color=GRAY,ls='--',lw=.7);d.set_xticks(range(3),['Original','Alternating','Permuted'],fontsize=6);d.set_ylabel('Full − random score');clean(d);title(d,'d','Order effects (12 paired seeds)');save(fig,7)

# Fig8: six necessary panels replace a fifth figure; 15 domains / 18 phases.
fig=make(8,'Long-term capacity boundaries and gated adaptation',(7.4,7.0));axs=fig.subplots(3,2);a,b,c,d,e,f=axs.flat;x=np.arange(1,19);base=budget_names.index('M4_B400')
for key,label,col,lsline in [('network_bytes','Network (1,654 scalars)',BLUE,'-'),('replay_bytes','Persistent replay',GREEN,'--'),('router_detector_bytes','Router / detector state',GOLD,'-.'),('total_bytes','Total numeric payload',GRAY,'-')]:
    a.plot(x,resources[key][:,base].mean(0)/1024,color=col,ls=lsline,label=label,lw=1.05)
ar=a.twinx();ar.step(x,occupancy[:,base].mean(0),where='mid',color='#333333',ls=':',lw=.8,label='Occupied slots');ar.set_ylim(0,5);ar.set_yticks([0,2,4]);ar.set_ylabel('Occupied slots',fontsize=7.1);ar.spines['top'].set_visible(False);a.set_xlabel('Deployment phase');a.set_ylabel('Persistent array payload (KiB)');a.set_xticks([1,5,10,15,18]);clean(a);a.legend(loc='center right',ncol=2,fontsize=5.5,handlelength=1.6);ar.legend(loc='lower right',fontsize=5.5);title(a,'a','Resource audit: M = 4, B = 400')
pair_audit=json.loads((RECORDS/'all_pair_audit.json').read_text(encoding='utf-8'));pf=np.array(pair_audit['functional']);pu=np.array(pair_audit['uniform_expectation'])
for vv,uu in zip(pf,pu):b.plot([0,1],[vv,uu],color='#c8cdd1',lw=.6)
for j,(vv,col,label) in enumerate([(pf,BLUE,'Functional choice'),(pu,GRAY,'Uniform-pair expectation')]):
    mean,er=error(vv);b.scatter(j+np.linspace(-.07,.07,12),vv,s=8,color=col,alpha=.45);b.errorbar(j,mean,yerr=er.reshape(2,1),fmt='D',color=col,capsize=2.5,ms=4.2)
b.set_xticks([0,1],['Functional choice','Uniform-pair expectation'],fontsize=5.8);b.set_ylabel('Worst historical score drop');clean(b);title(b,'b','Matched canonical checkpoints: all 72 pairs')
for name,col,style in [('M2_B400',GRAY,':'),('M4_B200',RED,'--'),('M4_B400',BLUE,'-'),('M4_B800',GREEN,'-.'),('M6_B400',PURPLE,(0,(5,2)))]:
    idx=budget_names.index(name);mean,lo,hi=bootstrap(current[:,idx]);c.plot(x,mean,color=col,ls=style,label=f'{long_boundary.CONFIGS[idx][1]} / {long_boundary.CONFIGS[idx][2]}');c.fill_between(x,lo,hi,color=col,alpha=.055)
    sat=[np.flatnonzero(occupancy[r,idx]>=long_boundary.CONFIGS[idx][1])[0]+1 for r in range(12)];median=float(np.median(sat));c.plot(median,np.interp(median,x,mean),marker='o',mfc='white',mec=col,ms=3.5,mew=.9)
c.axvspan(15.5,18.5,color='#eeeeee',alpha=.5);c.set_xlim(1,18);c.set_xticks([1,5,10,15,18]);c.set_xlabel('Deployment phase');c.set_ylabel('Current-domain tracking score');c.legend(loc='best',title='M / B; open point: median first-full phase',title_fontsize=5.2,ncol=3,fontsize=5.3);clean(c);title(c,'c','Long-horizon one-factor budget sweep')
gate_modes=[('M4_B400','Nominal',BLUE,'-'),('M4_B400_strict','Strict',PURPLE,'--'),('M4_B400_no_gate','Disabled',GRAY,':')]
rates=[]
for name,label,col,style in gate_modes:
    rej=np.zeros((12,18));den=np.zeros_like(rej);merged=0;total=0
    for run in range(1,13):
        for ev in long_docs[name,run]['events']:
            if ev['slots_before']!=4:continue
            phase=ev['decision_block']//64;den[run-1,phase]+=1;rej[run-1,phase]+=1-ev['passed'];merged+=ev['merge_committed'];total+=1
    mu=np.divide(rej.sum(0),den.sum(0),out=np.full(18,np.nan),where=den.sum(0)>0);d.plot(x,mu,color=col,ls=style,marker='o',ms=2.3,label=f'{label} (merge {merged}/{total})')
    rates.append({'configuration':name,'saturated_completed':total,'committed_merges':merged,'rejected':int(rej.sum()),'phase_denominators':den.sum(0).tolist(),'phase_rejections':rej.sum(0).tolist()})
d.set_ylim(-.035,1.08);d.set_xlim(1,18);d.set_xticks([1,5,10,15,18]);d.set_xlabel('Deployment phase');d.set_ylabel('Rejected / saturated proposals');d.legend(loc='best',fontsize=5.4);clean(d);title(d,'d','Saturated-pool decisions: M = 4, B = 400')
for name,label,col,style in gate_modes:
    idx=budget_names.index(name);mean,lo,hi=bootstrap(historical[:,idx]);e.plot(x,mean,color=col,ls=style,label=label);e.fill_between(x,lo,hi,color=col,alpha=.08)
e.axvspan(15.5,18.5,color='#eeeeee',alpha=.5);e.set_xlim(1,18);e.set_xticks([1,5,10,15,18]);e.set_xlabel('Deployment phase');e.set_ylabel('Worst cumulative historical score loss',fontsize=7.1);e.legend(loc='best');clean(e);title(e,'e','Historical cost of Gate policies')
audit=readcsv('canonical_gate_clause_audit.csv');branch=[]
for run in range(1,13):
    rr=[r for r in audit if int(r['run'])==run];force_loss=np.mean([max(0,float(r['audit_history_drop'])) for r in rr]);gate_loss=np.mean([max(0,float(r['audit_history_drop'])) if int(r['passed']) else 0 for r in rr]);force_gain=np.mean([float(r['audit_current_gain']) for r in rr]);gate_gain=np.mean([float(r['audit_current_gain']) if int(r['passed']) else 0 for r in rr]);branch.append([force_loss,gate_loss,force_gain,gate_gain]);f.plot([force_loss,gate_loss],[force_gain,gate_gain],color='#c8cdd1',lw=.65)
branch=np.array(branch)
for k,col,label,marker in [(0,RED,'Force candidate','D'),(1,BLUE,'Apply Gate','o')]:
    xx=branch[:,k];yy=branch[:,k+2];f.scatter(xx,yy,s=9,color=col,alpha=.4);mx,xe=error(xx);my,ye=error(yy);f.errorbar(mx,my,xerr=xe.reshape(2,1),yerr=ye.reshape(2,1),fmt=marker,color=col,capsize=2.2,ms=4.5,label=label)
f.set_xlabel('Mean worst-history score loss');f.set_ylabel('Mean current-domain score gain');f.legend(loc='best',fontsize=5.6);clean(f);title(f,'f','Same-event benefit–cost audit (110 proposals)');save(fig,8)

# Fig9: make C explicit and replace a redundant learning curve with a selection-composition audit.
fig=make(9,'Replay selection, coverage and downstream outcomes',(7.4,5.2));axs=fig.subplots(2,2);a,b,c,d=axs.flat
ref=np.load(RECORDS/'critical_event_reference.npz');pool=ref['rows'];selected=ref['selected'];S,R,C=pool[:,1],pool[:,2],pool[:,3];front=pool[:,4]==1
for kind,marker in [(0,'o'),(1,'^')]:
    mask=C==kind
    if mask.any():a.scatter(S[mask],R[mask],s=9,marker=marker,color='#aeb4b9',alpha=.35,label=f'Candidates C={kind} ({mask.sum()})')
    else:a.plot([],[],marker=marker,ls='none',color='#aeb4b9',ms=4,label=f'Candidates C={kind} (0)')
    z=mask&front;a.scatter(S[z],R[z],marker=marker,s=17,facecolors='none',edgecolors=RED,lw=.6,label='First 3-D front' if kind==0 else None)
    z=mask&selected;a.scatter(S[z],R[z],marker=marker,s=16,facecolors='none',edgecolors=BLUE,lw=.45,label='Selected replay' if kind==0 else None)
a.set_xscale('symlog',linthresh=.0001);a.set_yscale('symlog',linthresh=.0001);a.set_xlabel('Observed safety severity');a.set_ylabel('Adaptation-induced retention risk');a.legend(loc='best',fontsize=5.6);clean(a);title(a,'a','C-aware selection: run 5, event 10')
comp=readcsv('same_pool_domain_retention.csv');variants=['full','random_memory','coverage_only','pareto_only'];labels=['Pareto + coverage','Random','Coverage only','Pareto only'];mc=[BLUE,RED,GREEN,PURPLE];domain_ids=sorted({int(r['domain']) for r in comp});xx=np.arange(len(domain_ids))
for m,v in enumerate(variants):
    mat=np.full((12,len(domain_ids)),np.nan)
    for row in comp:
        if row['variant']==v:mat[int(row['run'])-1,domain_ids.index(int(row['domain']))]=float(row['retention_fraction'])
    mean=np.nanmean(mat,0);b.plot(xx,mean,color=mc[m],ls=['-','--','-.',':'][m],marker=['o','s','^','D'][m],ms=3,label=labels[m])
domain_ns={domain:len({int(r['run']) for r in comp if int(r['domain'])==domain}) for domain in domain_ids}
b.set_xticks(xx,[f'D{i}\nn={domain_ns[i]}' for i in domain_ids]);b.set_ylim(0,1.05);b.set_xlabel('Source domain; n: checkpoints containing domain',fontsize=7.1);b.set_ylabel('Selected / available samples');b.legend(loc='best',fontsize=5.6);clean(b);title(b,'b','Same-pool domain retention (12 checkpoints)')
ms=np.zeros((12,4,12,9));mr=ms.copy()
for run in range(1,13):
    for m,v in enumerate(variants):
        if v=='random_memory':ms[run-1,m]=score[run-1,1];mr[run-1,m]=success[run-1,1]
        else:
            doc=load_doc(SRC/'runs'/f'order_default_{v}_{run:02d}'/'result.json.gz')
            for row in doc['stages']:ms[run-1,m,row[2],row[4]]=row[5];mr[run-1,m,row[2],row[4]]=row[6]
for k,(vals,label,mark,off) in enumerate([(ms[:,:,-1].mean(2),'All-domain mean','o',-.10),(ms[:,:,-1].min(2),'Worst domain','s',.10)]):
    for j,m in enumerate([0,2,3]):
        mean,er=error(vals[:,m]-vals[:,1]);c.errorbar(j+off,mean,yerr=er.reshape(2,1),fmt=mark,color=mc[m],mfc=mc[m] if k==0 else 'white',capsize=2,ms=4,label=label if j==0 else None)
c.axhline(0,color=GRAY,ls='--',lw=.7);c.set_xticks(range(3),['Pareto +\ncoverage','Coverage\nonly','Pareto\nonly']);c.set_ylabel('Paired score difference to random');c.legend(loc='lower left',fontsize=5.8);clean(c);title(c,'c','Downstream effects relative to random')
for m in range(4):
    mean,er=error(100*(1-mr[:,m,-1].mean(1)));d.errorbar(m,mean,yerr=er.reshape(2,1),fmt=['o','s','^','D'][m],color=mc[m],capsize=2.5,ms=4.5)
d.set_xticks(range(4),['Pareto +\ncoverage','Random','Coverage\nonly','Pareto\nonly']);d.set_ylim(bottom=0);d.set_ylabel('Final unsuccessful episodes (%)');clean(d);title(d,'d','Independent closed-loop task failures');save(fig,9)

# Fig10: show an actually triggered example without the identical orange curve occluding it.
fig=make(10,'Event-consistent drift detection and deployment',(7.4,6.4));gs=fig.add_gridspec(3,2,height_ratios=[1.0,1.0,1.2]);a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:],sharex=a);c=fig.add_subplot(gs[2,0]);d=fig.add_subplot(gs[2,1])
for cond,col,label,style in [('coupled',BLUE,'Coupled shift','-'),('hidden',PURPLE,'Weak appearance','-.')]:
    doc=conditions[cond,1];tt=[];gg=[]
    for row in doc['blocks'][:128]:
        if row[10]:tt.append(row[2]);gg.append(0.)
        tt.append(row[2]+1);gg.append(row[7]/row[8])
    a.plot(tt,gg,color=col,ls=style,label=label,lw=1.2)
    b.plot(np.arange(128)+.5,[r[15] for r in doc['blocks'][:128]],color=col,ls=style,label=label,lw=1.2)
trace=diag['run1_event_trace']['coupled']['events'][0];onset=64;trigger=trace['trigger'];candidate=trace['candidate_ready'];validate=trace['validation_state_start'];commit=trace['final_gate_and_commit_decision']
assert (trigger,candidate,validate,commit)==(66,74,76,78)
for ax in [a,b]:
    ax.axvline(onset,color=GRAY,ls=':',lw=.75);ax.axvline(trigger,color=BLUE,ls='--',lw=.7);ax.axvline(candidate,color=GOLD,ls='--',lw=.7);ax.axvline(commit,color=GREEN,ls='--',lw=.8);ax.set_xlim(0,128);clean(ax)
a.axhline(1,color=GRAY,ls='--',lw=.8,label='Trigger threshold');a.plot([],[],color=BLUE,ls='--',label=f'Trigger: {trigger}');a.plot([],[],color=GOLD,ls='--',label=f'Candidate: {candidate}');a.plot([],[],color=GREEN,ls='--',label=f'Gate / commit: {commit}');a.legend(loc='upper left',ncol=3,fontsize=5.8);a.set_ylabel('Evidence / threshold');a.tick_params(labelbottom=False);title(a,'a','Run 1: evidence, candidate generation and commit')
b.axvspan(trigger,candidate,color=BLUE,alpha=.08);b.axvspan(candidate,validate,color=GOLD,alpha=.12);b.axvspan(validate,commit,color=PURPLE,alpha=.14);b.set_ylim(.55,1.02);b.set_ylabel('Deployed tracking score');b.set_xlabel('Time (blocks)');handles,lab=b.get_legend_handles_labels();handles+=[Patch(facecolor=BLUE,alpha=.12,label='Collect'),Patch(facecolor=GOLD,alpha=.17,label='Fit'),Patch(facecolor=PURPLE,alpha=.19,label='Validate')];b.legend(handles=handles,loc='lower left',ncol=5,fontsize=5.8);title(b,'b','Run 1: deployed control on the same time axis')
single=float(np.load(SRC/'single_block_calibration.npz')['threshold'])
def window(doc,start,end):
    bl=doc['blocks'][start:end];seq=[r[2]+1-start for r in bl if r[9]];instant=[r[2]+1-start for r in bl if r[-1]>single];commits=[ev for ev in doc['events'] if ev['passed'] and start<=ev['trigger_block']<end and ev['decision_block']<end]
    return [bool(seq),bool(instant),bool(commits),seq[0] if seq else 64,instant[0] if instant else 64]
conds=['appearance','hidden','coupled','gradual'];vals={name:np.array([window(conditions[name,run],64,128) for run in range(1,13)]) for name in conds};normal=np.array([window(conditions['coupled',run],0,64) for run in range(1,13)])
prob=np.stack([normal.mean(0),*[vals[name].mean(0) for name in conds]])[:,:3];xx=np.arange(5)
def wilson(y):
    z=1.96;center=(y+z*z/24)/(1+z*z/12);half=z*np.sqrt(y*(1-y)/12+z*z/(4*144))/(1+z*z/12);return center-half,center+half
for i,(col,mark,label,off) in enumerate([(BLUE,'o','Sequential trigger',-.22),(GRAY,'s','Single-block trigger',0),(GREEN,'^','Validated commit',.22)]):
    y=prob[:,i];lo,hi=wilson(y);c.errorbar(xx+off,y,yerr=np.maximum(0,np.array([y-lo,hi-y])),fmt=mark,color=col,mfc='white' if i==1 else col,ms=3.8,capsize=1.5,lw=.8,label=label)
c.set_xticks(xx,['Normal +\ntransient','Appearance\nonly','Weak\nappearance','Coupled','Gradual'],fontsize=5.9);c.set_ylim(-.035,1.18);c.set_ylabel('Probability within 64 blocks');c.legend(loc='upper left',fontsize=5.5);clean(c);title(c,'c','Trigger and commit probabilities (12 seeds)')
for off,k,col,label,mark in [(-.13,3,BLUE,'Sequential','o'),(.13,4,GRAY,'Single block','s')]:
    for j,name in enumerate(conds):
        mean,er=error(vals[name][:,k]);d.errorbar(j+off,mean,yerr=er.reshape(2,1),fmt=mark,color=col,mfc='white' if k==4 else col,ms=4,capsize=2,lw=.8,label=label if j==0 else None)
d.axhline(64,color=RED,ls=':',lw=.7,label='64-block restriction');d.set_ylim(0,71);d.set_xticks(range(4),['Appearance\nonly','Weak\nappearance','Coupled','Gradual'],fontsize=5.9);d.set_ylabel(r'Restricted mean $E[\min(T,64)]$ (blocks)',fontsize=7.2);d.legend(loc='upper right',bbox_to_anchor=(1,.89),fontsize=5.5);clean(d);title(d,'d','Restricted time, including undetected runs');save(fig,10)

stats['long_summary']={name:{'final_score':float(ls[:,i,-1].mean()),'final_success':float(lr[:,i,-1].mean()),'final_worst_domain':float(ls[:,i,-1].min(1).mean()),'final_current_score':float(current[:,i,-1].mean()),'final_cumulative_historical_loss':float(historical[:,i,-1].mean()),'completed_proposals':sum(len(long_docs[name,r]['events']) for r in range(1,13)),'commits':sum(sum(ev['passed'] for ev in long_docs[name,r]['events']) for r in range(1,13)),'accepted_unique_domains':float(np.mean([len({0}|{ev['training_domain'] for ev in long_docs[name,r]['events'] if ev['passed']}) for r in range(1,13)])),'final_numeric_payload_KiB':float(resources['total_bytes'][:,i,-1].mean()/1024),'network_scalars':14+410*long_boundary.CONFIGS[i][1]} for i,name in enumerate(budget_names)}
stats['saturated_decision_rates']=rates;stats['canonical_same_event_tradeoff']={'force_history_loss':float(branch[:,0].mean()),'gate_history_loss':float(branch[:,1].mean()),'force_current_gain':float(branch[:,2].mean()),'gate_current_gain':float(branch[:,3].mean()),'paired_gain_change95':np.array(bootstrap(branch[:,3]-branch[:,2])).tolist()};stats['run1_trace']=trace;stats['same_pool_composition']=diag['selection_summary'];stats['canonical_gate_denominators']=diag['canonical_gate_denominators'];stats['composition_checkpoints_per_domain']=domain_ns
(OUT/'figure_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'font_audit.json').write_text(json.dumps(fonts,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'data/long_plot_arrays.npz',score=ls,success=lr,current=current,historical=historical,occupancy=occupancy,**resources)
print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
