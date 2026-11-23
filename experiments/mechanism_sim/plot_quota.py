from pathlib import Path
import argparse
import json
import engine
import quota_revision as q

parser = argparse.ArgumentParser(description="Rebuild final synthetic mechanism figures from simulation artifacts.")
parser.add_argument('--records', type=Path, required=True, help='Local audit and replay records')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
RECORDS = args.records
FIGURE_OUTPUT = args.output
FIGURE_OUTPUT.mkdir(parents=True, exist_ok=False)
(FIGURE_OUTPUT / 'data').mkdir()

# Reuse the audited data loading, style, resource and detector panels verbatim.
src=(Path(__file__).parent/'plot_boundary.py').read_text(encoding='utf-8')
head=src.split('# Fig7:')[0]
head=head.replace("PLOTS=OUT/'figures'","PLOTS=FIGURE_OUTPUT/'figures'")
head=head.replace("Fig{n}_BoundaryRevision", "Fig{n}_MechanismRevision")
head=head.replace("OUT/f'preview_{n}.png'", "FIGURE_OUTPUT/f'preview_{n}.png'")
exec(compile(head,'<audited-data-and-style>','exec'))
variants=['full','random_memory','quota_risk','quota_random','coverage_only','pareto_only']
labels=['Original Pareto + coverage','Random replay','Quota + Pareto + coverage','Quota + random','Coverage only','Pareto only']
cols=[BLUE,RED,GREEN,PURPLE,GRAY,GOLD]
ms=np.zeros((12,6,12,9));mr=ms.copy();docs={};stage_records=[];event_records=[]
for r in range(1,13):
    for j,v in enumerate(variants):
        if v.startswith('quota'):doc=load_doc(q.OUT/'runs'/f'default_{v}_{r:02d}'/'result.json.gz')
        elif v=='random_memory':
            ms[r-1,j]=score[r-1,1];mr[r-1,j]=success[r-1,1]
            continue
        else:doc=load_doc(SRC/'runs'/f'order_default_{v}_{r:02d}'/'result.json.gz')
        docs[v,r]=doc
        for row in doc['stages']:ms[r-1,j,row[2],row[4]]=row[5];mr[r-1,j,row[2],row[4]]=row[6]
        if v.startswith('quota'):stage_records.extend(doc['stages']);event_records.extend(doc['events'])

fig=make(7,'Continual adaptation, retention and order effects',(8.1,6.0));gs=fig.add_gridspec(2,2,width_ratios=[1.35,1],height_ratios=[1.2,1]);a,b,c,d=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
mu=ms[:,0].mean(0);im=a.imshow(mu,cmap='cividis_r',vmin=0,vmax=1,aspect='auto')
counts=diag['canonical_phase_counts_30runs']
for i in range(12):
    for j in range(9):a.text(j,i,f'{mu[i,j]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if mu[i,j]>.48 else '#151515')
    ds=[docs['full',r] for r in range(1,13)];events=[ev for dd in ds for ev in dd['events'] if ev['decision_block']//64==i]
    slots=np.mean([next(row[7] for row in dd['stages'] if row[2]==i) for dd in ds]);commits=sum(ev['passed'] for ev in events);rejects=len(events)-commits
    a.text(8.62,i,f'{slots:.1f} / {commits} / {rejects}',ha='left',va='center',fontsize=5.2)
a.set_xlim(-.5,13);a.set_xticks(range(9),[f'D{i}' for i in range(9)]);a.set_yticks(range(12),[f'P{i+1}: D{k}'+(' (revisit)' if i>=9 else '') for i,k in enumerate(improve.ORDERS['default'])]);a.set_xlabel('Domain; side: mean slots / commits / rejects');a.set_ylabel('Deployment phase');title(a,'a','Original rule: scores and decisions (12 seeds)')
for j,sty in [(0,'--'),(1,'-'),(2,'-.')]:
    hist=np.array([[ms[r,j,p,list(sorted(set(improve.ORDERS['default'][:p])))].mean() if p else np.nan for p in range(12)] for r in range(12)])
    mean,lo,hi=bootstrap(hist);b.plot(np.arange(1,13),mean,color=cols[j],ls=sty,label=labels[j]);b.fill_between(np.arange(1,13),lo,hi,color=cols[j],alpha=.07)
cur=np.array([[ms[r,2,p,k] for p,k in enumerate(improve.ORDERS['default'])] for r in range(12)]);b.plot(np.arange(1,13),cur.mean(0),color=GREEN,ls=':',label='Quota: current domain');b.set_xlabel('Deployment phase');b.set_ylabel('Tracking score');b.legend(fontsize=5.3,loc='best');clean(b);title(b,'b','Exposed-history mean and current adaptation')
for j in [0,1,2,3]:
    xx=ms[:,j,-1].min(1);yy=ms[:,j,8,8];naccept=9 if j==1 else np.mean([len({0}|{ev['training_domain'] for ev in docs[variants[j],r]['events'] if ev['passed']}) for r in range(1,13)])
    xm,xe=error(xx);ym,ye=error(yy);c.errorbar(xm,ym,xerr=xe.reshape(2,1),yerr=ye.reshape(2,1),fmt='none',ecolor=cols[j],lw=.8);c.scatter(xm,ym,s=12*naccept,color=cols[j],alpha=.75,label=f'{labels[j]} (K={naccept:.2f})')
c.set_xlabel('Final worst-domain tracking score');c.set_ylabel('D8 score on first exposure (P9)');c.legend(fontsize=5.2,loc='best');clean(c);title(c,'c','Retention–adaptation trade-off; area = accepted domains')
order_summary={}
for k,order in enumerate(improve.ORDERS):
    rand=[];origv=[];quotav=[];counts=[]
    for r in range(1,13):
        dd=load_doc(q.OUT/'runs'/f'{order}_quota_risk_{r:02d}'/'result.json.gz');quotav.append(np.mean([x[5] for x in dd['stages'] if x[2]==11]));counts.append([sum(ev['merge_committed'] for ev in dd['events']),sum(not ev['passed'] for ev in dd['events'])])
        oo=load_doc(SRC/'runs'/f'order_{order}_full_{r:02d}'/'result.json.gz');origv.append(np.mean([x[5] for x in oo['stages'] if x[2]==11]))
        if order=='default':rand.append(ms[r-1,1,-1].mean())
        else:rr=load_doc(SRC/'runs'/f'order_{order}_random_memory_{r:02d}'/'result.json.gz');rand.append(np.mean([x[5] for x in rr['stages'] if x[2]==11]))
    for off,vals,col,lab in [(-.11,origv,BLUE,'Original − random'),(.11,quotav,GREEN,'Quota − random')]:
        diff=np.array(vals)-rand;mean,er=error(diff);d.errorbar(k+off,mean,yerr=er.reshape(2,1),fmt='o',color=col,ms=4,capsize=2,label=lab if k==0 else None)
    order_summary[order]={'quota_minus_random':np.array(bootstrap(np.array(quotav)-rand)).tolist(),'quota_mean_merges_rejects':np.mean(counts,axis=0).tolist()}
    d.text(k,.99,f'M/R: {np.mean(counts,axis=0)[0]:.1f}/{np.mean(counts,axis=0)[1]:.1f}',transform=d.get_xaxis_transform(),fontsize=5.3,ha='center',va='top')
d.axhline(0,color=GRAY,lw=.6,ls='--');d.set_xticks(range(3),['Original','Alternating','Permuted']);d.set_ylabel('Paired final score difference');d.legend(loc='lower left',fontsize=5.4);clean(d);title(d,'d','Order sensitivity (12 paired seeds)');save(fig,7)

body8=src.split('# Fig8:')[1].split('# Fig9:')[0]
body8=body8[body8.index('fig='):]
body8=body8.replace("Long-term capacity boundaries and gated adaptation","Capacity-constrained consolidation and validated deployment")
body8=body8.replace("title(b,'b','Matched canonical checkpoints: all 72 pairs')", "title(b,'b','Functional choice at 12 matched checkpoints');b.set_ylim(-.004,.058);b.text(.98,.97,f'{100*(1-pf.mean()/pu.mean()):.1f}% lower mean loss\\nGate: functional 11/12; all pairs 68/72',transform=b.transAxes,ha='right',va='top',fontsize=5.7)")
start=body8.index('for name,label,col,style in gate_modes:\n    idx=budget_names.index(name);mean,lo,hi=bootstrap(historical')
end=body8.index("audit=readcsv",start)
body8=body8[:start]+'''for name,label,col,style in gate_modes:
    idx=budget_names.index(name);xx=historical[:,idx,-1];yy=ls[:,idx,-1].min(1);xm,xe=error(xx);ym,ye=error(yy)
    e.errorbar(xm,ym,xerr=xe.reshape(2,1),yerr=ye.reshape(2,1),fmt='o',color=col,capsize=2,ms=4,label=f'{label}; revisit TS={current[:,idx,-1].mean():.2f}')
e.set_xlabel('Final cumulative historical loss');e.set_ylabel('Final worst-domain score');e.legend(loc='best',fontsize=5.1);clean(e);title(e,'e','Gate trade-off: retention, coverage and adaptation')
''' + body8[end:]
body8=body8.replace("title(f,'f','Same-event benefit–cost audit (110 proposals)')", "title(f,'f','Same-event Gate audit: benefit and adaptation cost');f.text(.98,.02,f'{100*(1-branch[:,1].mean()/branch[:,0].mean()):.1f}% lower mean history loss',transform=f.transAxes,ha='right',fontsize=5.5)")
exec(compile(body8,'<six-core-panels>','exec'))

fig=make(9,'Replay allocation, coverage and control outcomes',(8.1,5.8));a,b,c,d=fig.subplots(2,2).flat
ref=np.load(RECORDS/'critical_event_reference.npz');pool=ref['rows'];selected=ref['selected'];S,R,C=pool[:,1],pool[:,2],pool[:,3]
for kind,mark in [(0,'o'),(1,'^')]:
    mask=C==kind;a.scatter(S[mask],R[mask],s=8,marker=mark,color=GRAY,alpha=.25,label=f'Available C={kind}: {mask.sum()}');mask=mask&selected;a.scatter(S[mask],R[mask],s=15,marker=mark,facecolors='none',edgecolors=BLUE,lw=.5,label=f'Selected C={kind}: {mask.sum()}')
a.set_xscale('symlog',linthresh=.0001);a.set_yscale('symlog',linthresh=.0001);a.set_xlabel('Observed severity');a.set_ylabel('Retention risk');a.legend(loc='best',fontsize=5.4);clean(a);title(a,'a','Original 3-D selection: B = 400; critical states = 52/52')
comp=readcsv('same_pool_domain_retention.csv')
with (RECORDS/'quota_same_pool.csv').open(encoding='utf-8-sig') as ff:comp+=list(csv.DictReader(ff))
domains=sorted({int(x['domain']) for x in comp});mat=np.array([[np.mean([float(x['retention_fraction']) for x in comp if x['variant']==v and int(x['domain'])==dom]) for dom in domains] for v in variants]);im=b.imshow(mat,cmap='cividis',vmin=0,vmax=1,aspect='auto')
for i in range(6):
    for j in range(len(domains)):b.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=5.4,color='white' if mat[i,j]<.42 else 'black')
b.set_yticks(range(6),['Original','Random','Quota + risk','Quota + random','Coverage','Pareto']);b.set_xticks(range(len(domains)),[f'D{k}\nn='+str(len({int(x['run']) for x in comp if int(x['domain'])==k})) for k in domains]);b.set_xlabel('Source domain / available checkpoints');title(b,'b','Same-pool selected / available fractions')
for off,vals,mark,lab in [(-.10,ms[:,:,-1].mean(2),'o','All-domain mean'),(.10,ms[:,:,-1].min(2),'s','Worst domain')]:
    for j,k in enumerate([0,2,3,4,5]):
        mu,er=error(vals[:,k]-vals[:,1]);c.errorbar(j+off,mu,yerr=er.reshape(2,1),fmt=mark,color=cols[k],mfc=cols[k] if off<0 else 'white',capsize=2,ms=3.8,label=lab if j==0 else None)
c.axhline(0,color=GRAY,ls='--',lw=.6);c.set_xticks(range(5),['Original','Quota +\nrisk','Quota +\nrandom','Coverage','Pareto']);c.set_ylabel('Paired score difference to random');c.legend(fontsize=5.4,loc='lower left');clean(c);title(c,'c','Matched-budget replay ablation (12 seeds)')
for k in range(6):
    mu,er=error(100*(1-mr[:,k,-1].mean(1)));d.errorbar(k,mu,yerr=er.reshape(2,1),fmt='o',color=cols[k],ms=4,capsize=2)
d.set_xticks(range(6),['Original','Random','Quota +\nrisk','Quota +\nrandom','Coverage','Pareto']);d.set_ylabel('Final unsuccessful tasks (%)');d.set_ylim(bottom=0);clean(d);title(d,'d','Independent closed-loop control outcomes');save(fig,9)

body10=src.split('# Fig10:')[1].split("stats['long_summary']")[0];body10=body10[body10.index('fig='):]
body10=body10.replace('Event-consistent drift detection and deployment','Event-aligned detection, adaptation and validated deployment')
body10=body10.replace("title(b,'b','Run 1: deployed control on the same time axis')", "title(b,'b','Deployed policy held until validated version switch');b.annotate('Validated policy switch',xy=(78,.99),xytext=(88,.67),arrowprops={'arrowstyle':'->','lw':.6},fontsize=6,color=GREEN)")
start=body10.index('prob=np.stack');end=body10.index('for off,k,col,label,mark',start)
body10=body10[:start]+'''allnames=['normal']+conds
def ready(doc,start,end):
    # Candidate is ready 9 blocks after the trigger block; include rejected attempts.
    return any(start<=ev['trigger_block']<end and ev['trigger_block']+9<end for ev in doc['events'])
probs=[]
for name in allnames:
    arr=normal if name=='normal' else vals[name]
    rd=np.mean([ready(conditions['coupled' if name=='normal' else name,r],0 if name=='normal' else 64,64 if name=='normal' else 128) for r in range(1,13)])
    probs.append([arr[:,0].mean(),rd,arr[:,2].mean()])
prob=np.array(probs);xx=np.arange(5)
def wilson(y):
    z=1.96;center=(y+z*z/24)/(1+z*z/12);half=z*np.sqrt(y*(1-y)/12+z*z/(4*144))/(1+z*z/12);return center-half,center+half
for i,(col,mark,label,off) in enumerate([(BLUE,'o','Detect',-.22),(GOLD,'s','Candidate ready',0),(GREEN,'^','Gate accepted',.22)]):
    y=prob[:,i];lo,hi=wilson(y);c.errorbar(xx+off,y,yerr=np.maximum(0,np.array([y-lo,hi-y])),fmt=mark,color=col,mfc='white' if i==1 else col,ms=3.8,capsize=1.5,lw=.8,label=label)
c.set_xticks(xx,['Normal +\\ntransient','Appearance\\nonly','Weak\\nappearance','Coupled','Gradual'],fontsize=5.9);c.set_ylim(-.035,1.18);c.set_ylabel('Probability within 64 blocks');c.legend(loc='upper left',fontsize=5.5);clean(c);title(c,'c','Detect → candidate → validated commit (12 seeds)')
''' +body10[end:]
exec(compile(body10,'<event-chain-panels>','exec'))

summary={}
for j,v in enumerate(variants):
    summary[v]={'all_domain_score':float(ms[:,j,-1].mean()),'worst_domain_score':float(ms[:,j,-1].min(1).mean()),'unsuccessful_percent':float(100*(1-mr[:,j,-1].mean(1)).mean()),'paired_score_minus_random95':np.array(bootstrap(ms[:,j,-1].mean(1)-ms[:,1,-1].mean(1))).tolist()}
report={'canonical_12seed_results':summary,'orders':order_summary,'functional_loss_reduction_percent':float(100*(1-pf.mean()/pu.mean())),'gate_same_event_loss_reduction_percent':float(100*(1-branch[:,1].mean()/branch[:,0].mean())),'detector_pipeline_probabilities':prob.tolist()}
(FIGURE_OUTPUT/'figure_statistics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(FIGURE_OUTPUT/'font_audit.json').write_text(json.dumps(fonts,indent=2),encoding='utf-8')
np.savez_compressed(FIGURE_OUTPUT/'data/replay_plot_arrays.npz',score=ms,success=mr,coverage=mat)
engine.csv_write(FIGURE_OUTPUT/'data/new_stage_tests.csv',['run','variant','phase','presented_domain','test_domain','tracking_score','success_rate','occupied_slots','memory_count','commits','merges','model_hash','memory_hash'],stage_records)
print(json.dumps(report,indent=2),flush=True)
