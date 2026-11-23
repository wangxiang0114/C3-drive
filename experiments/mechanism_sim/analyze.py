from pathlib import Path
import json,csv,gzip,argparse,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import engine

parser = argparse.ArgumentParser(description="Render the archived synthetic mechanism plots.")
parser.add_argument('--directory', type=Path, required=True)
args = parser.parse_args()
OUT = args.directory
PLOTS = OUT / 'figures'
PLOTS.mkdir(exist_ok=True)
DATA=OUT/'data';config=json.loads((OUT/'simulation_config.json').read_text(encoding='utf-8'));summary=json.loads((OUT/'simulation_summary.json').read_text(encoding='utf-8'))
variants=config['variants'];nr=config['runs'];np_phase=len(config['sequence']);ND=9
def csvrows(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
stage=csvrows('stage_domain_tests.csv');events=csvrows('update_events.csv');blocks=csvrows('block_log.csv');attempts=csvrows('gate_attempts.csv')
score=np.zeros((nr,4,np_phase,ND));success=score.copy();slots=np.zeros((nr,4,np_phase));memsize=slots.copy();merges=slots.copy();commits=slots.copy()
for r in stage:
    run=int(r['run'])-1;m=variants.index(r['variant']);phase=int(r['phase']);domain=int(r['test_domain'])
    score[run,m,phase,domain]=float(r['tracking_score']);success[run,m,phase,domain]=float(r['success_rate']);slots[run,m,phase]=int(r['occupied_slots']);memsize[run,m,phase]=int(r['memory_count']);commits[run,m,phase]=int(r['commits']);merges[run,m,phase]=int(r['merges'])
forget=np.full((nr,4,np_phase),np.nan);bwt=forget.copy()
for run in range(1,nr+1):
    for m,variant in enumerate(variants):
        first={0:0}
        for e in events:
            if int(e['run'])==run and e['variant']==variant and int(e['passed']):first.setdefault(int(e['training_domain']),int(e['decision_block'])//64)
        for phase in range(np_phase):
            historical=[k for k,p in first.items() if p<phase]
            if historical:
                forget[run-1,m,phase]=np.mean([score[run-1,m,first[k]:phase,k].max()-score[run-1,m,phase,k] for k in historical])
                bwt[run-1,m,phase]=np.mean([score[run-1,m,phase,k]-score[run-1,m,first[k],k] for k in historical])

method_summary=[]
for m,variant in enumerate(variants):
    values=score[:,m,-1].mean(1);sr=success[:,m,-1].mean(1);ev=[e for e in events if e['variant']==variant]
    method_summary.append([variant,nr,float(values.mean()),float(values.std(ddof=1)) if nr>1 else 0,float(1.96*values.std(ddof=1)/np.sqrt(nr)) if nr>1 else 0,float(sr.mean()),sum(int(e['passed']) for e in ev),sum(int(e['merge_committed']) for e in ev),sum(1-int(e['passed']) for e in ev),float(np.nanmean(forget[:,m,-1])),float(np.nanmean(bwt[:,m,-1]))])

# Fixed reference rule: run 1, first tested proposal after all four slots are full.
reference=next((e for e in events if int(e['run'])==1 and e['variant']=='full' and int(e['slots_before'])==4),next(e for e in events if int(e['run'])==1 and e['variant']=='full'))
mc=np.load(DATA/'memory_candidates.npz')['rows'];reference_pool=mc[(mc[:,0]==1)&(mc[:,1]==int(reference['event']))]
with gzip.open(DATA/'online_frames_full_run01.csv.gz','rt',encoding='utf-8-sig') as f:raw=np.loadtxt(f,delimiter=',',skiprows=1)
fullblocks=[r for r in blocks if r['variant']=='full'];reference_blocks=[r for r in fullblocks if int(r['run'])==1]
diagnostic=np.zeros((nr,engine.TOTAL_BLOCKS));evidence=diagnostic.copy();pvalues=diagnostic.copy();states=diagnostic.copy()
for r in fullblocks:
    i=int(r['run'])-1;b=int(r['block']);diagnostic[i,b]=float(r['raw_diagnostic_mean']);evidence[i,b]=float(r['evidence']);pvalues[i,b]=float(r['union_p']);states[i,b]=int(r['state'])
delays=[];missed=[];known_trigger_count=0
for run in range(1,nr+1):
    ev=[e for e in events if int(e['run'])==run and e['variant']=='full'];triggers=[r for r in fullblocks if int(r['run'])==run and int(r['trigger'])]
    for phase,domain in enumerate(config['sequence']):
        start=phase*64;end=start+64
        accepted={0}|{int(e['training_domain']) for e in ev if int(e['passed']) and int(e['decision_time'])<=start}
        ts=[r for r in triggers if int(r['phase'])==phase]
        if domain in accepted:known_trigger_count+=len(ts);continue
        if ts:delays.append([run,phase,domain,int(ts[0]['block'])+1-start])
        else:missed.append([run,phase,domain])
cal=np.load(DATA/'threshold_calibration.npz');heldout_fa=int((cal['heldout_episode_maxima']>summary['threshold']).sum())
np.savez_compressed(DATA/'plot_arrays.npz',tracking_score=score,success=success,forgetting=forget,bwt=bwt,slots=slots,memory_count=memsize,merges=merges,commits=commits,diagnostic_block_mean=diagnostic,evidence=evidence,pvalue=pvalues,state=states,reference_pool=reference_pool)
derived={'reference_memory_event':{k:reference[k] for k in ['run','event','trigger_time','decision_time','passed','model_before','model_after','memory_hash_before','memory_hash_after']},'reference_pool_size':len(reference_pool),'reference_selected':int(reference_pool[:,8].sum()),'reference_front1':int((reference_pool[:,7]==1).sum()),'detection_delay_median':float(np.median(np.array(delays)[:,3])) if delays else None,'detection_delays':delays,'missed_unaccepted_domain_phases':missed,'known_domain_trigger_count':known_trigger_count,'heldout_clean_false_alarms':heldout_fa,'heldout_clean_episodes':len(cal['heldout_episode_maxima']),'method_summary':method_summary}
(OUT/'derived_summary.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')

for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    p=Path('C:/Windows/Fonts')/name
    if p.exists():font_manager.fontManager.addfont(str(p))
font = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
plt.rcParams.update({'font.family':font,'font.serif':[font],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,'axes.labelpad':2,'lines.linewidth':1.1,'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.pad':2,'ytick.major.pad':2,'legend.fontsize':6.5,'legend.borderpad':.3,'legend.labelspacing':.25,'mathtext.fontset':'custom','mathtext.rm':font,'mathtext.it':font + ':italic','mathtext.bf':font + ':bold','mathtext.sf':font,'mathtext.tt':font,'mathtext.cal':font + ':italic','mathtext.fallback':None,'svg.fonttype':'none','pdf.fonttype':42,'savefig.pad_inches':.025})
blue,red,purple,gray,green='#1768c5','#d92d40','#8a269e','#777777','#197c42'
colors=[blue,purple,red,gray];names=['Full system','Random replay','Slot replacement','Frozen policy'];audit={}
def title(ax,letter,text):ax.set_title(f'({letter}) {text}',loc='center',fontweight='bold',pad=3)
def grid(ax):ax.grid(alpha=.16,lw=.45);ax.set_axisbelow(True)
def save(fig,n):
    for ext in ['png','svg','pdf']:fig.savefig(PLOTS/f'Fig{n}_UnifiedSimulation.{ext}',dpi=600,bbox_inches='tight')
    fonts=sorted({f.decode('ascii') for f in re.findall(rb'/BaseFont\s*/([^\s/<>]+)',(PLOTS/f'Fig{n}_UnifiedSimulation.pdf').read_bytes())})
    assert fonts and all('TimesNewRoman' in f for f in fonts),fonts
    audit[str(n)]=fonts;plt.close(fig)
def ci(a):return 1.96*a.std(0,ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.zeros(a.shape[1:])

# Fig 7: all matrix cells and marginal statistics come from heldout rollouts.
fig=plt.figure(figsize=(7.2,5.4),layout='constrained');fig.set_constrained_layout_pads(w_pad=.012,h_pad=.012,wspace=.015,hspace=.015)
gs=fig.add_gridspec(2,3,width_ratios=[4.2,.12,1.6],height_ratios=[3.1,1.15]);a=fig.add_subplot(gs[0,0]);cb=fig.add_subplot(gs[0,1]);b=fig.add_subplot(gs[0,2]);c=fig.add_subplot(gs[1,:2]);dd=fig.add_subplot(gs[1,2])
mean=score[:,0].mean(0);mean_sr=success[:,0].mean(0);im=a.imshow(mean,vmin=0,vmax=1,cmap='RdYlBu_r',aspect='auto')
for i in range(np_phase):
    for j in range(ND):a.text(j,i,f'{mean[i,j]:.2f}/{mean_sr[i,j]:.2f}',ha='center',va='center',fontsize=5.3,color='white' if mean[i,j]>.85 or mean[i,j]<.3 else 'black')
a.set_xticks(range(9),[f'D{i}' for i in range(9)]);a.set_yticks(range(np_phase),[f'P{i+1}: D{k}' for i,k in enumerate(config['sequence'])],fontsize=6.4);a.set_ylabel('Deployment phase');title(a,'a','Tracking score / success rate');fig.colorbar(im,cax=cb,label='Tracking score')
for m in range(4):b.plot(score[:,m].mean((0,2)),range(np_phase),'o-',color=colors[m],ms=2.3,label=names[m])
b.set_ylim(np_phase-.5,-.5);b.set_yticks(range(np_phase),[str(i+1) for i in range(np_phase)]);b.set_xlim(.25,1);b.set_xticks([.4,.7,1]);b.set_xlabel('All-domain mean');b.legend(loc='lower left',fontsize=5.8);grid(b);title(b,'b','Row means')
x=np.arange(9)
for m in range(4):c.bar(x+(m-1.5)*.19,score[:,m,-1].mean(0),.19,color=colors[m],label=names[m])
c.set_xticks(x,[f'D{i}' for i in x]);c.set_ylim(0,1.12);c.set_ylabel('Tracking score');c.legend(ncol=2,fontsize=5.8,loc='upper right');title(c,'c','Final heldout performance')
for m in [0,1]:dd.plot(range(2,np_phase+1),np.nanmean(forget[:,m,1:],axis=0),color=colors[m],marker='o',ms=2,label=names[m])
dd.axhline(0,color='#777',ls='--',lw=.5);dd.set_xlabel('Phase');dd.set_ylabel('Historical forgetting');dd.set_xticks([2,6,9,12]);grid(dd);title(dd,'d','Accepted-domain retention');save(fig,7)

# Fig 8: capacity, performance and gate outcomes share the same commit events.
fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained');fig.set_constrained_layout_pads(w_pad=.012,h_pad=.012,wspace=.02,hspace=.02)
a,b,c,dd=axs.flat;x=np.arange(1,np_phase+1);assigned=engine.DX*2+slots[:,0]*(engine.DP*engine.H+engine.DB*2)
a.axhline(summary['allocated_parameters'],ls='--',color='#555',label='Reserved capacity');a.plot(x,assigned.mean(0),'o-',color=blue,ms=2.6,label='Assigned slot capacity');a.fill_between(x,*np.quantile(assigned,[.1,.9],axis=0),color=blue,alpha=.15);a.set_xlabel('Deployment phase');a.set_ylabel('Scalar parameter capacity');a.set_ylim(0,1800);a.legend(loc='lower right');grid(a);title(a,'a','Fixed adapter budget')
for m in [0,2]:
    values=score[:,m].mean(2);mu=values.mean(0);err=ci(values);b.plot(x,mu,color=colors[m],label=names[m]);b.fill_between(x,mu-err,mu+err,color=colors[m],alpha=.15)
b.set_xlabel('Deployment phase');b.set_ylabel('All-domain tracking score');b.legend();grid(b);title(b,'b','Performance from the same checkpoints')
rejections=np.zeros((nr,np_phase))
for e in events:
    if e['variant']=='full' and not int(e['passed']):rejections[int(e['run'])-1,int(e['decision_block'])//64:]+=1
for values,color,label in [(merges[:,0],green,'Committed merges'),(rejections,red,'Rejected updates')]:
    c.plot(x,values.mean(0),color=color,label=label);c.fill_between(x,*np.quantile(values,[.1,.9],axis=0),color=color,alpha=.15)
c.set_xlabel('Deployment phase');c.set_ylabel('Cumulative events');c.legend();grid(c);title(c,'c','Observed capacity decisions')
ev=[e for e in events if e['variant']=='full']
for flag,color,marker,label in [(1,green,'o','Committed'),(0,red,'x','Rejected')]:
    es=[e for e in ev if int(e['passed'])==flag];dd.scatter([float(e['current_gain']) for e in es],[float(e['historical_score_drop']) for e in es],s=12,color=color,marker=marker,alpha=.6,label=label)
dd.axvline(.004,color='#777',ls='--',lw=.6);dd.axhline(.06,color='#777',ls='--',lw=.6);dd.set_xlabel('Current-domain validation gain');dd.set_ylabel('Worst historical score drop');dd.legend();grid(dd);title(dd,'d','Finite-sample validation outcomes');save(fig,8)

# Fig 9: candidates are real collected/replayed samples, not a standalone cloud.
pool=reference_pool;selected=pool[:,8]==1;front=pool[:,7]==1
fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained');fig.set_constrained_layout_pads(w_pad=.012,h_pad=.012,wspace=.02,hspace=.02)
a,b,c,dd=axs.flat;a.scatter(pool[:,4],pool[:,5],s=8,color='#aeb4bc',alpha=.35,label=f'Candidates ({len(pool)})',rasterized=True);a.scatter(pool[front,4],pool[front,5],s=30,facecolors='none',edgecolors=red,label='3D front 1');a.scatter(pool[selected,4],pool[selected,5],marker='*',s=17,color='#ffda39',edgecolors='#333',lw=.3,label=f'Proposed replay ({selected.sum()})');a.set_xscale('symlog',linthresh=.005);a.set_yscale('symlog',linthresh=.005);a.set_xlabel('Safety severity');a.set_ylabel('Update-induced retention risk');a.legend(fontsize=6);grid(a);title(a,'a','Replay proposal at a saturated pool')
domains=np.unique(pool[:,3]).astype(int);xx=np.arange(len(domains));oldloss=[pool[pool[:,3]==k,9].mean() for k in domains];newloss=[pool[pool[:,3]==k,10].mean() for k in domains]
b.bar(xx-.18,oldloss,.36,color='#97b7dd',label='Deployed model');b.bar(xx+.18,newloss,.36,color='#e98692',label='Candidate model');b.set_xticks(xx,[f'D{k}' for k in domains]);b.set_ylabel('Observed behavior loss');b.legend(fontsize=6);title(b,'b','Retention risk from model predictions')
values=score[:,:2,-1].mean(2)
for row in values:c.plot([0,1],row,color='#aaa',lw=.6,alpha=.5)
c.scatter(np.zeros(nr),values[:,0],color=blue,s=13);c.scatter(np.ones(nr),values[:,1],color=purple,s=13);c.set_xticks([0,1],['Pareto + k-center','Random replay']);c.set_xlim(-.3,1.3);c.set_ylabel('Final all-domain tracking score');grid(c);title(c,'c','Downstream replay intervention')
candidate_fraction=np.array([(pool[:,3]==k).mean() for k in domains]);chosen_fraction=np.array([(pool[selected,3]==k).mean() for k in domains]);dd.bar(xx-.18,candidate_fraction,.36,color='#c2c7ce',label='Candidates');dd.bar(xx+.18,chosen_fraction,.36,color='#e8bd53',label='Proposed replay');dd.set_xticks(xx,[f'D{k}' for k in domains]);dd.set_ylabel('Within-group fraction');dd.legend(fontsize=6);title(dd,'d','Replay composition by source domain');save(fig,9)

# Fig 10: the same run's score, evidence, state changes, and population delays.
fig=plt.figure(figsize=(7.2,5.4),layout='constrained');fig.set_constrained_layout_pads(w_pad=.012,h_pad=.01,wspace=.025,hspace=.012)
gs=fig.add_gridspec(3,3,height_ratios=[1.4,1.45,1.1]);a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:],sharex=a);c=fig.add_subplot(gs[2,:2]);dd=fig.add_subplot(gs[2,2])
for ax in [a,b,c]:
    for i,domain in enumerate(config['sequence']):ax.axvspan(i*64,(i+1)*64,color=['#eaf5ed','#fff3da','#e7f0fa','#fcebed'][i%4]);ax.axvline(i*64,color='#aaa',ls='--',lw=.4)
    ax.set_xlim(0,engine.TOTAL_BLOCKS);grid(ax)
a.plot(raw[:,2],raw[:,12],color=blue,lw=.4,alpha=.75,label='Run 1 frame score');a.fill_between(np.arange(engine.TOTAL_BLOCKS)+.5,*np.quantile(diagnostic,[.1,.9],axis=0),color=blue,alpha=.18,label='10–90% of block means');a.plot(np.arange(engine.TOTAL_BLOCKS)+.5,np.median(diagnostic,axis=0),color='#153e75',lw=.8,label='Median block mean');a.set_yscale('log');a.set_ylabel(r'Diagnostic score $a_n^{\mathrm{raw}}$');a.tick_params(labelbottom=False);a.legend(loc='lower left',ncol=3,fontsize=6)
for i,domain in enumerate(config['sequence']):a.text(i*64+32,.95,f'D{domain}',ha='center',va='top',fontsize=7,transform=a.get_xaxis_transform())
title(a,'a','Observed domain stream and diagnostic scores')
for row in evidence:b.plot(np.arange(engine.TOTAL_BLOCKS)+1,row,color=purple,alpha=.1,lw=.55)
b.plot(np.arange(engine.TOTAL_BLOCKS)+1,evidence[0],color=purple,lw=1,label='Run 1');b.axhline(summary['threshold'],color='#444',ls='--',label=rf'$\eta_{{\mathrm{{FA}}}}={summary["threshold"]:.2f}$');b.set_yscale('symlog',linthresh=.5);b.set_ylim(0,evidence.max()*1.2);b.set_ylabel(r'Block evidence $g_b$');b.tick_params(labelbottom=False);b.legend(loc='upper left',ncol=2);title(b,'b','Sequential evidence and update attempts')
c.step(np.arange(engine.TOTAL_BLOCKS),states[0],where='post',color=green,lw=1.0);c.set_yticks(range(5),['Monitoring','Collecting','Training','Validating','Commit'],fontsize=6.6);c.set_ylim(-.2,4.5);c.set_xticks([0,192,384,576,768]);c.set_xlabel('Time (blocks)');title(c,'c','Update states in run 1')
delayvals=np.array(delays)[:,3] if delays else np.array([0]);dd.hist(delayvals,bins=np.arange(delayvals.min()-.5,delayvals.max()+1.5),color='#bca1cf',edgecolor=purple,lw=.5);dd.set_xlabel('Delay (blocks)');dd.set_ylabel('New-domain phases');dd.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True));dd.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));grid(dd);title(dd,'d','First-trigger delays');save(fig,10)
(PLOTS/'font_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

phase_summary=[]
for run in range(nr):
    for m,variant in enumerate(variants):
        for phase in range(np_phase):phase_summary.append([run+1,variant,phase,config['sequence'][phase],float(score[run,m,phase].mean()),float(success[run,m,phase].mean()),int(slots[run,m,phase]),int(memsize[run,m,phase]),int(commits[run,m,phase]),int(merges[run,m,phase]),float(forget[run,m,phase]) if phase else None,float(bwt[run,m,phase]) if phase else None])
engine.csv_write(DATA/'phase_summary.csv',['run','variant','phase','presented_domain','all_domain_score','all_domain_success','slots','memory_count','commits','merges','accepted_domain_forgetting','accepted_domain_bwt'],phase_summary)
engine.csv_write(DATA/'detection_delays.csv',['run','phase','domain','first_trigger_delay_blocks'],delays)
def typed(r):
    out=[]
    for key,value in r.items():
        if value=='':out.append(None);continue
        if key=='variant' or 'hash' in key or key in ['model_before','model_after']:
            out.append(value);continue
        try:out.append(float(value))
        except ValueError:out.append(value)
    return out
payload={'config':config,'summary':summary,'derived':derived,'method_summary':method_summary,'stage_headers':list(stage[0]),'stage_rows':[typed(r) for r in stage],'event_headers':list(events[0]),'event_rows':[typed(r) for r in events],'reference_event':reference,'reference_pool':pool.tolist(),'reference_block_headers':list(reference_blocks[0]),'reference_blocks':[typed(r) for r in reference_blocks],'reference_frames':raw.tolist(),'phase_summary':engine.plain(phase_summary),'mean_tracking_matrix':mean.tolist(),'mean_success_matrix':mean_sr.tolist(),'threshold_maxima':cal['episode_maxima'].tolist(),'heldout_maxima':cal['heldout_episode_maxima'].tolist()}
(OUT/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps({'reference_event':derived['reference_memory_event'],'method_summary':method_summary,'heldout_clean_false_alarms':heldout_fa,'missed_phases':len(missed),'figures':str(PLOTS)},ensure_ascii=False,indent=2))
