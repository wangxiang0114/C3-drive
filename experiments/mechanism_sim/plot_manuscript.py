from pathlib import Path
import argparse
import hashlib
import json
import quota_revision as q
import csv
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import improve
import long_boundary

parser = argparse.ArgumentParser(description="Render the figures from local input files.")
parser.add_argument('--records', type=Path, required=True, help='Local directory with the figure input files')
parser.add_argument('--inputs', type=Path, help='Override the local manuscript_plot_inputs.npz')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
RECORDS = args.records
inputs = args.inputs or RECORDS / 'manuscript_plot_inputs.npz'
sourcepaths = [inputs, *(RECORDS / name for name in (
    'all_pair_audit.json', 'canonical_gate_clause_audit.csv', 'critical_event_reference.npz',
))]
missing = [str(path) for path in sourcepaths if not path.is_file()]
if missing:
    parser.error('Missing local figure inputs: ' + ', '.join(missing))
DEST = args.output
PLOTS = DEST / 'figures'
PLOTS.mkdir(parents=True, exist_ok=False)

# Reuse the archived figure style and statistical definitions.
setup = Path(__file__).with_name('plot_boundary.py').read_text(encoding='utf-8')
style = setup[setup.index("for name in ['times.ttf'"):setup.index("def clean(ax):")]
exec(compile(style, '<mechanism-plot-style>', 'exec'))
with np.load(inputs) as archive:
    arrays = dict(archive)
ms = arrays['score']
mr = arrays['success']
coverage = arrays['coverage']
budget_names = [row[0] for row in long_boundary.CONFIGS]
current = arrays['current']
historical = arrays['historical']
ls = arrays['long_scores']
resources = {name: arrays[name] for name in ('network_bytes', 'router_detector_bytes', 'replay_bytes', 'total_bytes')}

def readcsv(name):
    with (RECORDS / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

BLUE='#2D547B';RED='#C87545';GREEN='#4F8C8D';PURPLE='#86759E';GRAY='#82909D';GOLD='#A99565'
COL=[GRAY,RED,BLUE,GREEN,PURPLE,GOLD];SHORT=['Original','Random','Quota-risk','Quota-rand.','Coverage','Pareto']
HEAT=LinearSegmentedColormap.from_list('paper_blue',['#F5F8FC','#D2DFEC','#90ADCA','#416C98','#153653'])
plt.rcParams.update({'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':5.4,'axes.linewidth':.55,'lines.linewidth':1.15,'axes.edgecolor':'#333333','axes.labelcolor':'black','text.color':'black','axes.labelpad':2,'xtick.major.size':2,'ytick.major.size':2,'legend.handlelength':1.8,'legend.columnspacing':.8,'legend.handletextpad':.4})
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths}
fonts={}
def fig(n,t,size):
    f=plt.figure(figsize=size,layout='constrained');f.set_constrained_layout_pads(w_pad=.025,h_pad=.018,wspace=.045,hspace=.045);f.suptitle(f'Fig. {n}. {t}',fontsize=9,fontweight='bold');return f
def panel(f,spec,l,t):
    g=spec.subgridspec(2,1,height_ratios=[1,.105],hspace=.01);a=f.add_subplot(g[1]);a.axis('off');a.text(.5,.18,f'({l}) {t}',ha='center',va='bottom',fontsize=7.3);return g[0]
def finishax(a):
    a.grid(False)
    for s in a.spines.values():s.set_visible(True);s.set_linewidth(.48);s.set_color('#53616D')
    a.tick_params(direction='out',width=.55)
def savefig(f, n, pair_panel=None):
    f.canvas.draw()
    # Preserve the panel positions without adding a title above the manuscript artwork.
    f.set_layout_engine('none')
    f._suptitle.remove()
    f._suptitle = None
    suffix = 'Paired' if pair_panel else 'Polished'
    if pair_panel:
        redraw_pair_panel(*pair_panel)
    for ext in ['png','pdf','svg']:f.savefig(PLOTS/f'Fig{n}_{suffix}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.025)
    f.savefig(DEST/f'preview_{n}.png',dpi=180,bbox_inches='tight',pad_inches=.025)
    ff=sorted({x.decode('ascii') for x in re.findall(rb'/BaseFont\s*/([^\s/<>]+)',(PLOTS/f'Fig{n}_{suffix}.pdf').read_bytes())});assert ff and all(font.replace(' ', '') in x for x in ff);fonts[str(n)]=ff;plt.close(f)
def meanbar(a,vals,pos,col,width=.28,label=None):
    mu,er=error(vals);a.bar(pos,mu,width=width,color=col,edgecolor='white',lw=.35,label=label,zorder=2);a.errorbar(pos,mu,yerr=er.reshape(2,1),fmt='none',ecolor='#333333',elinewidth=.7,capsize=1.8,zorder=3)
def seedbox(a,arrs,cols,labels,ylabel):
    for j,(values,col) in enumerate(zip(arrs,cols)):
        a.boxplot(values,positions=[j],widths=.44,patch_artist=True,showfliers=False,manage_ticks=False,boxprops={'facecolor':col,'edgecolor':col,'alpha':.25,'lw':.65},medianprops={'color':col,'lw':1.2},whiskerprops={'color':col,'lw':.7},capprops={'color':col,'lw':.7})
        a.scatter(j+np.linspace(-.11,.11,len(values)),values,s=6,color=col,alpha=.65,edgecolors='none',zorder=3)
    a.set_xticks(range(len(labels)),labels);a.set_ylabel(ylabel);finishax(a)
def heat(a,mat,xlab,ylab,fmt='.2f',vmax=1):
    im=a.imshow(mat,aspect='auto',cmap=HEAT,vmin=0,vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):a.text(j,i,format(mat[i,j],fmt),ha='center',va='center',fontsize=5.5,color='white' if mat[i,j]/vmax>.60 else '#132e48')
    a.set_xticks(range(len(xlab)),xlab);a.set_yticks(range(len(ylab)),ylab);a.tick_params(length=0);finishax(a);return im

def redraw_pair_panel(ax, functional, uniform):
    from matplotlib.patches import Rectangle
    if len(functional) != 12 or len(uniform) != 12:
        raise ValueError("Figure 8 requires 12 matched adapter-pair comparisons.")
    lower = min(0, np.floor(min(functional.min(), uniform.min()) * 100) / 100)
    upper = max(.05, np.ceil(max(functional.max(), uniform.max()) * 100) / 100)
    ax.clear()
    for p, u in zip(functional, uniform):
        ax.plot([0, 1], [p, u], color='#83919D', alpha=.35, lw=.68, zorder=1)
    for x, values, color in [(0, functional, BLUE), (1, uniform, RED)]:
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.add_patch(Rectangle((x-.16, q1), .32, q3-q1, facecolor=color,
                               edgecolor=color, lw=.55, alpha=.82, zorder=2))
        ax.scatter(np.full(len(values), x), values, s=9, color=color,
                   edgecolor='white', lw=.35, zorder=3)
        ax.hlines(median, x-.16, x+.16, color='white', lw=1.05, zorder=4)
    ax.set_xlim(-.28, 1.28)
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.linspace(lower, upper, 6))
    ax.set_xticks([0, 1], ['Functional', 'Uniform'])
    ax.set_ylabel('Worst historical loss')
    finishax(ax)
    ax.spines[['top', 'right']].set_visible(False)

# Fig7: consistent rectangular panels; no overlapping glyph layers.
f=fig(7,'Continual adaptation and order sensitivity',(7.2,4.35));gs=f.add_gridspec(2,2,height_ratios=[1.1,1])
a=f.add_subplot(panel(f,gs[0,0],'a','Cross-domain tracking performance'))
heat(a,ms[:,0].mean(0).T,[str(i) for i in range(1,13)],[f'D{i}' for i in range(9)]);a.axvline(8.5,color='white',lw=1.3);a.set_xlabel('Deployment phase (10–12: revisits)');a.set_ylabel('Test domain')
b=f.add_subplot(panel(f,gs[0,1],'b','Adaptation and historical retention'))
for j,sty in [(0,'--'),(1,'-'),(2,'-.')]:
    hist=np.array([[ms[r,j,p,sorted(set(improve.ORDERS['default'][:p]))].mean() if p else np.nan for p in range(12)] for r in range(12)]);mu,lo,hi=bootstrap(hist);b.plot(range(1,13),mu,color=COL[j],ls=sty,label=SHORT[j]);b.fill_between(range(1,13),lo,hi,color=COL[j],alpha=.09)
cur=np.array([[ms[r,2,p,k] for p,k in enumerate(improve.ORDERS['default'])] for r in range(12)]);b.plot(range(1,13),cur.mean(0),color=BLUE,ls=':',marker='o',ms=2,label='Quota: current');b.set_xlabel('Deployment phase');b.set_ylabel('Tracking score');b.set_xlim(1,12);b.set_ylim(.64,.94);b.legend(loc='lower left',ncol=2);finishax(b)
c=f.add_subplot(panel(f,gs[1,0],'c','Retention and new-domain adaptation'))
for j in range(4):
    for off,values,col,lab in [(-.17,ms[:,j,-1].min(1),BLUE,'Final worst domain'),(.17,ms[:,j,8,8],RED,'Last novel domain')]:meanbar(c,values,j+off,col,.31,lab if j==0 else None)
c.set_xticks(range(4),['Original','Random','Quota-risk','Quota-rand.']);c.set_ylim(0,1.1);c.set_ylabel('Tracking score');c.legend(loc='upper left',ncol=2);finishax(c)
d=f.add_subplot(panel(f,gs[1,1],'d','Paired score change relative to random'))
for i, result in enumerate(arrays['order_scores']):
    for j,off,col,lab in [(0,-.18,GRAY,'Original'),(1,.18,BLUE,'Quota-risk')]:
        vals=np.array(result[j])-result[2];parts=d.violinplot(vals,positions=[i+off],widths=.30,showextrema=False)
        for body in parts['bodies']:body.set_facecolor(col);body.set_edgecolor(col);body.set_alpha(.25);body.set_linewidth(.55)
        mu,er=error(vals);d.errorbar(i+off,mu,yerr=er.reshape(2,1),fmt='o',ms=2.8,color=col,capsize=1.8,label=lab if i==0 else None)
d.axhline(0,color='#555555',lw=.6,ls='--');d.set_xticks(range(3),['Original order','Alternating','Permuted']);d.set_ylabel('Paired score difference');d.set_ylim(-.13,.065);d.legend(loc='upper left',ncol=2);finishax(d);savefig(f,7)

# Fig8: uniform line/bar/box grammar; independent scales remain explicit.
f=fig(8,'Resource-constrained consolidation and validation',(7.2,4.6));gs=f.add_gridspec(2,3)
a=f.add_subplot(panel(f,gs[0,0],'a','Persistent resource budget'));x=np.arange(1,19);base=budget_names.index('M4_B400')
for key,lab,col,sty in [('network_bytes','Network',BLUE,'-'),('replay_bytes','Replay',GREEN,'--'),('router_detector_bytes','Auxiliary',RED,'-.'),('total_bytes','Total',GRAY,'-')]:a.plot(x,resources[key][:,base].mean(0)/1024,color=col,ls=sty,label=lab)
a.set_xlim(1,18);a.set_xticks([1,5,10,15,18]);a.set_xlabel('Phase');a.set_ylabel('Array payload (KiB)');a.legend(loc='center right',ncol=2,fontsize=4.9);finishax(a)
b=f.add_subplot(panel(f,gs[0,1],'b','Functional consolidation'))
pair=json.loads((RECORDS/'all_pair_audit.json').read_text(encoding='utf-8'));pf=np.array(pair['functional']);pu=np.array(pair['uniform_expectation'])
for p,u in zip(pf,pu):b.plot([0,1],[p,u],color='#d0d6dc',lw=.6,zorder=1)
seedbox(b,[pf,pu],[BLUE,GRAY],['Functional','Uniform'],'Worst historical loss');b.set_ylim(-.002,.053)
c=f.add_subplot(panel(f,gs[0,2],'c','Budget sensitivity'))
for name,col,sty in [('M2_B400',GRAY,':'),('M4_B200',RED,'--'),('M4_B400',BLUE,'-'),('M4_B800',GREEN,'-.'),('M6_B400',PURPLE,(0,(4,2)))]:
    idx=budget_names.index(name);mu,lo,hi=bootstrap(current[:,idx]);c.plot(x,mu,color=col,ls=sty,label=f'{long_boundary.CONFIGS[idx][1]}/{long_boundary.CONFIGS[idx][2]}');c.fill_between(x,lo,hi,color=col,alpha=.05)
c.set_xlim(1,18);c.set_xticks([1,5,10,15,18]);c.set_xlabel('Phase');c.set_ylabel('Current-domain score');c.legend(title='M / B',title_fontsize=5.2,ncol=2,loc='lower left',fontsize=4.8);finishax(c)
d=f.add_subplot(panel(f,gs[1,0],'d','Decisions at full capacity'))
modes=[('M4_B400','Nominal',BLUE),('M4_B400_strict','Strict',PURPLE),('M4_B400_no_gate','No Gate',GRAY)]
for mode_index, (_, lab, col) in enumerate(modes):
    rate = arrays['gate_rejection_rates'][mode_index]
    d.plot(x,rate,color=col,marker='o',ms=2,ls='--' if lab=='Strict' else '-',label=lab)
d.set_xlim(5,16);d.set_ylim(-.035,1.08);d.set_xticks([6,9,12,15]);d.set_ylabel('Rejected / proposed');d.set_xlabel('Phase');d.legend(loc='upper left',fontsize=5);finishax(d)
e=f.add_subplot(panel(f,gs[1,1],'e','Gate retention–coverage trade-off'))
for i,(name,lab,col) in enumerate(modes):
    idx=budget_names.index(name);meanbar(e,historical[:,idx,-1],i-.17,RED,.30,'History loss' if i==0 else None);meanbar(e,ls[:,idx,-1].min(1),i+.17,BLUE,.30,'Worst score' if i==0 else None)
e.set_xticks(range(3),['Nominal','Strict','Off']);e.set_xlabel('Gate policy');e.set_ylabel('Tracking-score units');e.set_ylim(0,.8);e.legend(loc='upper left',ncol=1,fontsize=5.1);finishax(e)
ff=f.add_subplot(panel(f,gs[1,2],'f','Gate benefit and adaptation cost'))
audit=readcsv('canonical_gate_clause_audit.csv');branch=[]
for run in range(1,13):
    rows=[r for r in audit if int(r['run'])==run];passed=np.array([int(r['passed']) for r in rows],bool);loss=np.maximum(0,[float(r['audit_history_drop']) for r in rows]);gain=np.array([float(r['audit_current_gain']) for r in rows]);branch.append([loss.mean(),np.where(passed,loss,0).mean(),gain.mean(),np.where(passed,gain,0).mean()])
branch=np.array(branch)
for j in range(2):
    for k,off,col,lab in [(0,-.17,RED,'Force'),(1,.17,BLUE,'Gate')]:
        vals=branch[:,j*2+k];meanbar(ff,vals,j+off,col,.30,lab if j==0 else None);ff.scatter(j+off+np.linspace(-.07,.07,12),vals,s=4,color=col,alpha=.45,edgecolors='none',zorder=4)
        if j==0:mu,er=error(vals);ff.text(j+off,mu+er[1]+.007,f'{mu:.3f}',ha='center',fontsize=5.2)
ff.set_xticks([0,1],['History loss','Current gain']);ff.set_ylabel('Tracking-score change');ff.set_ylim(0,.37);ff.legend(loc='upper left',ncol=2,fontsize=5.1);finishax(ff)
savefig(f, 8, pair_panel=(b, pf, pu))

# Fig9: control composition and ablation in standard compact benchmark panels.
f=fig(9,'Replay composition and closed-loop outcomes',(7.2,4.35));gs=f.add_gridspec(2,2)
a=f.add_subplot(panel(f,gs[0,0],'a','Three-objective replay selection'))
ref=np.load(RECORDS/'critical_event_reference.npz');pool=ref['rows'];S,R,C=pool[:,1],pool[:,2],pool[:,3];selected=ref['selected'];crit=C==1
a.scatter(S,R,s=4,color='#bfc8d0',alpha=.5,edgecolors='none',label='Available');a.scatter(S[selected],R[selected],s=8,facecolors='none',edgecolors=BLUE,lw=.4,label='Selected');a.scatter(S[crit],R[crit],s=15,marker='^',facecolors='none',edgecolors=RED,lw=.6,label='Critical')
a.set_xscale('symlog',linthresh=.0001);a.set_yscale('symlog',linthresh=.0001);a.set_xlabel('Severity');a.set_ylabel('Retention risk');a.legend(loc='upper left',ncol=3,fontsize=5.1);finishax(a)
b=f.add_subplot(panel(f,gs[0,1],'b','Domain retention in the same candidate pool'))
heat(b,coverage,[f'D{i}' for i in range(1,7)],SHORT);b.set_xlabel('Source domain')
c=f.add_subplot(panel(f,gs[1,0],'c','Replay ablation at a fixed budget'))
for j in range(6):
    meanbar(c,ms[:,j,-1].mean(1),j-.16,BLUE,.29,'All-domain mean' if j==0 else None);meanbar(c,ms[:,j,-1].min(1),j+.16,GREEN,.29,'Worst domain' if j==0 else None)
ticks=['Orig.','Random','Quota\nrisk','Quota\nrand.','Cover.','Pareto'];c.set_xticks(range(6),ticks);c.set_ylabel('Tracking score');c.set_ylim(0,1.12);c.legend(loc='upper left',ncol=2,fontsize=5.2);finishax(c)
d=f.add_subplot(panel(f,gs[1,1],'d','Independent closed-loop task failures'))
for j in range(6):
    values=100*(1-mr[:,j,-1].mean(1));meanbar(d,values,j,COL[j],.62);d.scatter(j+np.linspace(-.17,.17,12),values,s=5,color='#41464b',alpha=.36,edgecolors='none',zorder=4)
d.set_xticks(range(6),ticks);d.set_ylabel('Unsuccessful tasks (%)');d.set_ylim(0,20);finishax(d);savefig(f,9)

# Fig10: identical time axes and a compact summary/latency row.
f=fig(10,'Event-aligned detection and validated deployment',(7.2,3.95));gs=f.add_gridspec(2,2)
a=f.add_subplot(panel(f,gs[0,0],'a','Sequential detection evidence'));b=f.add_subplot(panel(f,gs[0,1],'b','Deployed control performance'))
for index, (col, lab, sty) in enumerate([(BLUE,'Coupled','-'), (PURPLE,'Weak','--')]):
    tt, gg = [], []
    for block, evidence, threshold, reset, _ in arrays['detector_traces'][index]:
        if reset:
            tt.append(block)
            gg.append(0.)
        tt.append(block + 1)
        gg.append(evidence / threshold)
    a.plot(tt,gg,color=col,ls=sty,label=lab)
    b.plot(np.arange(128)+.5,arrays['detector_traces'][index,:,4],color=col,ls=sty,label=lab)
for ax in [a,b]:
    ax.set_xlim(0,128);ax.set_xticks([0,32,64,96,128]);ax.set_xlabel('Time (blocks)');finishax(ax)
a.axhline(1,color='#555555',ls='--',lw=.65);a.set_ylabel('Evidence / threshold');a.legend(loc='upper left',ncol=2);b.set_ylabel('Tracking score');b.set_ylim(.78,1.025);b.legend(loc='lower left',ncol=2)
conds = ['appearance','hidden','coupled','gradual']
vals = dict(zip(conds, arrays['detector_outcomes']))
counts = np.stack([arrays['normal_outcomes'][:,:3].sum(0), *[vals[k][:,:3].sum(0) for k in conds]]).T
c=f.add_subplot(panel(f,gs[1,0],'c','Detection-to-commit outcomes'))
im=c.imshow(counts/12,cmap=HEAT,vmin=0,vmax=1,aspect='auto')
for i in range(3):
    for j in range(5):c.text(j,i,f'{int(counts[i,j])}/12',ha='center',va='center',fontsize=6.2,color='white' if counts[i,j]>6 else '#163c65')
c.set_yticks(range(3),['Detect','Candidate','Commit']);c.set_xticks(range(5),['Normal','Appear.','Weak','Coupled','Gradual']);c.tick_params(length=0);finishax(c)
d=f.add_subplot(panel(f,gs[1,1],'d','Restricted detection delay'))
for i,k in enumerate(conds):
    meanbar(d,vals[k][:,3],i-.17,BLUE,.31,'Sequential' if i==0 else None);meanbar(d,vals[k][:,4],i+.17,GRAY,.31,'Single block' if i==0 else None)
d.axhline(64,color='#555555',ls=':',lw=.6);d.set_ylim(0,72);d.set_xticks(range(4),['Appear.','Weak','Coupled','Gradual']);d.set_ylabel('Restricted time (blocks)');d.legend(loc='upper left',ncol=2,fontsize=5.1);finishax(d);savefig(f,10)

assert before=={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths}
(DEST/'data_and_font_audit.json').write_text(json.dumps({'status':'PASS','new_experiments':0,'unchanged_data_sha256':before,'fonts':fonts},indent=2),encoding='utf-8')
