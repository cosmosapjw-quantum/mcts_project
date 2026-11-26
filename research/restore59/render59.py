import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from predict59 import sha,dump
from PIL import Image
S=Path(__file__).resolve().parent;r=json.loads((S/'PHYSICS_RESULTS.json').read_text());d=json.loads((S/'DIAGNOSTICS59.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
manifest=[]
def save(fig,name,ids):
 fig.tight_layout();base=S/'plots'/name
 fig.savefig(base.with_suffix('.png'),dpi=180);fig.savefig(base.with_suffix('.pdf'));plt.close(fig)
 with Image.open(base.with_suffix('.png')) as im:im.verify()
 assert base.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
 manifest.append(dict(figure=name,ids=ids,png_sha=sha(base.with_suffix('.png')),pdf_sha=sha(base.with_suffix('.pdf'))))
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].bar(range(5),r['mse']);ax[0].set_xticks(range(5),['Zero','Constant','Action mean','State','State + history'],rotation=25,ha='right');ax[0].set_ylabel('Held-out mark MSE');ax[0].set_title('64 new searches; B=32')
for j,label in [(3,'State'),(4,'State + history')]:
 n=sorted(map(int,r['convergence']));ax[1].plot(n,[r['convergence'][str(k)][j] for k in n],marker='o',label=label)
ax[1].set(xlabel='Training histories',ylabel='Held-out MSE',title='Nested training sample sizes');ax[1].legend()
save(fig,'L40_L41_predictive_comparison',['L40','L41','L49'])
fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
for k,label in [('raw','Raw'),('state_resid','State residual'),('history_resid','History residual')]:
 ax[0].plot(range(1,6),r['acf'][k],marker='o',label=label);ax[1].plot(range(1,6),d['root_centered_acf'][k],marker='o',label=label)
for a,t in zip(ax,['Pooled wave pairs','Within-root / phase centered']):
 a.axhline(0,color='gray',lw=.8);a.set(xlabel='Wave lag',title=t,ylim=(-.35,.85));a.legend(fontsize=8)
ax[0].set_ylabel('Pair correlation')
save(fig,'L39_L44_wave_correlations',['L39','L44'])
fig,ax=plt.subplots(1,3,figsize=(11,3.8))
for a,key,title in zip(ax[:2],['transition_raw','transition_history_resid'],['Raw wave means','History-model residual']):
 z=np.array(r[key]);q=z/np.maximum(z.sum(1,keepdims=True),1);a.imshow(q,vmin=0,vmax=1,cmap='viridis')
 for i in range(3):
  for j in range(3):a.text(j,i,f'{q[i,j]:.2f}\n(n={z[i,j]})',ha='center',va='center',color='white' if q[i,j]<.65 else 'black',fontsize=8)
 a.set(xticks=range(3),yticks=range(3),xlabel='Next bin',ylabel='Current bin',title=title)
p=d['transition_prediction'];ax[2].bar(p['orders'],p['nll']);ax[2].set(xticks=[0,1,2],xlabel='Past wave order (clock conditioned)',ylabel='NLL (nats)',title=f"Exploratory test; n={p['n_test_transitions']}")
save(fig,'L40_L43_transition_predictions',['L40','L43'])
fig,ax=plt.subplots(1,3,figsize=(12,3.8))
for c,label in zip(r['calibration'],['State','State + history']):ax[0].plot([x['pred'] for x in c],[x['observed'] for x in c],marker='o',label=label)
ax[0].plot([-1,1],[-1,1],':',color='gray');ax[0].set(xlabel='Predicted mark',ylabel='Mean observed mark',title='Held-out calibration');ax[0].legend(fontsize=8)
ax[1].bar(range(4),r['core_differences']);ax[1].axhline(0,color='gray');ax[1].set(xticks=range(4),xlabel='Core block',ylabel='MSE(state) - MSE(history)',title='Paired core effects')
q=d['root_lag1'];x=np.arange(len(q))
for k,j in [('raw',0),('state_resid',1),('history_resid',2)]:ax[2].bar(x+(j-1)*.25,[a[k] for a in q],width=.25,label=k)
ax[2].set(xticks=x,xticklabels=[f"{a['L']}/{a['core']}" for a in q],xlabel='Board/core',ylabel='Lag-1 correlation',title='Root-specific wave pairs');ax[2].tick_params(axis='x',rotation=45);ax[2].legend(fontsize=7)
save(fig,'L42_L48_L49_diagnostic_summary',['L42','L48','L49'])
dump(S/'PLOT_MANIFEST59.json',dict(sources={k:sha(S/k) for k in ['PHYSICS_RESULTS.json','DIAGNOSTICS59.json','HELDOUT_PREDICTIONS.npz']},figures=manifest))
print('PASS',len(manifest),'PNG/PDF pairs')
