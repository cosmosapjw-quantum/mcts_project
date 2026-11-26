"""Post-outcome diagnostic on actual histories, not additional games or tuned evaluation."""
from arena68 import *
def wins(b,player):
 out=[]
 for a in np.flatnonzero(b==0):
  b[a]=player
  if won(b,int(a)):out.append(int(a))
  b[a]=0
 return out
def main():
 a=read(S/'AUDIT68.json');rows=[]
 for z in a['rows']:
  with gzip.open(S/'data'/(z['id']+'.json.gz'),'rt') as f:g=json.load(f)
  for x in g['rows']:
   if x['engine']!='native':continue
   r=x['search'];b=board(r['moves']);me=1 if x['side']==1 else 2;legal=np.flatnonzero(b==0)
   prior=np.array(r['cuts'][0]['prior']);top=sorted(legal,key=lambda i:(-prior[i],int(i)))[:5]
   law=r['events'][0]['root_selection']['law'];pr=np.array(law['p']);assert max(abs(pr-1/len(legal)))<1e-13
   assert max(abs(np.array(law['scores'])))==0
   first=[e['actions'][0] for e in r['events'][:4]]
   n=np.array(r['cuts'][-1]['n']);win=wins(b,me);th=wins(b,3-me);move=r['action']
   rows.append(dict(game=z['id'],core=z['job']['core'],match=z['job']['match'],arm=x['arm'],ply=x['ply'],legal=len(legal),nnue_top5_mass=float(prior[top].sum()),first_top5_uniform_mass=5/len(legal),first_top5_hits=sum(i in top for i in first),final_top5_visits=int(n[top].sum()),nnue_top5_recommendation=int(move in top),wins=win,threats=th,action=move,win_visited=any(n[i]>0 for i in win),win_selected=move in win,unique_defense=(not win and len(th)==1),defense_selected=not win and len(th)==1 and move==th[0]))
 summary=[]
 for cohort in ('external','internal'):
  for arm in ('clean','single','recovery'):
   q=[x for x in rows if x['match'].startswith('ext_' if cohort=='external' else 'internal_') and x['arm']==arm]
   w=[x for x in q if x['wins']];d=[x for x in q if x['unique_defense']]
   summary.append(dict(cohort=cohort,arm=arm,n=len(q),mean_top5_prior_mass=float(np.mean([x['nnue_top5_mass'] for x in q])),mean_top5_initial_law_mass=float(np.mean([x['first_top5_uniform_mass'] for x in q])),first_top5_hits=sum(x['first_top5_hits'] for x in q),final_top5_visit_fraction=sum(x['final_top5_visits'] for x in q)/(64*len(q)),recommendation_top5_fraction=float(np.mean([x['nnue_top5_recommendation'] for x in q])),immediate_win_positions=len(w),immediate_win_visited=sum(x['win_visited'] for x in w),immediate_win_selected=sum(x['win_selected'] for x in w),unique_defense_positions=len(d),unique_defense_selected=sum(x['defense_selected'] for x in d)))
 dump(S/'TACTICAL68.json',dict(scope='post-outcome diagnostic; exact immediate >=5 win and unique immediate opponent threat (without own win), not full tactical oracle; all initial wave p uniform; no new games',rows=rows,summary=summary))
 print('TACTICAL',summary,flush=True)
if __name__=='__main__':main()
