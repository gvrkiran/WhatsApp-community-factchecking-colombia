"""Analyses added for the FAccT 2027 version: verdict thresholds, a cross-partisan agreement rule,
and the feedback model with next-item fixed effects.
Run from the paper root: python analysis/extra_analyses.py
Inputs (analysis/): long.csv, items.csv. Output: analysis/numbers_extra.json
"""
import json, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
rng=np.random.default_rng(2027)
A='analysis'
L=pd.read_csv(f'{A}/long.csv'); I=pd.read_csv(f'{A}/items.csv')
L=L.drop(columns=[c for c in ['modality','topic'] if c in L.columns]).merge(I.drop(columns=['label']), on='item')
L['pol']=L.pol_pref_2.map({1:'Left',2:'Center',3:'Right'})
N={}
It=L.groupby('item').agg(label=('label','first'),frac_true=('rated_true','mean'),n=('rated_true','size'),genre=('genre','first'),modality=('modality','first'),leaning=('leaning','first')).reset_index()
# ---------------- verdict thresholds ----------------
# An item is labeled false when the share of raters calling it true is below t.
t_sh=It[It.label==1].frac_true.values; f_sh=It[It.label==0].frac_true.values
def auc(a,b): return float(np.mean([(x>y)+0.5*(x==y) for x in a for y in b]))
boot=[auc(rng.choice(t_sh,len(t_sh)),rng.choice(f_sh,len(f_sh))) for _ in range(2000)]
N['threshold']={'auc':round(auc(t_sh,f_sh),3),'auc_ci':[round(np.percentile(boot,2.5),3),round(np.percentile(boot,97.5),3)],
 'true_share_min':round(t_sh.min(),2),'true_share_median':round(float(np.median(t_sh)),2),'false_share_median':round(float(np.median(f_sh)),2),'false_share_max':round(f_sh.max(),2),
 'rows':[{'t':t,'false_flagged':int((f_sh<t).sum()),'true_flagged':int((t_sh<t).sum())} for t in [0.4,0.5,0.6,0.7]]}
# ---------------- cross-partisan agreement rule ----------------
# Label an item only when the Left- and Right-identifying raters who judged it agree (both majorities true, or both false).
S=L[L.pol.isin(['Left','Right'])].groupby(['item','pol']).rated_true.agg(['mean','size']).unstack()
S.columns=[f'{a}_{b}' for a,b in S.columns]; S=S.reset_index().merge(It,on='item')
S['bridge']=np.where((S.mean_Left>0.5)&(S.mean_Right>0.5),'true',np.where((S.mean_Left<0.5)&(S.mean_Right<0.5),'false','none'))
S['majority']=np.where(S.frac_true>0.5,'true',np.where(S.frac_true<0.5,'false','tie'))
def counts(d,col): return {k:int((d[col]==k).sum()) for k in ['true','false','none','tie'] if (d[col]==k).sum()>0}
N['cross_partisan']={'raters_per_side_median':{'Left':float(S.size_Left.median()),'Right':float(S.size_Right.median())},'raters_per_side_min':int(min(S.size_Left.min(),S.size_Right.min())),
 'false_items':{'rule':counts(S[S.label==0],'bridge'),'majority':counts(S[S.label==0],'majority')},
 'true_items':{'rule':counts(S[S.label==1],'bridge'),'majority':counts(S[S.label==1],'majority')},
 'false_item_table':S[S.label==0].sort_values('frac_true',ascending=False)[['item','modality','genre','leaning','frac_true','mean_Left','size_Left','mean_Right','size_Right','majority','bridge']].round(2).to_dict('records')}
# ---------------- feedback with next-item fixed effects ----------------
maj=It.set_index('item').frac_true; L['maj_true']=(L.item.map(maj)>0.5).astype(int); L['tie']=(L.item.map(maj)==0.5).astype(int)
L['minority']=((L.rated_true!=L.maj_true)&(L.tie==0)).astype(int)
L=L.sort_values(['ID','position'])
for c in ['position','rated_true','label','item']: L[f'next_{c}']=L.groupby('ID')[c].shift(-1)
Nn=L[L.next_position==L.position+1].copy()
fe=smf.glm('next_rated_true ~ minority + rated_true + C(next_item)',data=Nn,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(Nn.ID)[0]})
ci=np.exp(fe.conf_int().loc['minority']).round(2).tolist()
N['feedback_next_item_fe']={'pairs':int(len(Nn)),'or':round(float(np.exp(fe.params['minority'])),2),'ci':ci,'p':round(float(fe.pvalues['minority']),3)}
def conv(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    return str(o)
json.dump(N,open(f'{A}/numbers_extra.json','w'),indent=1,default=conv)
print(json.dumps({k:v for k,v in N.items() if k!='cross_partisan'},indent=0,default=conv))
print(json.dumps({k:v for k,v in N['cross_partisan'].items() if k!='false_item_table'},indent=0,default=conv))
