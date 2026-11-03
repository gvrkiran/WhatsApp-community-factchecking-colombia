"""Generate all figures and numbers.json for the manuscript.
Run from the paper root: python3 analysis/make_figures.py
Inputs (analysis/): long.csv (pseudonymized ratings), items.csv, aggregation_curves.csv, R.npy, lab.npy
Outputs: img/fig_*.pdf (+ .png previews), analysis/numbers.json
"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
rng=np.random.default_rng(2027)
A='analysis'; IMG='img'; import os; os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'figure.dpi':150,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
C_TRUE='#1f78b4'; C_FALSE='#e6550d'; C_GREY='#7f7f7f'; C_BAL='#33a02c'; C_L='#6a3d9a'; C_R='#b15928'
L=pd.read_csv(f'{A}/long.csv'); I=pd.read_csv(f'{A}/items.csv')
L=L.drop(columns=[c for c in ['modality','topic'] if c in L.columns]).merge(I.drop(columns=['label']), on='item')
L['pol']=L.pol_pref_2.map({1:'Left',2:'Center',3:'Right'}); L['is_video']=(L.modality=='video').astype(int)
N={}
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (c-h,c+h)
# ---------------- basic counts ----------------
N['n_ratings']=int(len(L)); N['n_participants']=int(L.ID.nunique()); N['n_items']=int(L.item.nunique())
g=L.groupby('ID').size(); N['ratings_per_participant']={'mean':round(g.mean(),1),'min':int(g.min()),'max':int(g.max())}
gi=L.groupby('item').size(); N['ratings_per_item']={'mean':round(gi.mean(),1),'min':int(gi.min()),'max':int(gi.max())}
N['response_rate']=round(len(L)/(38*20),3); N['participants_ge90']=int((g>=18).sum()); N['participants_ge60']=int((g>=12).sum())
N['acc_overall']=round(L.correct.mean(),3); N['acc_true_ratings']=round(L[L.label==1].correct.mean(),3); N['acc_false_ratings']=round(L[L.label==0].correct.mean(),3)
N['share_rated_true']=round(L.rated_true.mean(),3)
# ---------------- SDT per participant ----------------
def sdt(gp):
    nT=(gp.label==1).sum(); nF=(gp.label==0).sum(); hT=gp[gp.label==1].rated_true.sum(); fF=gp[gp.label==0].rated_true.sum()
    h=(hT+0.5)/(nT+1); f=(fF+0.5)/(nF+1)
    return pd.Series(dict(hit=hT/nT, fa=fF/nF, dprime=stats.norm.ppf(h)-stats.norm.ppf(f), crit=-0.5*(stats.norm.ppf(h)+stats.norm.ppf(f)), n=len(gp), acc=gp.correct.mean(), acc_true=gp[gp.label==1].correct.mean(), acc_false=gp[gp.label==0].correct.mean()))
P=L.groupby('ID').apply(sdt).reset_index()
t=stats.ttest_rel(P.acc_true,P.acc_false); w=stats.wilcoxon(P.acc_true,P.acc_false); dz=(P.acc_true-P.acc_false).mean()/(P.acc_true-P.acc_false).std()
tc=stats.ttest_1samp(P.crit,0); td=stats.ttest_1samp(P.dprime,0)
N['sdt']={'hit_mean':round(P.hit.mean(),3),'hit_sd':round(P.hit.std(),3),'fa_mean':round(P.fa.mean(),3),'fa_sd':round(P.fa.std(),3),
 'dprime_mean':round(P.dprime.mean(),2),'dprime_sd':round(P.dprime.std(),2),'dprime_t':round(td.statistic,2),'dprime_p':float(f"{td.pvalue:.1e}"),'dprime_le0_n':int((P.dprime<=0).sum()),
 'crit_mean':round(P.crit.mean(),2),'crit_ci':[round(P.crit.mean()-1.96*P.crit.sem(),2),round(P.crit.mean()+1.96*P.crit.sem(),2)],'crit_t':round(tc.statistic,2),'crit_p':float(f"{tc.pvalue:.1e}"),'crit_neg_n':int((P.crit<0).sum()),
 'paired_t':round(t.statistic,2),'paired_df':int(len(P)-1),'paired_p':float(f"{t.pvalue:.1e}"),'wilcoxon_p':float(f"{w.pvalue:.1e}"),'dz':round(dz,2),'n_true_better':int((P.acc_true>P.acc_false).sum()),
 'acc_participant_mean':round(P.acc.mean(),3),'acc_participant_sd':round(P.acc.std(),3),'acc_participant_min':round(P.acc.min(),2),'acc_participant_max':round(P.acc.max(),2),
 'n_above_70':int((P.acc>0.7).sum()),'n_50_70':int(((P.acc>=0.5)&(P.acc<=0.7)).sum()),'n_below_50':int((P.acc<0.5).sum()),
 'corr_acc_crit':round(stats.pearsonr(P.acc,P.crit)[0],2),'corr_acc_dprime':round(stats.pearsonr(P.acc,P.dprime)[0],2)}
# item level
It=L.groupby('item').agg(acc=('correct','mean'),label=('label','first'),n=('correct','size'),frac_true=('rated_true','mean'),genre=('genre','first'),modality=('modality','first'),leaning=('leaning','first')).reset_index()
def welch_df(a,b):  # SciPy < 1.11 does not return df for Welch's t-test
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b); return (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
tt=stats.ttest_ind(It[It.label==1].acc,It[It.label==0].acc,equal_var=False); mw=stats.mannwhitneyu(It[It.label==1].acc,It[It.label==0].acc)
N['item']={'true_mean':round(It[It.label==1].acc.mean(),3),'true_sd':round(It[It.label==1].acc.std(),3),'false_mean':round(It[It.label==0].acc.mean(),3),'false_sd':round(It[It.label==0].acc.std(),3),'welch_t':round(tt.statistic,2),'welch_df':round(getattr(tt,'df',welch_df(It[It.label==1].acc,It[It.label==0].acc)),1),'welch_p':float(f"{tt.pvalue:.1e}"),'mwu_p':float(f"{mw.pvalue:.1e}"),
 'majority_true_correct':int(((It.label==1)&(It.frac_true>0.5)).sum()),'majority_true_tie':int(((It.label==1)&(It.frac_true==0.5)).sum()),'majority_false_correct':int(((It.label==0)&(It.frac_true<0.5)).sum()),'majority_false_tie':int(((It.label==0)&(It.frac_true==0.5)).sum()),
 'majority_wrong_items':It[((It.frac_true>0.5)!=(It.label==1))&(It.frac_true!=0.5)][['item','genre','modality','frac_true']].round(2).to_dict('records')}
def gee(f,data):
    return smf.gee(f,groups='ID',data=data,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
m=gee('correct ~ label',L); N['gee_label']={'coef':round(m.params['label'],3),'se':round(m.bse['label'],3),'or':round(np.exp(m.params['label']),2),'p':float(f"{m.pvalues['label']:.1e}")}
m=gee('rated_true ~ label',L); b=m.params; N['gee_rated_true']={'p_true_given_false':round(1/(1+np.exp(-b['Intercept'])),3),'p_true_given_true':round(1/(1+np.exp(-(b['Intercept']+b['label']))),3),'or':round(np.exp(b['label']),2),'p':float(f"{m.pvalues['label']:.1e}")}
# ---------------- modality / genre ----------------
f=It[It.label==0]; tv=stats.ttest_ind(f[f.modality=='video'].acc,f[f.modality!='video'].acc,equal_var=False); tr=It[It.label==1]; tv2=stats.ttest_ind(tr[tr.modality=='video'].acc,tr[tr.modality!='video'].acc,equal_var=False)
m=gee('correct ~ label*is_video',L)
N['modality']={'false_video_acc':round(f[f.modality=='video'].acc.mean(),3),'false_video_n':int((f.modality=='video').sum()),'false_nonvideo_acc':round(f[f.modality!='video'].acc.mean(),3),'false_nonvideo_n':int((f.modality!='video').sum()),'false_video_welch_t':round(tv.statistic,2),'false_video_welch_p':round(tv.pvalue,3),
 'true_video_acc':round(tr[tr.modality=='video'].acc.mean(),3),'true_video_n':int((tr.modality=='video').sum()),'true_nonvideo_acc':round(tr[tr.modality!='video'].acc.mean(),3),'true_video_welch_p':round(tv2.pvalue,3),
 'gee_interaction_coef':round(m.params['label:is_video'],2),'gee_interaction_z':round(m.tvalues['label:is_video'],2),'gee_interaction_p':float(f"{m.pvalues['label:is_video']:.1e}"),'gee_video_main_coef':round(m.params['is_video'],2),'gee_video_main_z':round(m.tvalues['is_video'],2),
 'by_label_modality':It.assign(modality=It.modality.replace({'text+image':'text'})).groupby(['label','modality']).acc.agg(['mean','count']).round(3).reset_index().to_dict('records'),
 'anova_modality_all_items_p':round(stats.f_oneway(*[gg.acc.values for _,gg in It.assign(modality=It.modality.replace({'text+image':'text'})).groupby('modality')]).pvalue,3)}
N['genre']={'false_by_genre':f.groupby('genre').acc.agg(['mean','count']).round(3).reset_index().to_dict('records'),'true_by_genre':tr.groupby('genre').acc.agg(['mean','count']).round(3).reset_index().to_dict('records'),
 'kruskal_false_p':round(stats.kruskal(*[gg.acc.values for _,gg in f.groupby('genre')]).pvalue,3),
 'false_warning_majority_wrong':int(((f.genre=='warning')&(f.frac_true>0.5)).sum()),'false_warning_n':int((f.genre=='warning').sum()),'false_political_video_majority_wrong':int(((f.genre=='political')&(f.modality=='video')&(f.frac_true>0.5)).sum()),'false_political_video_n':int(((f.genre=='political')&(f.modality=='video')).sum()),
 'political_vs_other_welch_p':round(stats.ttest_ind(It[It.genre=='political'].acc,It[It.genre!='political'].acc,equal_var=False).pvalue,3)}
# ---------------- position / week ----------------
W=L.pivot_table(index='ID',columns='week',values='correct',aggfunc='mean').dropna()
N['week']={'week1_acc':round(L[L.week==1].correct.mean(),3),'week2_acc':round(L[L.week==2].correct.mean(),3),'paired_p':round(stats.ttest_rel(W[1],W[2]).pvalue,3),'paired_n':int(len(W)),'item_level_welch_p':round(stats.ttest_ind(It[It.item.isin([f'LC{i}' for i in range(1,21)])].acc,It[~It.item.isin([f'LC{i}' for i in range(1,21)])].acc,equal_var=False).pvalue,3),'positions_per_item':int(pd.crosstab(L.item,L.position).gt(0).sum(axis=1).max())}
# ---------------- stability ----------------
W2=L.pivot_table(index='ID',columns='week',values='correct',aggfunc=['mean','size']).dropna(); W2.columns=['a1','a2','n1','n2']; W2=W2[(W2.n1>=5)&(W2.n2>=5)]
L['odd']=L.position%2; SH=L.pivot_table(index='ID',columns='odd',values='correct',aggfunc='mean').dropna(); rsh=stats.pearsonr(SH[0],SH[1])[0]
N['stability']={'week_r':round(stats.pearsonr(W2.a1,W2.a2)[0],2),'week_r_p':round(stats.pearsonr(W2.a1,W2.a2)[1],3),'week_n':int(len(W2)),'week_rho':round(stats.spearmanr(W2.a1,W2.a2)[0],2),'oddeven_r':round(rsh,2)}
# ---------------- demographics ----------------
D=L.groupby('ID').agg(acc=('correct','mean'),age=('age','first'),gender=('gender','first'),edu=('edu_years','first'),polnum=('political_pref','first'),crt=('crt','first'),pol=('pol','first'),internet=('internet_use','first'),relig=('religious','first'))
Dd=D.dropna(subset=['age','gender','edu','polnum'])
ols=smf.ols('acc ~ age + gender + edu + polnum + crt',data=Dd).fit()
N['demographics']={'female':int((D.gender==0).sum()),'male':int((D.gender==1).sum()),'gender_na':int(D.gender.isna().sum()),'age_mean':round(D.age.mean(),1),'age_sd':round(D.age.std(),1),'age_min':int(D.age.min()),'age_max':int(D.age.max()),'edu_mean':round(D.edu.mean(),1),'edu_sd':round(D.edu.std(),1),
 'left':int((D.pol=='Left').sum()),'right':int((D.pol=='Right').sum()),'center':int((D.pol=='Center').sum()),'pol_na':int(D.pol.isna().sum()),'polnum_mean':round(D.polnum.mean(),2),'crt_dist':{int(k):int(v) for k,v in D.crt.value_counts().sort_index().items()},'internet_ge4':int((D.internet>=4).sum()),'religious_yes':int((D.relig==1).sum()),
 'ols_n':int(ols.nobs),'ols_r2':round(ols.rsquared,3),'ols_f_p':round(ols.f_pvalue,3),'ols_max_abs_t':round(ols.tvalues.drop('Intercept').abs().max(),2),
 'acc_by_pol':D.groupby('pol').acc.agg(['mean','std','count']).round(3).reset_index().to_dict('records'),'anova_pol_p':round(stats.f_oneway(*[gg.acc.values for _,gg in D.groupby('pol')]).pvalue,3),
 'corr_crt_acc':[round(x,3) for x in stats.pearsonr(D.crt.fillna(D.crt.mean()),D.acc)],'corr_age_acc':[round(x,3) for x in stats.pearsonr(Dd.age,Dd.acc)],'corr_edu_acc':[round(x,3) for x in stats.pearsonr(Dd.edu,Dd.acc)],
 'political_items_acc_by_pol':L[L.genre=='political'].groupby('pol').correct.mean().round(3).to_dict()}
# ---------------- congruence ----------------
K=L[(L.leaning!='none')&(L.label==0)&(L.pol.isin(['Left','Right']))].copy(); K['congruent']=(((K.leaning=='left')&(K.pol=='Left'))|((K.leaning=='right')&(K.pol=='Right'))).astype(int)
m=gee('rated_true ~ congruent',K)
N['congruence']={'items':sorted(K.item.unique().tolist()),'n_ratings':int(len(K)),'congruent_true':round(K[K.congruent==1].rated_true.mean(),3),'congruent_n':int((K.congruent==1).sum()),'incongruent_true':round(K[K.congruent==0].rated_true.mean(),3),'incongruent_n':int((K.congruent==0).sum()),
 'gee_or':round(np.exp(m.params['congruent']),2),'gee_coef':round(m.params['congruent'],2),'gee_se':round(m.bse['congruent'],2),'gee_p':round(m.pvalues['congruent'],3),
 'left_fakes_left_true':round(K[(K.leaning=='left')&(K.pol=='Left')].rated_true.mean(),3),'left_fakes_right_true':round(K[(K.leaning=='left')&(K.pol=='Right')].rated_true.mean(),3),
 'right_fake_right_true':round(K[(K.leaning=='right')&(K.pol=='Right')].rated_true.mean(),3),'right_fake_left_true':round(K[(K.leaning=='right')&(K.pol=='Left')].rated_true.mean(),3),
 'per_item':K.groupby(['item','leaning','pol']).rated_true.agg(['mean','size']).round(2).reset_index().to_dict('records')}
# ---------------- crowd simulations ----------------
R=np.load(f'{A}/R.npy'); lab=np.load(f'{A}/lab.npy'); n_p=R.shape[0]; Cm=np.where(np.isnan(R),np.nan,(R==lab).astype(float))
def crowd_per_item(members,idx=None,w=None):
    idx=np.arange(40) if idx is None else idx; Rm=R[np.ix_(members,idx)]
    Wm=np.ones_like(Rm) if w is None else np.repeat(np.asarray(w)[members][:,None],len(idx),1)
    Wm=np.where(np.isnan(Rm),0,Wm); s=(Wm*(np.nan_to_num(Rm)*2-1)).sum(0); seen=Wm.sum(0)>0
    v=np.where(s>0,1,np.where(s<0,0,0.5)); c=np.where(v==0.5,0.5,(v==lab[idx]).astype(float)); return np.where(seen,c,np.nan)
# participant leaning aligned to R rows: R rows are sorted original IDs; long.csv IDs were pseudonymized preserving row-level attributes, so rebuild pol vector from the pseudonymized long via the same sorted order is impossible -> use the ordering saved in R by recomputing from long: R was built as pivot over sorted(ID) of ORIGINAL ids. We therefore rebuild R here from long.csv to stay self-contained.
ids=sorted(L.ID.unique()); items=[f'LC{i}' for i in range(1,41)]
R=L.pivot(index='ID',columns='item',values='rated_true').reindex(index=ids,columns=items).values.astype(float); lab=L.groupby('item').label.first().reindex(items).values; n_p=len(ids)
polv=L.groupby('ID').pol.first().reindex(ids).values; Lidx=np.where(polv=='Left')[0]; Ridx=np.where(polv=='Right')[0]
pol_mask=I.set_index('item').reindex(items).leaning.ne('none').values
comp={}; sizes_c=[2,4,6,8,10,12,14]
for s in sizes_c:
    comp[s]={}
    for name,sampler in {'allL':lambda: rng.choice(Lidx,s,replace=False),'allR':lambda: rng.choice(Ridx,s,replace=False),'balanced':lambda: np.concatenate([rng.choice(Lidx,s//2,replace=False),rng.choice(Ridx,s//2,replace=False)]),'random':lambda: rng.choice(n_p,s,replace=False)}.items():
        pis=np.array([crowd_per_item(sampler()) for _ in range(1500)])
        comp[s][name]={'all':round(np.nanmean(pis),3),'political':round(np.nanmean(pis[:,pol_mask]),3),'nonpolitical':round(np.nanmean(pis[:,~pol_mask]),3),'false':round(np.nanmean(pis[:,lab==0]),3),'political_lo':round(np.nanpercentile(np.nanmean(pis[:,pol_mask],axis=1),2.5),3),'political_hi':round(np.nanpercentile(np.nanmean(pis[:,pol_mask],axis=1),97.5),3)}
N['composition']={'left_n':int(len(Lidx)),'right_n':int(len(Ridx)),'n_political_items':int(pol_mask.sum()),'political_items':[it for it,m in zip(items,pol_mask) if m],'by_size':comp}
curves=pd.read_csv(f'{A}/aggregation_curves.csv'); piv=curves.pivot(index='size',columns='scheme',values='mean')
N['aggregation']={'unweighted':{int(s):round(piv.loc[s,'unweighted'],3) for s in [3,5,7,10,15,20,38]},'in_sample':{int(s):round(piv.loc[s,'acc_in_sample(linear)'],3) for s in [3,5,7,10,15,20,38]},'loo':{int(s):round(piv.loc[s,'acc_LOO(linear)'],3) for s in [3,5,7,10,15,20,38]},'crt':{int(s):round(piv.loc[s,'CRT+1'],3) for s in [7,15,38]}}
full=crowd_per_item(np.arange(n_p)); N['aggregation']['full_unweighted']=round(np.nanmean(full),3); N['aggregation']['full_unweighted_true']=round(np.nanmean(full[lab==1]),3); N['aggregation']['full_unweighted_false']=round(np.nanmean(full[lab==0]),3)
b=[np.nanmean(crowd_per_item(rng.choice(n_p,n_p,replace=True))) for _ in range(3000)]; N['aggregation']['full_unweighted_boot_ci']=[round(np.percentile(b,2.5),3),round(np.percentile(b,97.5),3)]
# in-sample weighted full crowd (leakage demo)
acc_in=np.nansum(Cm:=np.where(np.isnan(R),np.nan,(R==lab).astype(float)),axis=1)/np.sum(~np.isnan(R),axis=1)
N['aggregation']['full_in_sample_weighted']=round(np.nanmean(crowd_per_item(np.arange(n_p),w=acc_in)),3)
# split-half CV
def acc_on(idx):
    c=Cm[:,idx]; n=np.sum(~np.isnan(c),axis=1); return np.where(n>0,np.nansum(c,axis=1)/np.maximum(n,1),np.nan),n
def logit(a): a=np.clip(a,0.05,0.95); return np.log(a/(1-a))
T=np.where(lab==1)[0]; F=np.where(lab==0)[0]; rows=[]; rs=[]
for rep in range(1500):
    Aidx=np.concatenate([rng.choice(T,10,replace=False),rng.choice(F,10,replace=False)]); B=np.setdiff1d(np.arange(40),Aidx)
    aA,nA=acc_on(Aidx); aB,nB=acc_on(B); ok=(nA>=4)&(nB>=4); rs.append(np.corrcoef(aA[ok],aB[ok])[0,1])
    wlin=np.where(np.isnan(aA),np.nanmean(aA),aA); row={'unw':np.nanmean(crowd_per_item(np.arange(n_p),B)),'wlin':np.nanmean(crowd_per_item(np.arange(n_p),B,wlin)),'wlog':np.nanmean(crowd_per_item(np.arange(n_p),B,logit(wlin)))}
    elig=np.where(nA>=4)[0]; order=elig[np.argsort(-aA[elig])]
    for k in [3,5,7,10,15]: row[f'top{k}']=np.nanmean(crowd_per_item(order[:k],B)); row[f'rand{k}']=np.nanmean(crowd_per_item(rng.choice(n_p,k,replace=False),B))
    rows.append(row)
CV=pd.DataFrame(rows); rs=np.array(rs)
r_half=float(np.nanmean(rs)); n_half=8.6
N['cv']={'split_half_r':round(r_half,2),'split_half_r_lo':round(np.nanpercentile(rs,2.5),2),'split_half_r_hi':round(np.nanpercentile(rs,97.5),2),'share_r_positive':round(float(np.mean(rs>0)),2),
 'unw_mean':round(CV.unw.mean(),3),'wlin_mean':round(CV.wlin.mean(),3),'wlog_mean':round(CV.wlog.mean(),3),'gain_lin':round((CV.wlin-CV.unw).mean(),3),'gain_log':round((CV.wlog-CV.unw).mean(),3),'p_gain_lin_pos':round(float(np.mean(CV.wlin>CV.unw)),2),'p_gain_lin_neg':round(float(np.mean(CV.wlin<CV.unw)),2),
 'topk':{k:{'top':round(CV[f'top{k}'].mean(),3),'top_lo':round(CV[f'top{k}'].quantile(.025),3),'top_hi':round(CV[f'top{k}'].quantile(.975),3),'random':round(CV[f'rand{k}'].mean(),3),'p_beats_random':round(float(np.mean(CV[f'top{k}']>CV[f'rand{k}'])),2),'p_beats_full':round(float(np.mean(CV[f'top{k}']>CV.unw)),2)} for k in [3,5,7,10,15]},
 'reliability_projection':{n:round(( (n/n_half)*r_half/(1+((n/n_half)-1)*r_half) ),2) for n in [20,40,80,120,200]}}
# ---------------- feedback ----------------
maj=It.set_index('item').frac_true; L['maj_true']=(L.item.map(maj)>0.5).astype(int); L['tie']=(L.item.map(maj)==0.5).astype(int); L['minority']=((L.rated_true!=L.maj_true)&(L.tie==0)).astype(int)
L=L.sort_values(['ID','position']); 
for c in ['position','rated_true','correct','label','item']: L[f'next_{c}']=L.groupby('ID')[c].shift(-1)
Nn=L[L.next_position==L.position+1].copy(); Nn['switch']=(Nn.next_rated_true!=Nn.rated_true).astype(int)
m1=gee('next_rated_true ~ rated_true*minority + next_label',Nn); m2=gee('next_rated_true ~ minority + rated_true + next_label',Nn); m3=gee('next_correct ~ minority + next_label',Nn)
cells=Nn.groupby(['rated_true','minority']).agg(p_true_next=('next_rated_true','mean'),acc_next=('next_correct','mean'),k=('next_rated_true','sum'),n=('next_rated_true','size'),switch=('switch','mean')).reset_index()
cells['ci_lo']=[wilson(k,n)[0] for k,n in zip(cells.k,cells.n)]; cells['ci_hi']=[wilson(k,n)[1] for k,n in zip(cells.k,cells.n)]
N['feedback']={'pairs':int(len(Nn)),'minority_share':round(Nn.minority.mean(),3),'minority_n':int(Nn.minority.sum()),'cells':cells.round(3).to_dict('records'),
 'gee_main_minority_coef':round(m2.params['minority'],2),'gee_main_minority_or':round(np.exp(m2.params['minority']),2),'gee_main_minority_p':round(m2.pvalues['minority'],3),
 'gee_interaction_minority_coef':round(m1.params['minority'],2),'gee_interaction_minority_p':round(m1.pvalues['minority'],3),'gee_interaction_term_coef':round(m1.params['rated_true:minority'],2),'gee_interaction_term_p':round(m1.pvalues['rated_true:minority'],3),
 'nextacc_minority_coef':round(m3.params['minority'],2),'nextacc_minority_p':round(m3.pvalues['minority'],3),'majority_true_items':int((It.frac_true>0.5).sum()),'mean_frac_true_shown':round(It.frac_true.mean(),3)}
Nn['perf']=pd.cut(Nn.ID.map(P.set_index('ID').acc),[0,0.55,0.7,1],labels=['low','mid','high'])
N['feedback']['by_perf']=Nn.groupby(['perf','minority'],observed=True).agg(switch=('switch','mean'),acc_next=('next_correct','mean'),n=('switch','size')).round(3).reset_index().astype({'perf':str}).to_dict('records')
# ================= FIGURES =================
# Fig 2: SDT
fig,ax=plt.subplots(1,2,figsize=(6.5,3.0),gridspec_kw={'width_ratios':[1.15,1],'wspace':0.35})
a=ax[0]; a.plot([0,1],[0,1],ls='--',c=C_GREY,lw=1); a.scatter(P.fa,P.hit,s=18+2*(P.n-9),c='k',alpha=0.65,edgecolor='none')
a.scatter([P.fa.mean()],[P.hit.mean()],marker='D',s=60,c='none',edgecolor=C_FALSE,lw=1.6,zorder=5)
a.set_xlabel('False-alarm rate'); a.set_ylabel('Hit rate'); a.set_xlim(-0.02,1.02); a.set_ylim(-0.02,1.02); a.set_aspect('equal')
a.text(0.03,0.93,'(a)',transform=a.transAxes,fontweight='bold'); a.text(0.62,0.08,'chance',c=C_GREY,rotation=45,fontsize=7)
a.annotate('mean',(P.fa.mean(),P.hit.mean()),xytext=(P.fa.mean()+0.13,P.hit.mean()-0.12),fontsize=7,arrowprops=dict(arrowstyle='-',lw=0.6,color=C_FALSE),color=C_FALSE)
b=ax[1]; jit=rng.uniform(-0.12,0.12,len(P)); b.scatter(jit,P.crit,c='k',alpha=0.6,s=16,edgecolor='none'); b.axhline(0,ls='--',c=C_GREY,lw=1)
b.errorbar([0.35],[P.crit.mean()],yerr=[[P.crit.mean()-N['sdt']['crit_ci'][0]],[N['sdt']['crit_ci'][1]-P.crit.mean()]],fmt='D',c=C_FALSE,capsize=3,ms=5)
b.set_xlim(-0.5,0.7); b.set_xticks([]); b.set_ylabel("Criterion c"); b.text(0.03,0.93,'(b)',transform=b.transAxes,fontweight='bold')
b.text(0.4,P.crit.mean()+0.05,f"mean {P.crit.mean():.2f}\n95% CI [{N['sdt']['crit_ci'][0]:.2f}, {N['sdt']['crit_ci'][1]:.2f}]",fontsize=7,color=C_FALSE)
fig.savefig(f'{IMG}/fig_sdt.pdf'); fig.savefig(f'{IMG}/fig_sdt.png'); plt.close(fig)
# Fig 3: items
Is=It.sort_values('frac_true').reset_index(drop=True); mk={'image':'o','video':'s','text':'^','text+image':'D'}
fig,a=plt.subplots(figsize=(6.5,6.2))
for i,r in Is.iterrows():
    lo,hi=wilson(round(r.frac_true*r.n),r.n); col=C_TRUE if r.label==1 else C_FALSE
    a.plot([lo,hi],[i,i],c=col,lw=0.8,alpha=0.5); a.scatter(r.frac_true,i,marker=mk[r.modality],c=col,s=28,edgecolor='k',lw=0.4,zorder=3)
a.axvline(0.5,ls='--',c=C_GREY,lw=1); a.set_yticks(range(len(Is))); a.set_yticklabels([f"{r['item']}  ({r['genre'][:4]})" for _,r in Is.iterrows()],fontsize=6.5)
a.set_xlabel("Share of raters who judged the item 'true' (95% Wilson CI)"); a.set_xlim(0,1); a.set_ylim(-1,len(Is))
from matplotlib.lines import Line2D
h=[Line2D([],[],marker='o',ls='',c=C_TRUE,label='true item'),Line2D([],[],marker='o',ls='',c=C_FALSE,label='false item'),Line2D([],[],marker='o',ls='',c='w',markeredgecolor='k',label='image'),Line2D([],[],marker='s',ls='',c='w',markeredgecolor='k',label='video'),Line2D([],[],marker='^',ls='',c='w',markeredgecolor='k',label='text'),Line2D([],[],marker='D',ls='',c='w',markeredgecolor='k',label='text+image')]
a.legend(handles=h,loc='lower right',frameon=True,fontsize=7,ncol=2)
a.text(0.52,len(Is)-0.5,"majority says 'true' →",fontsize=7,color=C_GREY,va='top'); a.text(0.48,len(Is)-0.5,"← majority says 'false'",fontsize=7,color=C_GREY,va='top',ha='right')
fig.savefig(f'{IMG}/fig_items.pdf'); fig.savefig(f'{IMG}/fig_items.png'); plt.close(fig)
# Fig 4: congruence + composition
fig,ax=plt.subplots(1,3,figsize=(6.8,2.7),gridspec_kw={'width_ratios':[0.8,1.1,1.1],'wspace':0.42})
a=ax[0]; groups=[('Congruent\n(item favours\nown side)',K[K.congruent==1]),('Incongruent\n(item favours\nother side)',K[K.congruent==0])]
for x,(lbl,gp) in enumerate(groups):
    k=gp.rated_true.sum(); n=len(gp); lo,hi=wilson(k,n); a.bar(x,k/n,color=[C_FALSE,C_GREY][x],width=0.6); a.errorbar(x,k/n,yerr=[[k/n-lo],[hi-k/n]],fmt='none',c='k',capsize=3); a.text(x,hi+0.03,f"{k/n:.2f}\nn={n}",ha='center',fontsize=7)
a.set_xticks([0,1]); a.set_xticklabels([g[0] for g in groups],fontsize=7); a.set_ylabel("P(false political item rated 'true')"); a.set_ylim(0,0.85); a.text(0.03,0.93,'(a)',transform=a.transAxes,fontweight='bold')
for j,(key,title) in enumerate([('political','(b) partisan items (7)'),('nonpolitical','(c) all other items (33)')]):
    a=ax[j+1]
    for name,col,ls,lbl in [('allL',C_L,'-','all Left'),('allR',C_R,'-','all Right'),('balanced',C_BAL,'-','balanced L/R'),('random',C_GREY,'--','random')]:
        a.plot(sizes_c,[comp[s][name][key] for s in sizes_c],c=col,ls=ls,marker='o',ms=3,label=lbl)
    a.set_xlabel('Crowd size'); a.set_ylim(0.4,0.95); a.set_title(title,loc='left',fontsize=9,fontweight='bold'); a.set_xticks(sizes_c)
    if j==0: a.set_ylabel('Majority accuracy'); a.legend(frameon=False,fontsize=7,loc='lower right')
fig.savefig(f'{IMG}/fig_congruence.pdf'); fig.savefig(f'{IMG}/fig_congruence.png'); plt.close(fig)
# Fig 5: aggregation
fig,ax=plt.subplots(1,2,figsize=(6.5,2.9),gridspec_kw={'width_ratios':[1.3,1]})
a=ax[0]
for scheme,col,lbl in [('unweighted',C_GREY,'unweighted majority'),('acc_in_sample(linear)',C_FALSE,'accuracy-weighted, in-sample (leaks)'),('acc_LOO(linear)',C_TRUE,'accuracy-weighted, held-out items')]:
    d=curves[curves.scheme==scheme].sort_values('size'); a.plot(d['size'],d['mean'],c=col,label=lbl,lw=1.5); a.fill_between(d['size'],d['lo'],d['hi'],color=col,alpha=0.12,lw=0)
a.set_xlabel('Crowd size (sampled without replacement)'); a.set_ylabel('Majority accuracy on 40 items'); a.set_ylim(0.5,1.0); a.legend(frameon=False,fontsize=7,loc='lower right'); a.text(0.03,0.93,'(a)',transform=a.transAxes,fontweight='bold')
b=ax[1]; ks=[3,5,7,10,15]
b.errorbar(ks,[N['cv']['topk'][k]['top'] for k in ks],yerr=[[N['cv']['topk'][k]['top']-N['cv']['topk'][k]['top_lo'] for k in ks],[N['cv']['topk'][k]['top_hi']-N['cv']['topk'][k]['top'] for k in ks]],fmt='o-',c=C_TRUE,capsize=2,ms=4,label='top-k by held-out accuracy')
b.plot(ks,[N['cv']['topk'][k]['random'] for k in ks],'s--',c=C_GREY,ms=4,label='random k')
b.axhline(N['cv']['unw_mean'],c='k',lw=1,ls=':'); b.text(15.3,N['cv']['unw_mean']+0.01,'full crowd (38)',fontsize=7,ha='right')
b.set_xlabel('Panel size k'); b.set_ylabel('Accuracy on held-out half of items'); b.set_ylim(0.5,1.0); b.set_xticks(ks); b.legend(frameon=False,fontsize=7,loc='lower right'); b.text(0.03,0.93,'(b)',transform=b.transAxes,fontweight='bold')
fig.savefig(f'{IMG}/fig_aggregation.pdf'); fig.savefig(f'{IMG}/fig_aggregation.png'); plt.close(fig)
# Fig 6: feedback
fig,a=plt.subplots(figsize=(6.5,2.8)); xs=[]; labels=[]
order=[(0,0,"said 'false',\ncommunity agreed"),(0,1,"said 'false',\ncommunity disagreed"),(1,0,"said 'true',\ncommunity agreed"),(1,1,"said 'true',\ncommunity disagreed")]
for x,(rt,mn,lbl) in enumerate(order):
    c=cells[(cells.rated_true==rt)&(cells.minority==mn)].iloc[0]; col=C_FALSE if mn==1 else C_GREY
    a.bar(x,c.p_true_next,color=col,width=0.6); a.errorbar(x,c.p_true_next,yerr=[[c.p_true_next-c.ci_lo],[c.ci_hi-c.p_true_next]],fmt='none',c='k',capsize=3)
    a.text(x,max(c.ci_hi,c.acc_next)+0.03,f"{c.p_true_next:.2f}\nn={int(c.n)}",ha='center',fontsize=7); a.scatter(x,c.acc_next,marker='_',s=300,c='k',zorder=5)
    xs.append(x); labels.append(lbl)
a.set_xticks(xs); a.set_xticklabels(labels,fontsize=7.5); a.set_ylabel("P(next item rated 'true')"); a.set_ylim(0,1.0); a.axhline(0.5,ls=':',c=C_GREY,lw=0.8)
a.scatter([],[],marker='_',s=200,c='k',label='accuracy on next item'); a.legend(frameon=False,fontsize=7,loc='upper left')
fig.savefig(f'{IMG}/fig_feedback.pdf'); fig.savefig(f'{IMG}/fig_feedback.png'); plt.close(fig)
def conv(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,np.ndarray): return o.tolist()
    return str(o)
json.dump(N,open(f'{A}/numbers.json','w'),indent=1,default=conv)
print(json.dumps({k:N[k] for k in ['sdt','item','modality','congruence','cv','feedback']},indent=0,default=conv)[:6000])
print("\nFIGURES + numbers.json written")
