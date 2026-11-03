import pandas as pd, numpy as np
rng=np.random.default_rng(2026)
L=pd.read_csv('long.csv', dtype={'ID':str})
ids=sorted(L.ID.unique()); items=[f'LC{i}' for i in range(1,41)]
R=L.pivot(index='ID', columns='item', values='rated_true').reindex(index=ids, columns=items).values.astype(float)  # NaN if unseen
lab=L.groupby('item').label.first().reindex(items).values
C=np.where(np.isnan(R), np.nan, (R==lab).astype(float))  # correctness matrix
n_i=np.nansum(~np.isnan(C),axis=1); tot=np.nansum(C,axis=1)
acc_in=tot/n_i                                   # in-sample accuracy (as in paper)
loo=np.where(np.isnan(C), np.nan, (tot[:,None]-np.nan_to_num(C))/(n_i[:,None]-1))  # leave-one-item-out accuracy per (participant,item)
week=np.array([1]*20+[2]*20); 
def acc_on(mask): 
    c=np.where(mask[None,:], C, np.nan); n=np.nansum(~np.isnan(c),axis=1); return np.where(n>0, np.nansum(c,axis=1)/np.maximum(n,1), np.nan), n
acc_w1,n_w1=acc_on(week==1); acc_w2,n_w2=acc_on(week==2)
crossweek=np.zeros_like(C); crossweek[:,week==2]=acc_w1[:,None]; crossweek[:,week==1]=acc_w2[:,None]  # weights for week-2 items from week-1 performance and vice versa
demo=L.groupby('ID')[['crt1','crt2','crt3','pol_pref_2']].first().reset_index()  # covariates from long.csv (same study codes)
demo['crt']=(demo.crt1==500).astype(int)+(demo.crt2==2).astype(int)+(demo.crt3.astype(str).str.lower()=='emily').astype(int)
demo.loc[demo.crt1.isna(),'crt']=float('nan')  # no pre-survey: filled with the mean below
crt=demo.set_index('ID').crt.reindex(ids).fillna(demo.crt.mean()).values
pol=demo.set_index('ID').pol_pref_2.reindex(ids).values  # 1 L 2 C 3 R
def logit(a): a=np.clip(a,0.05,0.95); return np.log(a/(1-a))
def crowd_acc(members, W, per_item=False):
    """members: index array; W: weight matrix (n_participants x n_items) or None. Weighted majority per item; ties -> 0.5 credit."""
    Rm=R[members]; Wm=np.ones_like(Rm) if W is None else W[members]
    Wm=np.where(np.isnan(Rm),0,Wm); V=np.nan_to_num(Rm)*2-1  # +1 true, -1 false
    s=(Wm*V).sum(axis=0); seen=(Wm.sum(axis=0)>0)
    verdict=np.where(s>0,1,np.where(s<0,0,0.5)); corr=np.where(verdict==0.5,0.5,(verdict==lab).astype(float))
    corr=np.where(seen,corr,np.nan)
    return corr if per_item else np.nanmean(corr)
schemes={'unweighted':None,'acc_in_sample(linear)':np.repeat(acc_in[:,None],40,1),'acc_LOO(linear)':loo,'acc_LOO(log-odds)':logit(loo),
         'acc_cross_week(linear)':crossweek,'acc_cross_week(log-odds)':logit(crossweek),'CRT+1':np.repeat((crt+1)[:,None],40,1)}
sizes=list(range(1,39)); NS=3000; out=[]
for name,W in schemes.items():
    for s in sizes:
        accs=[crowd_acc(rng.choice(len(ids),size=s,replace=False),W) for _ in range(NS if s<38 else 1)]
        out.append(dict(scheme=name,size=s,mean=np.mean(accs),lo=np.percentile(accs,2.5),hi=np.percentile(accs,97.5)))
res=pd.DataFrame(out); res.to_csv('aggregation_curves.csv',index=False)
piv=res.pivot(index='size',columns='scheme',values='mean').round(3)
print("=== Mean crowd accuracy by size (sampling WITHOUT replacement, 3000 crowds/size, ties=0.5) ===")
print(piv.loc[[1,3,5,7,10,15,20,25,30,38]].to_string())
print("\nfull crowd (38) per scheme:", piv.loc[38].to_dict())
# per item true vs false for full crowd unweighted
pi=crowd_acc(np.arange(len(ids)),None,per_item=True); print(f"\nfull crowd unweighted: acc on TRUE items={np.nanmean(pi[lab==1]):.3f}, on FALSE items={np.nanmean(pi[lab==0]):.3f}")
pi2=crowd_acc(np.arange(len(ids)),loo,per_item=True); print(f"full crowd LOO-weighted: TRUE={np.nanmean(pi2[lab==1]):.3f}, FALSE={np.nanmean(pi2[lab==0]):.3f}")
pi3=crowd_acc(np.arange(len(ids)),crossweek,per_item=True); print(f"full crowd cross-week-weighted: TRUE={np.nanmean(pi3[lab==1]):.3f}, FALSE={np.nanmean(pi3[lab==0]):.3f}")
print("\n=== WITH replacement (as in the paper's code), unweighted vs in-sample vs LOO at sizes 7,15,38 ===")
for s in [7,15,38]:
    for name in ['unweighted','acc_in_sample(linear)','acc_LOO(linear)']:
        W=schemes[name]; accs=[crowd_acc(rng.choice(len(ids),size=s,replace=True),W) for _ in range(2000)]
        print(f"  size {s:2d} {name:24s} {np.mean(accs):.3f}")
print("\n=== 'Expert panel' selected OUT-OF-SAMPLE: pick top-k by week-1 accuracy, evaluate unweighted majority on week-2 items (and vice versa) ===")
def panel_eval(train_acc, n_train, test_mask, k, random=False):
    elig=np.where(n_train>=5)[0]
    if random: mem=rng.choice(elig,size=k,replace=False)
    else: mem=elig[np.argsort(-train_acc[elig])][:k]
    c=crowd_acc(mem,None,per_item=True); return np.nanmean(c[test_mask])
for k in [3,5,7,10,15]:
    a12=panel_eval(acc_w1,n_w1,week==2,k); a21=panel_eval(acc_w2,n_w2,week==1,k)
    r12=np.mean([panel_eval(acc_w1,n_w1,week==2,k,random=True) for _ in range(2000)]); r21=np.mean([panel_eval(acc_w2,n_w2,week==1,k,random=True) for _ in range(2000)])
    print(f"  k={k:2d}: top-k(w1)->w2 items {a12:.3f} | random-k->w2 {r12:.3f} || top-k(w2)->w1 items {a21:.3f} | random-k->w1 {r21:.3f} || avg top {np.mean([a12,a21]):.3f} vs random {np.mean([r12,r21]):.3f}")
full_w2=np.nanmean(crowd_acc(np.arange(len(ids)),None,per_item=True)[week==2]); full_w1=np.nanmean(crowd_acc(np.arange(len(ids)),None,per_item=True)[week==1])
print(f"  reference: full-crowd unweighted majority on week-2 items {full_w2:.3f}, week-1 items {full_w1:.3f}")
print("\n=== PARTISAN COMPOSITION: homogeneous (all-Left / all-Right) vs balanced vs random crowds ===")
Lidx=np.where(pol==1)[0]; Ridx=np.where(pol==3)[0]; print(f"Left n={len(Lidx)}, Right n={len(Ridx)}")
polit=pd.read_csv('items.csv').set_index('item').reindex(items); pol_mask=(polit.genre=='political').values
for s in [4,6,8,10,12,14]:
    rows={}
    for name,sampler in {'all-Left':lambda: rng.choice(Lidx,s,replace=False),'all-Right':lambda: rng.choice(Ridx,s,replace=False),
                         'balanced':lambda: np.concatenate([rng.choice(Lidx,s//2,replace=False),rng.choice(Ridx,s//2,replace=False)]),'random':lambda: rng.choice(len(ids),s,replace=False)}.items():
        pis=np.array([crowd_acc(sampler(),None,per_item=True) for _ in range(2000)])
        rows[name]=(np.nanmean(pis), np.nanmean(pis[:,pol_mask]), np.nanmean(pis[:,~pol_mask]), np.nanmean(pis[:,(lab==0)]))
    print(f"size {s:2d}: " + " | ".join(f"{k}: all={v[0]:.3f} polit={v[1]:.3f} nonpol={v[2]:.3f} false={v[3]:.3f}" for k,v in rows.items()))
print("\n=== Bootstrap CI for full-crowd unweighted accuracy (resample participants with replacement) ===")
b=[crowd_acc(rng.choice(len(ids),size=len(ids),replace=True),None) for _ in range(3000)]; print(f"  {np.mean(b):.3f} [{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]")
b2=[crowd_acc(rng.choice(len(ids),size=len(ids),replace=True),loo) for _ in range(3000)]; print(f"  LOO-weighted: {np.mean(b2):.3f} [{np.percentile(b2,2.5):.3f},{np.percentile(b2,97.5):.3f}]")
d=np.array(b2)-np.array(b); print(f"  paired difference LOO - unweighted: {d.mean():.3f} [{np.percentile(d,2.5):.3f},{np.percentile(d,97.5):.3f}]; P(diff<=0)={np.mean(d<=0):.3f}")
np.save('R.npy',R); np.save('lab.npy',lab)
