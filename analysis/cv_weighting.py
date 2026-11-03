import numpy as np, pandas as pd
rng=np.random.default_rng(7)
R=np.load('R.npy'); lab=np.load('lab.npy'); C=np.where(np.isnan(R),np.nan,(R==lab).astype(float)); n_p=R.shape[0]
def acc_on(idx):
    c=C[:,idx]; n=np.sum(~np.isnan(c),axis=1); return np.where(n>0,np.nansum(c,axis=1)/np.maximum(n,1),np.nan), n
def maj(members, idx, w=None):
    Rm=R[np.ix_(members,idx)]; Wm=np.ones_like(Rm) if w is None else np.repeat(w[members][:,None],len(idx),1)
    Wm=np.where(np.isnan(Rm),0,Wm); s=(Wm*(np.nan_to_num(Rm)*2-1)).sum(0); seen=Wm.sum(0)>0
    v=np.where(s>0,1,np.where(s<0,0,0.5)); c=np.where(v==0.5,0.5,(v==lab[idx]).astype(float)); return np.nanmean(np.where(seen,c,np.nan))
def logit(a): a=np.clip(a,0.05,0.95); return np.log(a/(1-a))
T=np.where(lab==1)[0]; F=np.where(lab==0)[0]; res=[]; rs=[]
for rep in range(2000):
    A=np.concatenate([rng.choice(T,10,replace=False),rng.choice(F,10,replace=False)]); B=np.setdiff1d(np.arange(40),A)
    aA,nA=acc_on(A); aB,nB=acc_on(B); ok=(nA>=4)&(nB>=4); rs.append(np.corrcoef(aA[ok],aB[ok])[0,1])
    wlin=np.where(np.isnan(aA),np.nanmean(aA),aA); wlog=logit(wlin)
    row=dict(unw=maj(np.arange(n_p),B), wlin=maj(np.arange(n_p),B,wlin), wlog=maj(np.arange(n_p),B,wlog))
    elig=np.where(nA>=4)[0]; order=elig[np.argsort(-aA[elig])]
    for k in [3,5,7,10,15]:
        row[f'top{k}']=maj(order[:k],B); row[f'rand{k}']=maj(rng.choice(np.arange(n_p),k,replace=False),B); row[f'wlog{k}']=maj(rng.choice(np.arange(n_p),k,replace=False),B,wlog)
    res.append(row)
D=pd.DataFrame(res); rs=np.array(rs)
print("=== Random stratified 20/20 item splits (2000 reps): weights/selection from half A, evaluated on half B ===")
print(f"corr(acc_A, acc_B) across participants: mean r={np.nanmean(rs):.3f} (95% of splits: {np.nanpercentile(rs,2.5):.2f}..{np.nanpercentile(rs,97.5):.2f}); share of splits with r>0: {np.mean(rs>0):.2f}")
def s(c): return f"{D[c].mean():.3f} [{D[c].quantile(.025):.3f},{D[c].quantile(.975):.3f}]"
print(f"full crowd on B: unweighted {s('unw')} | acc-weighted linear {s('wlin')} | log-odds {s('wlog')}")
print(f"  paired diff wlin-unw: {(D.wlin-D.unw).mean():+.3f} (P>0: {np.mean(D.wlin>D.unw):.2f}, P<0: {np.mean(D.wlin<D.unw):.2f}); wlog-unw: {(D.wlog-D.unw).mean():+.3f} (P>0: {np.mean(D.wlog>D.unw):.2f})")
for k in [3,5,7,10,15]:
    print(f"k={k:2d}: top-k panel {s(f'top{k}')} | random-k unweighted {s(f'rand{k}')} | random-k log-odds-weighted {s(f'wlog{k}')} | P(top-k > full crowd)={np.mean(D[f'top{k}']>D.unw):.2f} | P(top-k > random-k)={np.mean(D[f'top{k}']>D[f'rand{k}']):.2f}")
# reliability extrapolation via Spearman-Brown from split-half of ~half the items each
r_half=np.nanmean(rs); n_half=np.nanmean([np.nanmean(acc_on(np.arange(20))[1])])  # approx items per half per participant
print(f"\nmean items per participant per half ~{n_half:.1f}; split-half r={r_half:.3f}")
for n in [20,40,80,120,200]:
    f=n/n_half; print(f"  projected reliability of individual accuracy with {n:3d} items: {f*r_half/(1+(f-1)*r_half):.2f}")
