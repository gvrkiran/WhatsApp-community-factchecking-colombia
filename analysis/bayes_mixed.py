import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as BB
L=pd.read_csv('long_enriched.csv', dtype={'ID':str})
def fit(f, data, vc={'ID':'0+C(ID)','item':'0+C(item)'}, name=''):
    m=BB.from_formula(f, vc, data).fit_vb(); t=m.summary().tables[0] if hasattr(m.summary(),'tables') else None
    print(f"--- {name or f} (N={len(data)}) ---"); 
    fe=[(n,) for n in m.model.exog_names]; 
    print(pd.DataFrame({'coef':m.fe_mean,'sd':m.fe_sd,'z':m.fe_mean/m.fe_sd}, index=m.model.exog_names).round(3))
    print("  VC sd (ID, item):", np.exp(m.vcp_mean).round(2))
fit('correct ~ label', L, name='M1 correct ~ label')
fit('rated_true ~ label', L, name='M2 rated_true ~ label')
fit('correct ~ label*is_video', L, name='M3 correct ~ label x video')
fit('correct ~ is_video', L[L.label==0], name='M3b false items: correct ~ video')
K=L[(L.leaning!='none')&(L.label==0)&(L.pol.isin(['Left','Right']))].copy(); K['congruent']=(((K.leaning=='left')&(K.pol=='Left'))|((K.leaning=='right')&(K.pol=='Right'))).astype(int)
fit('rated_true ~ congruent', K, name='M7 congruence (fake leaning items)')
N=L[L.next_pos==L.position+1].copy(); N['next_item']=N.next_item.astype(str)
fit('next_rt ~ rated_true*minority + next_label', N, vc={'ID':'0+C(ID)','next_item':'0+C(next_item)'}, name='M8 next response ~ own x minority')
fit('next_correct ~ minority + next_label', N, vc={'ID':'0+C(ID)','next_item':'0+C(next_item)'}, name='M9 next correct ~ minority')
fit('correct ~ label + week', L, name='M10 correct ~ label + week (item RE absorbs; week is item-level)')
