import os, pandas as pd, numpy as np
# The raw files are keyed by WhatsApp number and are not shared; set RAW_DIR to rebuild long.csv from them.
# The released long.csv is this script's output with numbers replaced by study codes and ages truncated to whole years.
B=os.environ.get('RAW_DIR','raw')
d=pd.read_csv(f'{B}/data_woc_2025.csv', dtype={'ID':str})
order=pd.read_csv(f'{B}/groups_content_order.txt', sep='\t')
lab=pd.read_csv(f'{B}/content_true_false_labels.txt', sep='\t')
demo=pd.read_excel(f'{B}/Demographic Data.xlsx', sheet_name='dem_var')
demo['ID']=demo['id'].astype(str).str.split('.').str[0]
print("participants in data:", d.shape[0], "| unique IDs:", d.ID.nunique())
print("groups:", d.group.value_counts().to_dict())
rows=[]
for _,r in d.iterrows():
    g=r['group']; o=order[order.group==g].iloc[0]
    for i in range(1,21):
        resp=r[f'e{i}']; corr=r[f'a{i}']
        if pd.isna(corr) or str(resp).strip() in ('','______','nan'): 
            continue
        item=o[f'a{i}']
        rows.append(dict(ID=r['ID'], group=g, position=i, week=1 if i<=10 else 2,
                         day=(i-1)//2+1, item=item, response=str(resp).strip(),
                         correct=int(corr)))
L=pd.DataFrame(rows).merge(lab, left_on='item', right_on='content_id')
L['rated_true']=(L.response=='A').astype(int)
# sanity: correct should equal (rated_true==label)
L['correct_chk']=(L.rated_true==L.label).astype(int)
print("mismatch between recorded correctness and recomputed:", (L.correct!=L.correct_chk).sum())
L=L.merge(demo[['ID','gender','edu_years','political_pref','pol_pref_2','internet_use\n' if 'internet_use\n' in demo.columns else 'internet_use','religious','crt1','crt2','crt3','birth']].rename(columns=lambda c:c.strip()), on='ID', how='left')
L['crt']=(L.crt1==500).astype(int)+(L.crt2==2).astype(int)+(L.crt3.astype(str).str.lower()=='emily').astype(int)
L['age']=(pd.Timestamp('2025-08-15')-pd.to_datetime(L.birth,errors='coerce')).dt.days/365.25
L.to_csv('long.csv', index=False)
print("\nN ratings:", len(L), "| N participants:", L.ID.nunique(), "| N items:", L.item.nunique())
print("ratings per participant: mean %.1f min %d max %d" % (L.groupby('ID').size().mean(), L.groupby('ID').size().min(), L.groupby('ID').size().max()))
print("ratings per item: mean %.1f min %d max %d" % (L.groupby('item').size().mean(), L.groupby('item').size().min(), L.groupby('item').size().max()))
print("\noverall acc %.3f | true acc %.3f | false acc %.3f" % (L.correct.mean(), L[L.label==1].correct.mean(), L[L.label==0].correct.mean()))
print("share rated TRUE overall: %.3f" % L.rated_true.mean())
print("by modality:", L.groupby('modality').correct.mean().round(3).to_dict())
print("by topic:", L.groupby('topic').correct.mean().round(3).to_dict())
print("by week:", L.groupby('week').correct.mean().round(3).to_dict())
print("by block:", L.groupby(pd.cut(L.position,[0,5,10,15,20])).correct.mean().round(3).to_dict())
P=L.groupby('ID').agg(acc=('correct','mean'),n=('correct','size'),pol=('pol_pref_2','first'),gender=('gender','first'),crt=('crt','first'),age=('age','first'),edu=('edu_years','first')).reset_index()
print("\npolitical (pol_pref_2 1=L,2=C,3=R):", P.pol.value_counts(dropna=False).to_dict())
print("political 1-10 scale summary:", L.groupby('ID').political_pref.first().describe().round(2).to_dict())
print("gender:", P.gender.value_counts(dropna=False).to_dict(), "| age mean %.1f sd %.1f" % (P.age.mean(), P.age.std()))
print("edu years mean %.1f sd %.1f" % (P.edu.mean(), P.edu.std()), "| CRT dist:", P.crt.value_counts().sort_index().to_dict())
print("\nindividual accuracy: mean %.3f sd %.3f min %.2f max %.2f" % (P.acc.mean(), P.acc.std(), P.acc.min(), P.acc.max()))
print("participants >0.7:", (P.acc>0.7).sum(), " 0.5-0.7:", ((P.acc>=0.5)&(P.acc<=0.7)).sum(), " <0.5:", (P.acc<0.5).sum())
# item-level: majority verdict accuracy with all raters
I=L.groupby('item').agg(n=('correct','size'), frac_true=('rated_true','mean'), label=('label','first'), acc=('correct','mean')).reset_index()
I['maj_correct']=((I.frac_true>0.5).astype(int)==I.label).astype(int)
print("\nfull-crowd majority accuracy: %.3f (ties: %d)" % (I.maj_correct.mean(), (I.frac_true==0.5).sum()))
print("items where majority wrong:", I[I.maj_correct==0][['item','label','frac_true','n']].to_string(index=False))
