"""Train an illustrative Random Forest on transparent, synthetic labelled profiles.
Replace synthetic training data with consented, representative real labelled data before claiming real-world validity.
"""
import csv, random, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from model.careers import PROFILES, QUESTIONS
ROOT=Path(__file__).resolve().parents[1]
def train():
    rng=random.Random(42); rows=[]
    for career,profile in PROFILES.items():
        for _ in range(160):
            features=[max(1,min(5,v+rng.choice([-2,-1,0,0,0,1,2]))) for v in profile]
            rows.append((features,career))
    rng.shuffle(rows)
    with (ROOT/'data'/'synthetic_career_profiles.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow([k for k,_ in QUESTIONS]+['career']);w.writerows([x+[y] for x,y in rows])
    X=np.array([x for x,y in rows]); y=np.array([y for x,y in rows]);
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=250,min_samples_leaf=2,random_state=42,class_weight='balanced');clf.fit(Xtr,ytr)
    pred=clf.predict(Xte)
    report=f'SYNTHETIC-DATA EVALUATION ONLY — not evidence of career suitability.\nAccuracy: {accuracy_score(yte,pred):.3f}\n\n'+classification_report(yte,pred,zero_division=0)
    (ROOT/'model'/'evaluation.txt').write_text(report,encoding='utf-8')
    with (ROOT/'model'/'career_model.pkl').open('wb') as f:pickle.dump(clf,f)
    print(report)
if __name__=='__main__':train()
