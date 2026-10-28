from pathlib import Path
import pickle, sqlite3, json
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify
from model.careers import CAREERS, QUESTIONS
ROOT=Path(__file__).resolve().parent
app=Flask(__name__)
DB=ROOT/'data'/'results.db'
def init_db():
    with sqlite3.connect(DB) as c:c.execute('CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, created_at TEXT, answers TEXT, matches TEXT)')
def model():
    path=ROOT/'model'/'career_model.pkl'
    if not path.exists():
        from model.train import train
        train()
    with path.open('rb') as f:return pickle.load(f)
clf=model();init_db()
@app.get('/')
def home():return render_template('index.html',questions=QUESTIONS)
@app.post('/api/predict')
def predict():
    body=request.get_json(silent=True) or {}; answers=body.get('answers')
    if not isinstance(answers,list) or len(answers)!=len(QUESTIONS) or any(type(x)!=int or not 1<=x<=5 for x in answers):
        return jsonify(error='Please answer all 10 questions with values from 1 to 5.'),400
    probabilities=clf.predict_proba([answers])[0]
    matches=[]
    for idx in probabilities.argsort()[::-1][:3]:
        name=clf.classes_[idx]
        matches.append({'name':name,'score':round(float(probabilities[idx])*100,1),**CAREERS[name]})
    with sqlite3.connect(DB) as c:c.execute('INSERT INTO results (created_at,answers,matches) VALUES (?,?,?)',(datetime.now(timezone.utc).isoformat(),json.dumps(answers),json.dumps(matches)))
    return jsonify(matches=matches,disclaimer='Scores are model probabilities from synthetic example data, not validated career-fit percentages. Use these suggestions for exploration, not a definitive decision.')
if __name__=='__main__':app.run(debug=True)
