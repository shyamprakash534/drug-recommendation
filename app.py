from flask import Flask, render_template, request, session, send_file
import numpy as np
import os, io, base64, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "drug_ai_2026"

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "dataset", "real_drug_dataset.csv")
MDL_DIR  = os.path.join(BASE, "models")

# ── Auto-train on first run ───────────────────────────────────
def train_models():
    import pandas as pd, joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    os.makedirs(MDL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df = df.drop(columns=[c for c in
         ["Patient_ID","Side_Effects","Improvement_Score","Treatment_Duration_days"]
         if c in df.columns])

    le_g=LabelEncoder(); le_c=LabelEncoder()
    le_d=LabelEncoder(); le_do=LabelEncoder()

    df["ge"]  = le_g.fit_transform(df["Gender"])
    df["ce"]  = le_c.fit_transform(df["Condition"])
    df["de"]  = le_d.fit_transform(df["Drug_Name"])
    df["doe"] = le_do.fit_transform(df["Dosage_mg"])

    X = df[["Age","ge","ce"]]
    Xtr,_,ydr,_ = train_test_split(X, df["de"],  test_size=0.2, random_state=42, stratify=df["de"])
    _,  _,ydo,_ = train_test_split(X, df["doe"], test_size=0.2, random_state=42, stratify=df["doe"])

    dm = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42,class_weight="balanced")
    dm.fit(Xtr,ydr)
    dosm = GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,max_depth=4,random_state=42)
    dosm.fit(Xtr,ydo)

    joblib.dump(dm,   os.path.join(MDL_DIR,"drug_model.pkl"))
    joblib.dump(dosm, os.path.join(MDL_DIR,"dosage_model.pkl"))
    joblib.dump(le_g, os.path.join(MDL_DIR,"le_gender.pkl"))
    joblib.dump(le_c, os.path.join(MDL_DIR,"le_condition.pkl"))
    joblib.dump(le_d, os.path.join(MDL_DIR,"le_drug.pkl"))
    joblib.dump(le_do,os.path.join(MDL_DIR,"le_dosage.pkl"))
    print("✅ Models trained!")

if not os.path.exists(os.path.join(MDL_DIR,"drug_model.pkl")):
    train_models()

import joblib
drug_model   = joblib.load(os.path.join(MDL_DIR,"drug_model.pkl"))
dosage_model = joblib.load(os.path.join(MDL_DIR,"dosage_model.pkl"))
le_gender    = joblib.load(os.path.join(MDL_DIR,"le_gender.pkl"))
le_condition = joblib.load(os.path.join(MDL_DIR,"le_condition.pkl"))
le_drug      = joblib.load(os.path.join(MDL_DIR,"le_drug.pkl"))
le_dosage    = joblib.load(os.path.join(MDL_DIR,"le_dosage.pkl"))

DRUG_INFO = {
    "Metformin":        {"class":"Biguanide",              "use":"Lowers blood glucose in Type 2 Diabetes",        "note":"Take with food to reduce stomach upset"},
    "Glipizide":        {"class":"Sulfonylurea",           "use":"Stimulates insulin release for Type 2 Diabetes", "note":"Monitor blood sugar regularly"},
    "Insulin Glargine": {"class":"Long-acting Insulin",    "use":"Controls blood sugar in Diabetes Type 1 & 2",    "note":"Inject at same time daily"},
    "Amlodipine":       {"class":"Calcium Channel Blocker","use":"Treats high blood pressure and chest pain",      "note":"May cause ankle swelling"},
    "Metoprolol":       {"class":"Beta Blocker",           "use":"Reduces heart rate and blood pressure",          "note":"Do not stop suddenly"},
    "Losartan":         {"class":"ARB",                    "use":"Relaxes blood vessels to lower BP",              "note":"Avoid potassium supplements"},
    "Sertraline":       {"class":"SSRI",                   "use":"Treats depression and anxiety disorders",        "note":"Takes 2-4 weeks to show effect"},
    "Escitalopram":     {"class":"SSRI",                   "use":"Treats major depressive disorder",               "note":"Avoid alcohol during treatment"},
    "Bupropion":        {"class":"NDRI",                   "use":"Treats depression and aids smoking cessation",   "note":"May cause dry mouth"},
    "Amoxicillin":      {"class":"Penicillin Antibiotic",  "use":"Treats bacterial infections",                    "note":"Complete the full course"},
    "Ciprofloxacin":    {"class":"Fluoroquinolone",        "use":"Treats urinary and respiratory infections",      "note":"Avoid dairy products when taking"},
    "Azithromycin":     {"class":"Macrolide Antibiotic",   "use":"Treats respiratory and skin infections",         "note":"Usually a 3-5 day course"},
    "Paracetamol":      {"class":"Analgesic",              "use":"Relieves mild to moderate pain and fever",       "note":"Do not exceed 4g per day"},
    "Ibuprofen":        {"class":"NSAID",                  "use":"Anti-inflammatory pain relief",                  "note":"Take with food to protect stomach"},
    "Tramadol":         {"class":"Opioid Analgesic",       "use":"Treats moderate to severe pain",                 "note":"May cause drowsiness — avoid driving"},
}

STATS = {
    "total_patients":941,"conditions":5,"unique_drugs":15,
    "drug_accuracy":94.2,"dosage_accuracy":23.5,
    "model_drug":"Random Forest (200 trees)",
    "model_dosage":"Gradient Boosting (150 estimators)",
}

def make_chart(alternatives):
    drugs  = [a["drug"] for a in alternatives]
    confs  = [a["confidence"] for a in alternatives]
    fig,ax = plt.subplots(figsize=(6,3))
    bars   = ax.barh(drugs,confs,color=["#667eea","#764ba2","#2ecc71"][:len(drugs)],edgecolor="white")
    ax.set_xlabel("Confidence %")
    ax.set_title("Drug Recommendation Confidence")
    ax.set_xlim(0,115)
    ax.spines[["top","right"]].set_visible(False)
    for bar,val in zip(bars,confs):
        ax.text(val+1,bar.get_y()+bar.get_height()/2,f"{val}%",va="center",fontsize=11,fontweight="bold")
    plt.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=120); plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age       = int(request.form["age"])
    gender    = request.form["gender"]
    condition = request.form["condition"]
    bp        = request.form.get("blood_pressure","Normal")
    severity  = request.form.get("severity","Moderate")

    g_enc = le_gender.transform([gender])[0]
    c_enc = le_condition.transform([condition])[0]
    feat  = np.array([[age,g_enc,c_enc]])

    drug_pred   = drug_model.predict(feat)[0]
    dosage_pred = dosage_model.predict(feat)[0]
    proba       = drug_model.predict_proba(feat)[0]

    drug_name  = le_drug.inverse_transform([drug_pred])[0]
    dosage_val = int(le_dosage.inverse_transform([dosage_pred])[0])
    confidence = round(proba[drug_pred]*100,1)

    top3 = np.argsort(proba)[::-1][:3]
    alternatives = [{"drug":le_drug.inverse_transform([i])[0],
                     "confidence":round(proba[i]*100,1)} for i in top3]

    drug_info = DRUG_INFO.get(drug_name,{"class":"N/A","use":"N/A","note":"Consult your doctor"})
    chart_b64 = make_chart(alternatives)
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    if "history" not in session: session["history"]=[]
    record={"time":timestamp,"age":age,"gender":gender,"condition":condition,
            "drug":drug_name,"dosage":dosage_val,"confidence":confidence}
    history=session["history"]; history.insert(0,record); session["history"]=history[:10]
    session["last_result"]={**record,"bp":bp,"severity":severity,
        "drug_class":drug_info["class"],"drug_use":drug_info["use"],"drug_note":drug_info["note"]}

    return render_template("result.html",drug=drug_name,dosage=dosage_val,
        condition=condition,age=age,gender=gender,bp=bp,severity=severity,
        confidence=confidence,alternatives=alternatives,drug_info=drug_info,
        chart_b64=chart_b64,timestamp=timestamp,history=session["history"])

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",stats=STATS)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    return render_template("history.html",history=session.get("history",[]))

@app.route("/download_pdf")
def download_pdf():
    r=session.get("last_result",{})
    if not r: return "No prediction found.",400
    pdf=FPDF(); pdf.add_page()
    pdf.set_fill_color(102,126,234); pdf.rect(0,0,210,40,"F")
    pdf.set_font("Helvetica","B",20); pdf.set_text_color(255,255,255)
    pdf.set_xy(0,8)
    pdf.cell(0,12,"  AI Drug Recommendation Report",new_x="LMARGIN",new_y="NEXT")
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,8,f"  Generated: {r.get('time','')}",new_x="LMARGIN",new_y="NEXT")
    pdf.set_text_color(0,0,0); pdf.ln(8)
    pdf.set_font("Helvetica","B",13); pdf.set_fill_color(240,244,255)
    pdf.cell(0,10," Patient Details",new_x="LMARGIN",new_y="NEXT",fill=True)
    pdf.set_font("Helvetica","",11); pdf.ln(2)
    for lbl,val in [("Age",str(r.get("age",""))),("Gender",str(r.get("gender",""))),
                    ("Condition",str(r.get("condition",""))),("BP",str(r.get("bp",""))),
                    ("Severity",str(r.get("severity","")))]:
        pdf.cell(60,8,f"  {lbl}:"); pdf.cell(130,8,val,new_x="LMARGIN",new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica","B",13); pdf.set_fill_color(240,255,244)
    pdf.cell(0,10," AI Recommendation",new_x="LMARGIN",new_y="NEXT",fill=True)
    pdf.set_font("Helvetica","",11); pdf.ln(2)
    for lbl,val in [("Drug",str(r.get("drug",""))),("Dosage",f"{r.get('dosage','')} mg"),
                    ("Confidence",f"{r.get('confidence','')}%"),("Class",str(r.get("drug_class",""))),
                    ("Use",str(r.get("drug_use",""))),("Note",str(r.get("drug_note","")))]:
        pdf.set_font("Helvetica","B",11); pdf.cell(60,8,f"  {lbl}:")
        pdf.set_font("Helvetica","",11); pdf.multi_cell(130,8,val); pdf.ln(1)
    pdf.ln(4); pdf.set_fill_color(255,248,225); pdf.set_font("Helvetica","I",9)
    pdf.multi_cell(190,6," DISCLAIMER: Academic project only. NOT a real prescription. Consult a licensed doctor.",fill=True)
    pdf.set_y(-18); pdf.set_font("Helvetica","I",8); pdf.set_text_color(150,150,150)
    pdf.cell(0,10,"AI Drug Recommendation System | Final Year Project",align="C")
    buf=io.BytesIO(); pdf.output(buf); buf.seek(0)
    return send_file(buf,as_attachment=True,download_name="Drug_Report.pdf",mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=5000)
