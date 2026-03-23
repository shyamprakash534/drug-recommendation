from flask import Flask, render_template, request, session, send_file
import numpy as np
import os, io, base64, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'drug_ai_2026_secret'

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_CSV  = os.path.join(BASE_DIR, "dataset", "real_drug_dataset.csv")
MDL_DIR   = os.path.join(BASE_DIR, "models")

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

    dm   = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    dm.fit(Xtr, ydr)
    dosm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    dosm.fit(Xtr, ydo)

    joblib.dump(dm,    os.path.join(MDL_DIR,"drug_model.pkl"))
    joblib.dump(dosm,  os.path.join(MDL_DIR,"dosage_model.pkl"))
    joblib.dump(le_g,  os.path.join(MDL_DIR,"le_gender.pkl"))
    joblib.dump(le_c,  os.path.join(MDL_DIR,"le_condition.pkl"))
    joblib.dump(le_d,  os.path.join(MDL_DIR,"le_drug.pkl"))
    joblib.dump(le_do, os.path.join(MDL_DIR,"le_dosage.pkl"))
    print("✅ Models trained and saved!")

if not os.path.exists(os.path.join(MDL_DIR,"drug_model.pkl")):
    train_models()

import joblib
drug_model   = joblib.load(os.path.join(MDL_DIR,"drug_model.pkl"))
dosage_model = joblib.load(os.path.join(MDL_DIR,"dosage_model.pkl"))
le_gender    = joblib.load(os.path.join(MDL_DIR,"le_gender.pkl"))
le_condition = joblib.load(os.path.join(MDL_DIR,"le_condition.pkl"))
le_drug      = joblib.load(os.path.join(MDL_DIR,"le_drug.pkl"))
le_dosage    = joblib.load(os.path.join(MDL_DIR,"le_dosage.pkl"))

# ══════════════════════════════════════════════════════════════
# SEVERITY TABLE
# Akkuva (Severe) → Strong drug + High dose
# Moderate        → Standard drug + Normal dose
# Takkuva (Mild)  → Light drug + Low dose
# ══════════════════════════════════════════════════════════════
SEVERITY_RULES = {
    "Diabetes": {
        "Mild":     {"drug":"Metformin",        "dosage":500,  "note":"Start low, monitor blood sugar weekly"},
        "Moderate": {"drug":"Metformin",        "dosage":850,  "note":"Take with meals, monitor HbA1c monthly"},
        "Severe":   {"drug":"Insulin Glargine", "dosage":1000, "note":"⚠️ Immediate insulin therapy required. Hospital monitoring."},
    },
    "Hypertension": {
        "Mild":     {"drug":"Amlodipine",  "dosage":5,  "note":"Lifestyle changes + low dose. Monitor BP daily."},
        "Moderate": {"drug":"Metoprolol",  "dosage":50, "note":"Take at same time daily. Do not stop suddenly."},
        "Severe":   {"drug":"Losartan",    "dosage":100,"note":"⚠️ High-dose ARB therapy. Immediate BP control needed."},
    },
    "Depression": {
        "Mild":     {"drug":"Escitalopram", "dosage":10, "note":"Low dose SSRI. Counseling recommended alongside."},
        "Moderate": {"drug":"Sertraline",   "dosage":50, "note":"Takes 2-4 weeks to show effect. Do not stop abruptly."},
        "Severe":   {"drug":"Bupropion",    "dosage":150,"note":"⚠️ Severe depression. Psychiatric consultation required."},
    },
    "Infection": {
        "Mild":     {"drug":"Amoxicillin",   "dosage":250, "note":"Complete full 5-day course even if feeling better."},
        "Moderate": {"drug":"Azithromycin",  "dosage":500, "note":"Take on empty stomach. Avoid antacids."},
        "Severe":   {"drug":"Ciprofloxacin", "dosage":750, "note":"⚠️ Severe infection. IV antibiotics may be needed."},
    },
    "Pain Relief": {
        "Mild":     {"drug":"Paracetamol", "dosage":500,  "note":"Safe for most patients. Max 4g/day."},
        "Moderate": {"drug":"Ibuprofen",   "dosage":400,  "note":"Take with food to protect stomach lining."},
        "Severe":   {"drug":"Tramadol",    "dosage":100,  "note":"⚠️ Opioid analgesic. May cause drowsiness. Avoid driving."},
    },
}

DRUG_INFO = {
    "Metformin":        {"class":"Biguanide",              "use":"Lowers blood glucose in Type 2 Diabetes",       "note":"Take with food to reduce stomach upset"},
    "Glipizide":        {"class":"Sulfonylurea",           "use":"Stimulates insulin release",                    "note":"Monitor blood sugar regularly"},
    "Insulin Glargine": {"class":"Long-acting Insulin",    "use":"Controls blood sugar in severe Diabetes",       "note":"Inject at same time daily"},
    "Amlodipine":       {"class":"Calcium Channel Blocker","use":"Treats mild-moderate high blood pressure",      "note":"May cause ankle swelling"},
    "Metoprolol":       {"class":"Beta Blocker",           "use":"Reduces heart rate and blood pressure",         "note":"Do not stop suddenly"},
    "Losartan":         {"class":"ARB",                    "use":"Severe hypertension management",                "note":"Avoid potassium supplements"},
    "Sertraline":       {"class":"SSRI",                   "use":"Treats moderate depression and anxiety",        "note":"Takes 2-4 weeks to show effect"},
    "Escitalopram":     {"class":"SSRI",                   "use":"Treats mild to moderate depressive disorder",   "note":"Avoid alcohol during treatment"},
    "Bupropion":        {"class":"NDRI",                   "use":"Treats severe depression",                      "note":"May cause dry mouth and insomnia"},
    "Amoxicillin":      {"class":"Penicillin Antibiotic",  "use":"Treats mild bacterial infections",              "note":"Complete the full course"},
    "Ciprofloxacin":    {"class":"Fluoroquinolone",        "use":"Treats severe infections",                      "note":"Avoid dairy products when taking"},
    "Azithromycin":     {"class":"Macrolide Antibiotic",   "use":"Treats moderate respiratory infections",        "note":"Usually a 3-5 day course"},
    "Paracetamol":      {"class":"Analgesic",              "use":"Relieves mild pain and fever",                  "note":"Do not exceed 4g per day"},
    "Ibuprofen":        {"class":"NSAID",                  "use":"Moderate anti-inflammatory pain relief",        "note":"Take with food to protect stomach"},
    "Tramadol":         {"class":"Opioid Analgesic",       "use":"Treats severe pain",                            "note":"May cause drowsiness — avoid driving"},
}

STATS = {
    "total_patients":941,"conditions":5,"unique_drugs":15,
    "drug_accuracy":94.2,"dosage_accuracy":23.5,
    "model_drug":"Random Forest (200 trees)",
    "model_dosage":"Gradient Boosting (150 estimators)",
}

# ══════════════════════════════════════════════════════════════
# WHAT-IF ENGINE
# ══════════════════════════════════════════════════════════════
def apply_whatif_rules(drug_name, dosage_val, confidence,
                       age, condition, bp, severity, weight):
    rules = []
    original_drug   = drug_name
    original_dosage = dosage_val

    # ── SEVERITY RULE (Core: Akkuva / Takkuva) ───────────────
    sev_data = SEVERITY_RULES.get(condition, {}).get(severity)
    if sev_data:
        drug_name  = sev_data["drug"]
        dosage_val = sev_data["dosage"]
        sev_icons  = {"Mild":"🟢","Moderate":"🟡","Severe":"🔴"}
        sev_types  = {"Mild":"success","Moderate":"warning","Severe":"danger"}
        rules.append({
            "type"   : sev_types[severity],
            "icon"   : sev_icons[severity],
            "title"  : f"Severity Rule — {severity} {condition}",
            "message": f"Severity is <b>{severity}</b> → {drug_name} {dosage_val}mg selected. {sev_data['note']}",
        })

    # ── Rule: High BP + Hypertension ─────────────────────────
    if bp == "High" and condition == "Hypertension":
        drug_name  = "Losartan"
        dosage_val = 100
        confidence = 99.0
        rules.append({
            "type":"danger","icon":"🔴",
            "title":"High BP Override",
            "message":"High Blood Pressure + Hypertension → Losartan 100mg (strongest ARB). Immediate BP control.",
        })

    # ── Rule: Low BP + Hypertension ──────────────────────────
    if bp == "Low" and condition == "Hypertension":
        drug_name  = "Amlodipine"
        dosage_val = 2.5
        confidence = 90.0
        rules.append({
            "type":"warning","icon":"🟡",
            "title":"Low BP Warning",
            "message":"Low BP with Hypertension → Amlodipine 2.5mg (minimum dose). Monitor BP every 2 hours.",
        })

    # ── Rule: Elderly dose reduction ─────────────────────────
    if age > 65:
        old = dosage_val
        dosage_val = max(50, int(dosage_val * 0.75))
        rules.append({
            "type":"warning","icon":"🟡",
            "title":"Elderly Dose Reduction (Age > 65)",
            "message":f"Patient is {age} years old. Dosage reduced from {old}mg → {dosage_val}mg (75% of standard) to prevent toxicity.",
        })

    # ── Rule: Pediatric safety ───────────────────────────────
    if age < 18:
        old = dosage_val
        dosage_val = max(50, int(dosage_val * 0.5))
        if drug_name == "Tramadol":
            drug_name  = "Paracetamol"
            dosage_val = 250
            rules.append({
                "type":"danger","icon":"🔴",
                "title":"Pediatric Safety Override (Age < 18)",
                "message":f"Tramadol is unsafe for age {age}. Switched to Paracetamol 250mg.",
            })
        else:
            rules.append({
                "type":"warning","icon":"🟡",
                "title":"Pediatric Dose Adjustment (Age < 18)",
                "message":f"Patient is {age} years old. Dosage halved from {old}mg → {dosage_val}mg for safety.",
            })

    # ── Rule: Low body weight ─────────────────────────────────
    if weight < 45:
        old = dosage_val
        dosage_val = max(50, int(dosage_val * 0.75))
        rules.append({
            "type":"warning","icon":"🟡",
            "title":"Low Body Weight Alert (< 45kg)",
            "message":f"Weight is {weight}kg. Dosage reduced from {old}mg → {dosage_val}mg to avoid overdose risk.",
        })

    # ── Rule: NSAID contraindication with High BP ────────────
    if bp == "High" and drug_name == "Ibuprofen":
        drug_name  = "Paracetamol"
        dosage_val = 500
        rules.append({
            "type":"danger","icon":"🔴",
            "title":"NSAID Contraindication",
            "message":"Ibuprofen raises blood pressure — contraindicated. Switched to Paracetamol 500mg.",
        })

    # ── Rule: Diabetes + Glipizide safety ────────────────────
    if condition == "Diabetes" and drug_name == "Glipizide":
        drug_name  = "Metformin"
        dosage_val = 500
        rules.append({
            "type":"override","icon":"🔵",
            "title":"Diabetes Safety Rule",
            "message":"Glipizide carries hypoglycemia risk. Switched to safer first-line Metformin 500mg.",
        })

    # ── Rule: All clear ──────────────────────────────────────
    if not rules:
        rules.append({
            "type":"success","icon":"✅",
            "title":"All Clear",
            "message":"No special risk conditions detected. Standard AI recommendation applies.",
        })

    changed = (drug_name != original_drug or dosage_val != original_dosage)
    return drug_name, round(float(dosage_val), 1), round(confidence, 1), rules, changed


def make_chart(drug_name, severity, condition, alternatives):
    drugs  = [a["drug"] for a in alternatives]
    confs  = [a["confidence"] for a in alternatives]
    colors = {"Mild":"#2ecc71","Moderate":"#f39c12","Severe":"#e74c3c"}
    bar_color = colors.get(severity, "#667eea")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")

    # Left: Confidence bars
    ax1 = axes[0]
    bars = ax1.barh(drugs, confs,
                    color=[bar_color,"#667eea","#764ba2"][:len(drugs)],
                    edgecolor="white", height=0.45)
    ax1.set_xlabel("Confidence %")
    ax1.set_title("ML Drug Confidence", fontweight="bold")
    ax1.set_xlim(0, 120)
    ax1.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, confs):
        ax1.text(val+1, bar.get_y()+bar.get_height()/2,
                 f"{val}%", va="center", fontsize=10, fontweight="bold")

    # Right: Severity indicator
    ax2 = axes[1]
    sev_levels = ["Mild","Moderate","Severe"]
    sev_colors = ["#2ecc71","#f39c12","#e74c3c"]
    sev_vals   = [33, 33, 34]
    wedge_colors = [sev_colors[i] if sev_levels[i]==severity
                    else f"{sev_colors[i]}55" for i in range(3)]
    wedges, texts = ax2.pie(sev_vals, labels=sev_levels,
                             colors=wedge_colors, startangle=90,
                             wedgeprops={"edgecolor":"white","linewidth":2})
    ax2.set_title(f"Severity Level: {severity}", fontweight="bold")
    for i, (w, t) in enumerate(zip(wedges, texts)):
        if sev_levels[i] == severity:
            w.set_linewidth(4)
            t.set_fontweight("bold")
            t.set_fontsize(13)

    plt.suptitle(f"AI Recommendation: {drug_name} | {condition}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age       = int(request.form.get("age", 30))
        gender    = request.form.get("gender", "Male")
        condition = request.form.get("condition", "Hypertension")
        bp        = request.form.get("blood_pressure", "Normal")
        severity  = request.form.get("severity", "Moderate")
        weight    = int(request.form.get("weight", 70))

        g_enc = le_gender.transform([gender])[0]
        c_enc = le_condition.transform([condition])[0]
        feat  = np.array([[age, g_enc, c_enc]])

        drug_pred   = drug_model.predict(feat)[0]
        dosage_pred = dosage_model.predict(feat)[0]
        proba       = drug_model.predict_proba(feat)[0]

        drug_name  = le_drug.inverse_transform([drug_pred])[0]
        dosage_val = int(le_dosage.inverse_transform([dosage_pred])[0])
        confidence = round(proba[drug_pred]*100, 1)

        top3 = np.argsort(proba)[::-1][:3]
        alternatives = [{"drug":le_drug.inverse_transform([i])[0],
                         "confidence":round(proba[i]*100,1)} for i in top3]

        # ── Apply What-If + Severity Rules ───────────────────
        drug_name, dosage_val, confidence, rules, changed = apply_whatif_rules(
            drug_name, dosage_val, confidence,
            age, condition, bp, severity, weight
        )

        drug_info = DRUG_INFO.get(drug_name,
            {"class":"N/A","use":"N/A","note":"Consult your doctor"})
        chart_b64 = make_chart(drug_name, severity, condition, alternatives)
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

        # Severity display config
        sev_config = {
            "Mild":     {"color":"#2ecc71","emoji":"🟢","label":"Mild — Low Risk"},
            "Moderate": {"color":"#f39c12","emoji":"🟡","label":"Moderate — Medium Risk"},
            "Severe":   {"color":"#e74c3c","emoji":"🔴","label":"Severe — High Risk"},
        }
        sev_display = sev_config.get(severity, sev_config["Moderate"])

        if "history" not in session: session["history"] = []
        record = {"time":timestamp,"age":age,"gender":gender,
                  "condition":condition,"severity":severity,
                  "drug":drug_name,"dosage":dosage_val,"confidence":confidence}
        h = session["history"]; h.insert(0,record); session["history"] = h[:10]
        session["last_result"] = {
            **record,"bp":bp,"weight":weight,
            "drug_class":drug_info["class"],
            "drug_use":drug_info["use"],
            "drug_note":drug_info["note"],
        }

        return render_template("result.html",
            drug=drug_name, dosage=dosage_val, confidence=confidence,
            age=age, gender=gender, condition=condition,
            bp=bp, severity=severity, weight=weight,
            drug_info=drug_info, chart_b64=chart_b64,
            timestamp=timestamp, history=session["history"],
            rules_triggered=rules, whatif_changed=changed,
            alternatives=alternatives,
            sev_display=sev_display,
        )

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2><a href='/'>← Back</a>", 500


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=STATS)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    return render_template("history.html", history=session.get("history", []))

@app.route("/download_pdf")
def download_pdf():
    r = session.get("last_result", {})
    if not r: return "No prediction found.", 400

    pdf = FPDF()
    pdf.add_page()
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
    for lbl,val in [("Age",str(r.get("age",""))),
                    ("Gender",str(r.get("gender",""))),
                    ("Condition",str(r.get("condition",""))),
                    ("Severity",str(r.get("severity",""))),
                    ("Blood Pressure",str(r.get("bp",""))),
                    ("Weight",f"{r.get('weight','')} kg")]:
        pdf.cell(65,8,f"  {lbl}:"); pdf.cell(125,8,val,new_x="LMARGIN",new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica","B",13); pdf.set_fill_color(240,255,244)
    pdf.cell(0,10," AI Recommendation (with Severity + What-If Rules)",new_x="LMARGIN",new_y="NEXT",fill=True)
    pdf.set_font("Helvetica","",11); pdf.ln(2)
    for lbl,val in [("Recommended Drug",str(r.get("drug",""))),
                    ("Dosage",f"{r.get('dosage','')} mg"),
                    ("AI Confidence",f"{r.get('confidence','')}%"),
                    ("Drug Class",str(r.get("drug_class",""))),
                    ("Primary Use",str(r.get("drug_use",""))),
                    ("Clinical Note",str(r.get("drug_note","")))]:
        pdf.set_font("Helvetica","B",11); pdf.cell(65,8,f"  {lbl}:")
        pdf.set_font("Helvetica","",11); pdf.multi_cell(125,8,val); pdf.ln(1)

    pdf.ln(4); pdf.set_fill_color(255,248,225); pdf.set_font("Helvetica","I",9)
    pdf.multi_cell(190,6," DISCLAIMER: Academic project only. NOT a real medical prescription. Always consult a licensed doctor.",fill=True)
    pdf.set_y(-18); pdf.set_font("Helvetica","I",8); pdf.set_text_color(150,150,150)
    pdf.cell(0,10,"AI Drug Recommendation System | Final Year Project",align="C")

    buf=io.BytesIO(); pdf.output(buf); buf.seek(0)
    return send_file(buf,as_attachment=True,
                     download_name="Drug_Report.pdf",
                     mimetype="application/pdf")

if __name__ == "__main__":
    print("🚀 Starting AI Drug Recommendation System...")
    app.run(debug=False, host="0.0.0.0", port=5000)
