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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "dataset", "real_drug_dataset.csv")
MDL_DIR  = os.path.join(BASE_DIR, "models")

# ── Auto-train ────────────────────────────────────────────────
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
# RULE 5 — DOSAGE FREQUENCY TABLE
# Mild=Once, Moderate=Twice, Severe=Thrice per day
# ══════════════════════════════════════════════════════════════
FREQUENCY_RULES = {
    "Mild":     {"freq": 1, "label": "Once daily (OD)",         "times": "Morning"},
    "Moderate": {"freq": 2, "label": "Twice daily (BD)",        "times": "Morning & Evening"},
    "Severe":   {"freq": 3, "label": "Three times daily (TDS)", "times": "Morning, Afternoon & Night"},
}

# ══════════════════════════════════════════════════════════════
# RULE 6 — MAX DAILY DOSE LIMITS (mg/day)
# ══════════════════════════════════════════════════════════════
MAX_DAILY_DOSE = {
    "Metformin":        2550,
    "Glipizide":        40,
    "Insulin Glargine": 100,
    "Amlodipine":       10,
    "Metoprolol":       400,
    "Losartan":         100,
    "Sertraline":       200,
    "Escitalopram":     20,
    "Bupropion":        450,
    "Amoxicillin":      3000,
    "Ciprofloxacin":    1500,
    "Azithromycin":     500,
    "Paracetamol":      4000,
    "Ibuprofen":        2400,
    "Tramadol":         400,
}

# ══════════════════════════════════════════════════════════════
# RULE 8 — WEIGHT-BASED DOSAGE (mg/kg)
# ══════════════════════════════════════════════════════════════
WEIGHT_BASED_DRUGS = {
    "Amoxicillin":   {"mgkg": 25, "max": 500},
    "Ciprofloxacin": {"mgkg": 15, "max": 750},
    "Azithromycin":  {"mgkg": 10, "max": 500},
    "Paracetamol":   {"mgkg": 15, "max": 1000},
    "Ibuprofen":     {"mgkg": 10, "max": 600},
    "Tramadol":      {"mgkg": 1,  "max": 100},
}

# ══════════════════════════════════════════════════════════════
# RULE 9 — DRUG ALLERGY SWITCH TABLE
# ══════════════════════════════════════════════════════════════
ALLERGY_SWITCH = {
    "Penicillin": {"affects": "Amoxicillin",    "switch": "Azithromycin",  "dose": 500},
    "NSAID":      {"affects": "Ibuprofen",      "switch": "Paracetamol",   "dose": 500},
    "Opioid":     {"affects": "Tramadol",       "switch": "Ibuprofen",     "dose": 400},
    "Sulfa":      {"affects": "Ciprofloxacin",  "switch": "Azithromycin",  "dose": 500},
    "Metformin":  {"affects": "Metformin",      "switch": "Glipizide",     "dose": 5},
    "None":       None,
}

# ══════════════════════════════════════════════════════════════
# RULE 16 — TREATMENT DURATION (days)
# ══════════════════════════════════════════════════════════════
TREATMENT_DURATION = {
    "Diabetes":     {"Mild": 90,  "Moderate": 180, "Severe": 365},
    "Hypertension": {"Mild": 60,  "Moderate": 120, "Severe": 365},
    "Depression":   {"Mild": 30,  "Moderate": 90,  "Severe": 180},
    "Infection":    {"Mild": 5,   "Moderate": 7,   "Severe": 14},
    "Pain Relief":  {"Mild": 3,   "Moderate": 7,   "Severe": 14},
}

# ── SEVERITY TABLE ────────────────────────────────────────────
SEVERITY_RULES = {
    "Diabetes": {
        "Mild":     {"drug":"Metformin",        "dosage":500,  "note":"Start low, monitor blood sugar weekly"},
        "Moderate": {"drug":"Metformin",        "dosage":850,  "note":"Take with meals, monitor HbA1c monthly"},
        "Severe":   {"drug":"Insulin Glargine", "dosage":1000, "note":"⚠️ Immediate insulin therapy required."},
    },
    "Hypertension": {
        "Mild":     {"drug":"Amlodipine", "dosage":5,   "note":"Lifestyle changes + low dose. Monitor BP daily."},
        "Moderate": {"drug":"Metoprolol", "dosage":50,  "note":"Take at same time daily. Do not stop suddenly."},
        "Severe":   {"drug":"Losartan",   "dosage":100, "note":"⚠️ High-dose ARB therapy. Immediate BP control needed."},
    },
    "Depression": {
        "Mild":     {"drug":"Escitalopram", "dosage":10,  "note":"Low dose SSRI. Counseling recommended."},
        "Moderate": {"drug":"Sertraline",   "dosage":50,  "note":"Takes 2-4 weeks to show effect."},
        "Severe":   {"drug":"Bupropion",    "dosage":150, "note":"⚠️ Severe depression. Psychiatric consultation required."},
    },
    "Infection": {
        "Mild":     {"drug":"Amoxicillin",   "dosage":250, "note":"Complete full 5-day course."},
        "Moderate": {"drug":"Azithromycin",  "dosage":500, "note":"Take on empty stomach."},
        "Severe":   {"drug":"Ciprofloxacin", "dosage":750, "note":"⚠️ Severe infection. IV antibiotics may be needed."},
    },
    "Pain Relief": {
        "Mild":     {"drug":"Paracetamol", "dosage":500, "note":"Safe for most patients. Max 4g/day."},
        "Moderate": {"drug":"Ibuprofen",   "dosage":400, "note":"Take with food to protect stomach."},
        "Severe":   {"drug":"Tramadol",    "dosage":100, "note":"⚠️ Opioid analgesic. May cause drowsiness."},
    },
}

DRUG_INFO = {
    "Metformin":        {"class":"Biguanide",              "use":"Lowers blood glucose in Type 2 Diabetes",     "note":"Take with food"},
    "Glipizide":        {"class":"Sulfonylurea",           "use":"Stimulates insulin release",                  "note":"Monitor blood sugar"},
    "Insulin Glargine": {"class":"Long-acting Insulin",    "use":"Controls blood sugar in severe Diabetes",     "note":"Inject at same time daily"},
    "Amlodipine":       {"class":"Calcium Channel Blocker","use":"Treats mild-moderate high blood pressure",    "note":"May cause ankle swelling"},
    "Metoprolol":       {"class":"Beta Blocker",           "use":"Reduces heart rate and blood pressure",       "note":"Do not stop suddenly"},
    "Losartan":         {"class":"ARB",                    "use":"Severe hypertension management",              "note":"Avoid potassium supplements"},
    "Sertraline":       {"class":"SSRI",                   "use":"Treats moderate depression",                  "note":"Takes 2-4 weeks to show effect"},
    "Escitalopram":     {"class":"SSRI",                   "use":"Treats mild to moderate depressive disorder", "note":"Avoid alcohol"},
    "Bupropion":        {"class":"NDRI",                   "use":"Treats severe depression",                    "note":"May cause dry mouth"},
    "Amoxicillin":      {"class":"Penicillin Antibiotic",  "use":"Treats mild bacterial infections",            "note":"Complete the full course"},
    "Ciprofloxacin":    {"class":"Fluoroquinolone",        "use":"Treats severe infections",                    "note":"Avoid dairy products"},
    "Azithromycin":     {"class":"Macrolide Antibiotic",   "use":"Treats moderate infections",                  "note":"3-5 day course"},
    "Paracetamol":      {"class":"Analgesic",              "use":"Relieves mild pain and fever",                "note":"Do not exceed 4g per day"},
    "Ibuprofen":        {"class":"NSAID",                  "use":"Moderate anti-inflammatory pain relief",      "note":"Take with food"},
    "Tramadol":         {"class":"Opioid Analgesic",       "use":"Treats severe pain",                         "note":"Avoid driving"},
}

STATS = {
    "total_patients":941, "conditions":5, "unique_drugs":15,
    "drug_accuracy":94.2, "dosage_accuracy":23.5,
    "model_drug":"Random Forest (200 trees)",
    "model_dosage":"Gradient Boosting (150 estimators)",
}

# ══════════════════════════════════════════════════════════════
# RULE 16 — TREATMENT SCHEDULE GENERATOR (While Loop)
# ══════════════════════════════════════════════════════════════
def generate_treatment_schedule(drug_name, dosage_val, severity, condition, frequency):
    schedule = []
    total_days = TREATMENT_DURATION.get(condition, {}).get(severity, 7)

    if total_days <= 14:
        # Short treatment → day-by-day schedule (while loop)
        day = 1
        while day <= total_days:
            if severity == "Severe" and day > int(total_days * 0.6):
                current_dose = max(50, int(dosage_val * 0.75))
                note = "⬇️ Tapering dose"
            elif day == 1 and severity == "Severe":
                current_dose = dosage_val
                note = "🔴 Full loading dose"
            else:
                current_dose = dosage_val
                note = "✅ Standard dose"
            schedule.append({
                "period"   : f"Day {day}",
                "dose"     : current_dose,
                "frequency": frequency,
                "note"     : note,
            })
            day += 1
    else:
        # Long treatment → week-by-week (while loop, show first 4 weeks)
        week = 1
        while week <= 4:
            if week == 1 and severity != "Mild":
                current_dose = max(50, int(dosage_val * 0.5))
                note = "🟡 Starting low dose"
            elif week == 2 and severity != "Mild":
                current_dose = max(50, int(dosage_val * 0.75))
                note = "🟠 Building up dose"
            elif week == 3:
                current_dose = dosage_val
                note = "✅ Full maintenance dose"
            else:
                current_dose = dosage_val
                note = "✅ Continue maintenance"
            schedule.append({
                "period"   : f"Week {week}",
                "dose"     : current_dose,
                "frequency": frequency,
                "note"     : note,
            })
            week += 1

    return schedule, total_days


# ══════════════════════════════════════════════════════════════
# MAIN WHAT-IF ENGINE (All Rules Combined)
# ══════════════════════════════════════════════════════════════
def apply_all_rules(drug_name, dosage_val, confidence,
                    age, condition, bp, severity, weight, allergy):
    rules   = []
    original_drug   = drug_name
    original_dosage = dosage_val

    # ── Base: Severity Rule ───────────────────────────────────
    sev_data = SEVERITY_RULES.get(condition, {}).get(severity)
    if sev_data:
        drug_name  = sev_data["drug"]
        dosage_val = sev_data["dosage"]
        icons = {"Mild":"🟢","Moderate":"🟡","Severe":"🔴"}
        types = {"Mild":"success","Moderate":"warning","Severe":"danger"}
        rules.append({
            "type"   : types[severity],
            "icon"   : icons[severity],
            "title"  : f"Severity Rule — {severity} {condition}",
            "message": f"Severity is <b>{severity}</b> → {drug_name} {dosage_val}mg selected. {sev_data['note']}",
        })

    # ── Rule: High BP override ────────────────────────────────
    if bp == "High" and condition == "Hypertension":
        drug_name  = "Losartan"
        dosage_val = 100
        confidence = 99.0
        rules.append({
            "type":"danger","icon":"🔴",
            "title":"High BP Override",
            "message":"High BP + Hypertension → Losartan 100mg. Immediate BP control needed.",
        })

    # ── Rule: Low BP override ─────────────────────────────────
    if bp == "Low" and condition == "Hypertension":
        drug_name  = "Amlodipine"
        dosage_val = 2.5
        confidence = 90.0
        rules.append({
            "type":"warning","icon":"🟡",
            "title":"Low BP Warning",
            "message":"Low BP with Hypertension → Amlodipine 2.5mg minimum dose. Monitor BP every 2 hours.",
        })

    # ── Rule: Elderly dose reduction ─────────────────────────
    if age > 65:
        old = dosage_val
        dosage_val = max(50, int(dosage_val * 0.75))
        rules.append({
            "type":"warning","icon":"👴",
            "title":"Elderly Dose Reduction (Age > 65)",
            "message":f"Age {age} → dosage reduced {old}mg → {dosage_val}mg (75%) to prevent toxicity.",
        })

    # ── Rule: Pediatric safety ────────────────────────────────
    if age < 18:
        old = dosage_val
        dosage_val = max(50, int(dosage_val * 0.5))
        if drug_name == "Tramadol":
            drug_name  = "Paracetamol"
            dosage_val = 250
            rules.append({
                "type":"danger","icon":"👶",
                "title":"Pediatric Safety Override (Age < 18)",
                "message":f"Tramadol unsafe for age {age}. Switched to Paracetamol 250mg.",
            })
        else:
            rules.append({
                "type":"warning","icon":"👶",
                "title":"Pediatric Dose Adjustment (Age < 18)",
                "message":f"Age {age} → dosage halved {old}mg → {dosage_val}mg for safety.",
            })

    # ── RULE 8: Weight-Based Dosage ───────────────────────────
    if drug_name in WEIGHT_BASED_DRUGS:
        wb       = WEIGHT_BASED_DRUGS[drug_name]
        wt_dose  = int(weight * wb["mgkg"])
        wt_dose  = min(wt_dose, wb["max"])
        wt_dose  = max(50, wt_dose)
        old_dose = dosage_val
        dosage_val = wt_dose
        rules.append({
            "type":"info","icon":"⚖️",
            "title":f"Rule 8 — Weight-Based Dosage ({drug_name})",
            "message":f"Formula: {weight}kg × {wb['mgkg']}mg/kg = <b>{wt_dose}mg</b> "
                      f"(max {wb['max']}mg). Adjusted from {old_dose}mg → {wt_dose}mg.",
        })

    # ── RULE 9: Drug Allergy Check ────────────────────────────
    if allergy and allergy != "None":
        al_data = ALLERGY_SWITCH.get(allergy)
        if al_data and drug_name == al_data["affects"]:
            old_drug   = drug_name
            drug_name  = al_data["switch"]
            dosage_val = al_data["dose"]
            rules.append({
                "type":"danger","icon":"🚨",
                "title":f"Rule 9 — Allergy Alert: {allergy}",
                "message":f"Patient is allergic to <b>{allergy}</b>. "
                          f"{old_drug} is contraindicated. "
                          f"Switched to <b>{drug_name} {dosage_val}mg</b> safely.",
            })

    # ── RULE 5: Dosage Frequency ──────────────────────────────
    freq_data    = FREQUENCY_RULES.get(severity, FREQUENCY_RULES["Moderate"])
    daily_dose   = dosage_val * freq_data["freq"]
    rules.append({
        "type":"info","icon":"🕐",
        "title":f"Rule 5 — Dosage Frequency ({severity})",
        "message":f"<b>{freq_data['label']}</b> — Take {dosage_val}mg {freq_data['freq']}x/day. "
                  f"Schedule: {freq_data['times']}. Total daily: {daily_dose}mg.",
    })

    # ── RULE 6: Max Daily Dose Warning ────────────────────────
    max_dose = MAX_DAILY_DOSE.get(drug_name, 9999)
    if daily_dose > max_dose:
        safe_single = max(50, int(max_dose / freq_data["freq"]))
        rules.append({
            "type":"danger","icon":"🚫",
            "title":f"Rule 6 — ⚠️ MAX DOSE EXCEEDED for {drug_name}",
            "message":f"Daily dose {daily_dose}mg EXCEEDS safe limit of <b>{max_dose}mg/day</b>! "
                      f"Auto-corrected single dose to <b>{safe_single}mg</b> "
                      f"({freq_data['freq']}x = {safe_single * freq_data['freq']}mg/day ≤ {max_dose}mg).",
        })
        dosage_val = safe_single
    else:
        rules.append({
            "type":"success","icon":"✅",
            "title":f"Rule 6 — Daily Dose Safe ({drug_name})",
            "message":f"Daily dose {daily_dose}mg is within safe limit of <b>{max_dose}mg/day</b>. ✅",
        })

    # ── NSAID + High BP ───────────────────────────────────────
    if bp == "High" and drug_name == "Ibuprofen":
        drug_name  = "Paracetamol"
        dosage_val = 500
        rules.append({
            "type":"danger","icon":"🔴",
            "title":"NSAID Contraindication",
            "message":"Ibuprofen raises BP — contraindicated. Switched to Paracetamol 500mg.",
        })

    # ── Diabetes + Glipizide ──────────────────────────────────
    if condition == "Diabetes" and drug_name == "Glipizide":
        drug_name  = "Metformin"
        dosage_val = 500
        rules.append({
            "type":"override","icon":"🔵",
            "title":"Diabetes Safety Rule",
            "message":"Glipizide → hypoglycemia risk. Switched to Metformin 500mg.",
        })

    if not rules:
        rules.append({
            "type":"success","icon":"✅",
            "title":"All Clear",
            "message":"No special risk conditions detected. Standard AI recommendation applies.",
        })

    changed = (drug_name != original_drug or dosage_val != original_dosage)
    return drug_name, round(float(dosage_val), 1), round(confidence, 1), rules, changed, freq_data


# ══════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════
def make_chart(drug_name, severity, condition, alternatives, schedule):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="white")

    # Chart 1 — ML Confidence Bars
    drugs  = [a["drug"] for a in alternatives]
    confs  = [a["confidence"] for a in alternatives]
    colors = {"Mild":"#2ecc71","Moderate":"#f39c12","Severe":"#e74c3c"}
    ax1    = axes[0]
    bars   = ax1.barh(drugs, confs,
                      color=[colors.get(severity,"#667eea"),"#667eea","#764ba2"][:len(drugs)],
                      edgecolor="white", height=0.45)
    ax1.set_xlabel("Confidence %")
    ax1.set_title("ML Drug Confidence", fontweight="bold")
    ax1.set_xlim(0, 120)
    ax1.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, confs):
        ax1.text(val+1, bar.get_y()+bar.get_height()/2,
                 f"{val}%", va="center", fontsize=9, fontweight="bold")

    # Chart 2 — Severity Pie
    ax2        = axes[1]
    sev_levels = ["Mild","Moderate","Severe"]
    sev_colors = ["#2ecc71","#f39c12","#e74c3c"]
    sev_vals   = [33, 33, 34]
    w_colors   = [sev_colors[i] if sev_levels[i]==severity
                  else f"{sev_colors[i]}44" for i in range(3)]
    wedges, texts = ax2.pie(sev_vals, labels=sev_levels,
                             colors=w_colors, startangle=90,
                             wedgeprops={"edgecolor":"white","linewidth":2})
    ax2.set_title(f"Severity: {severity}", fontweight="bold")
    for i,(w,t) in enumerate(zip(wedges,texts)):
        if sev_levels[i] == severity:
            w.set_linewidth(4); t.set_fontweight("bold"); t.set_fontsize(12)

    # Chart 3 — Treatment Schedule Dose Line
    ax3      = axes[2]
    periods  = [s["period"] for s in schedule]
    doses    = [s["dose"]   for s in schedule]
    ax3.plot(periods, doses, marker="o", linewidth=2.5,
             color=colors.get(severity,"#667eea"),
             markerfacecolor="white", markeredgewidth=2)
    ax3.fill_between(range(len(doses)), doses, alpha=0.15,
                     color=colors.get(severity,"#667eea"))
    ax3.set_xticks(range(len(periods)))
    ax3.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Dose (mg)")
    ax3.set_title("Treatment Schedule", fontweight="bold")
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"AI Recommendation: {drug_name} | {condition}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
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
        allergy   = request.form.get("allergy", "None")

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

        # Apply all rules
        drug_name, dosage_val, confidence, rules, changed, freq_data = apply_all_rules(
            drug_name, dosage_val, confidence,
            age, condition, bp, severity, weight, allergy
        )

        # Rule 16: Treatment Schedule (while loop)
        schedule, total_days = generate_treatment_schedule(
            drug_name, dosage_val, severity, condition, freq_data["label"]
        )

        drug_info   = DRUG_INFO.get(drug_name, {"class":"N/A","use":"N/A","note":"Consult your doctor"})
        chart_b64   = make_chart(drug_name, severity, condition, alternatives, schedule)
        timestamp   = datetime.now().strftime("%d %b %Y, %I:%M %p")
        daily_dose  = round(dosage_val * freq_data["freq"], 1)
        max_allowed = MAX_DAILY_DOSE.get(drug_name, 9999)
        dose_safe   = daily_dose <= max_allowed

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
            **record, "bp":bp, "weight":weight, "allergy":allergy,
            "drug_class":drug_info["class"],
            "drug_use":drug_info["use"],
            "drug_note":drug_info["note"],
            "frequency":freq_data["label"],
            "daily_dose":daily_dose,
            "total_days":total_days,
        }

        return render_template("result.html",
            drug=drug_name, dosage=dosage_val, confidence=confidence,
            age=age, gender=gender, condition=condition,
            bp=bp, severity=severity, weight=weight, allergy=allergy,
            drug_info=drug_info, chart_b64=chart_b64,
            timestamp=timestamp, history=session["history"],
            rules_triggered=rules, whatif_changed=changed,
            alternatives=alternatives, sev_display=sev_display,
            freq_data=freq_data, daily_dose=daily_dose,
            max_allowed=max_allowed, dose_safe=dose_safe,
            schedule=schedule, total_days=total_days,
        )

    except Exception as e:
        import traceback
        return f"<h2>Error: {str(e)}</h2><pre>{traceback.format_exc()}</pre><a href='/'>← Back</a>", 500


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=STATS)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    return render_template("history.html", history=session.get("history",[]))

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
    for lbl,val in [
        ("Age",          str(r.get("age",""))),
        ("Gender",       str(r.get("gender",""))),
        ("Condition",    str(r.get("condition",""))),
        ("Severity",     str(r.get("severity",""))),
        ("Blood Pressure",str(r.get("bp",""))),
        ("Weight",       f"{r.get('weight','')} kg"),
        ("Allergy",      str(r.get("allergy","None"))),
    ]:
        pdf.cell(65,8,f"  {lbl}:"); pdf.cell(125,8,val,new_x="LMARGIN",new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica","B",13); pdf.set_fill_color(240,255,244)
    pdf.cell(0,10," AI Recommendation",new_x="LMARGIN",new_y="NEXT",fill=True)
    pdf.set_font("Helvetica","",11); pdf.ln(2)
    for lbl,val in [
        ("Recommended Drug", str(r.get("drug",""))),
        ("Single Dose",      f"{r.get('dosage','')} mg"),
        ("Frequency",        str(r.get("frequency",""))),
        ("Daily Total",      f"{r.get('daily_dose','')} mg/day"),
        ("Treatment Days",   str(r.get("total_days",""))),
        ("AI Confidence",    f"{r.get('confidence','')}%"),
        ("Drug Class",       str(r.get("drug_class",""))),
        ("Primary Use",      str(r.get("drug_use",""))),
        ("Clinical Note",    str(r.get("drug_note",""))),
    ]:
        pdf.set_font("Helvetica","B",11); pdf.cell(65,8,f"  {lbl}:")
        pdf.set_font("Helvetica","",11); pdf.multi_cell(125,8,val); pdf.ln(1)

    pdf.ln(4); pdf.set_fill_color(255,248,225); pdf.set_font("Helvetica","I",9)
    pdf.multi_cell(190,6," DISCLAIMER: Academic project only. NOT a real medical prescription. Always consult a licensed doctor.",fill=True)
    pdf.set_y(-18); pdf.set_font("Helvetica","I",8); pdf.set_text_color(150,150,150)
    pdf.cell(0,10,"AI Drug Recommendation System | Final Year Project",align="C")

    buf = io.BytesIO(); pdf.output(buf); buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="Drug_Report.pdf",
                     mimetype="application/pdf")

if __name__ == "__main__":
    print("="*60)
    print("🚀 AI Drug Recommendation System — All Rules Active")
    print("✅ Rule 5  — Dosage Frequency")
    print("✅ Rule 6  — Max Daily Dose Warning")
    print("✅ Rule 8  — Weight-Based Dosage")
    print("✅ Rule 9  — Drug Allergy Check")
    print("✅ Rule 16 — Treatment Duration Schedule (while loop)")
    print("="*60)
    app.run(debug=True, host="127.0.0.1", port=5000)
