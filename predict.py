import pandas as pd
import joblib

# ==========================
# Load trained objects
# ==========================

model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

protocol_encoder = joblib.load("model/protocol_encoder.pkl")
service_encoder = joblib.load("model/service_encoder.pkl")
flag_encoder = joblib.load("model/flag_encoder.pkl")


# ==========================
# Feature order
# ==========================

FEATURES = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]


# ==========================
# Prediction Function
# ==========================


def safe_float(value, default=0):
    if value is None or str(value).strip() == "":
        return default
    return float(value)

def predict_intrusion(form_data):

    # ----------------------
    # Encode categorical data
    # ----------------------

    protocol = protocol_encoder.transform([form_data["protocol_type"]])[0]

    service = service_encoder.transform([form_data["service"]])[0]

    flag = flag_encoder.transform([form_data["flag"]])[0]

    # ----------------------
    # Create input dictionary
    # ----------------------

    sample = {

    "duration": safe_float(form_data.get("duration")),
    "protocol_type": protocol,
    "service": service,
    "flag": flag,
    "src_bytes": safe_float(form_data.get("src_bytes")),
    "dst_bytes": safe_float(form_data.get("dst_bytes")),
    "land": safe_float(form_data.get("land")),
    "wrong_fragment": safe_float(form_data.get("wrong_fragment")),
    "urgent": safe_float(form_data.get("urgent")),
    "hot": safe_float(form_data.get("hot")),
    "num_failed_logins": safe_float(form_data.get("num_failed_logins")),
    "logged_in": safe_float(form_data.get("logged_in")),
    "num_compromised": safe_float(form_data.get("num_compromised")),
    "root_shell": safe_float(form_data.get("root_shell")),
    "su_attempted": safe_float(form_data.get("su_attempted")),
    "num_root": safe_float(form_data.get("num_root")),
    "num_file_creations": safe_float(form_data.get("num_file_creations")),
    "num_shells": safe_float(form_data.get("num_shells")),
    "num_access_files": safe_float(form_data.get("num_access_files")),
    "num_outbound_cmds": safe_float(form_data.get("num_outbound_cmds")),
    "is_host_login": safe_float(form_data.get("is_host_login")),
    "is_guest_login": safe_float(form_data.get("is_guest_login")),
    "count": safe_float(form_data.get("count")),
    "srv_count": safe_float(form_data.get("srv_count")),
    "serror_rate": safe_float(form_data.get("serror_rate")),
    "srv_serror_rate": safe_float(form_data.get("srv_serror_rate")),
    "rerror_rate": safe_float(form_data.get("rerror_rate")),
    "srv_rerror_rate": safe_float(form_data.get("srv_rerror_rate")),
    "same_srv_rate": safe_float(form_data.get("same_srv_rate")),
    "diff_srv_rate": safe_float(form_data.get("diff_srv_rate")),
    "srv_diff_host_rate": safe_float(form_data.get("srv_diff_host_rate")),
    "dst_host_count": safe_float(form_data.get("dst_host_count")),
    "dst_host_srv_count": safe_float(form_data.get("dst_host_srv_count")),
    "dst_host_same_srv_rate": safe_float(form_data.get("dst_host_same_srv_rate")),
    "dst_host_diff_srv_rate": safe_float(form_data.get("dst_host_diff_srv_rate")),
    "dst_host_same_src_port_rate": safe_float(form_data.get("dst_host_same_src_port_rate")),
    "dst_host_srv_diff_host_rate": safe_float(form_data.get("dst_host_srv_diff_host_rate")),
    "dst_host_serror_rate": safe_float(form_data.get("dst_host_serror_rate")),
    "dst_host_srv_serror_rate": safe_float(form_data.get("dst_host_srv_serror_rate")),
    "dst_host_rerror_rate": safe_float(form_data.get("dst_host_rerror_rate")),
    "dst_host_srv_rerror_rate": safe_float(form_data.get("dst_host_srv_rerror_rate"))

}

   

    df = pd.DataFrame([sample])

    df = df[FEATURES]



    scaled = scaler.transform(df)



    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0]
    print("Prediction probabilities:", probability)


    confidence = round(max(probability) * 100, 2)
    print("Confidence:", confidence)

  

    if prediction == 1:

        result = {
            "prediction": "attack",
            "confidence": confidence,
            "risk": "HIGH",
            "message": "Potential malicious network activity detected.",
            "protocol": form_data["protocol_type"],
            "service": form_data["service"],
            "flag": form_data["flag"]
        }

    else:

        result = {
            "prediction": "normal",
            "confidence": confidence,
            "risk": "LOW",
            "message": "Network traffic appears normal.",
            "protocol": form_data["protocol_type"],
            "service": form_data["service"],
            "flag": form_data["flag"]
        }

    return result