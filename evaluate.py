import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

model = joblib.load("model/model.pkl")
X_test = pd.read_csv("data/X_test_processed.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.show()

importance = model.feature_importances_

features = [
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



importance_df = pd.DataFrame({

    "Feature":features,

    "Importance":importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,8))

sns.barplot(

    data=importance_df.head(15),

    x="Importance",

    y="Feature"

)

plt.title("Top 15 Important Features")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

plt.show()



with open(
    "results/evaluation_results.txt",
    "w"
) as f:

    f.write("NETWORK INTRUSION DETECTOR\n")
    f.write("="*40+"\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")

    f.write(f"ROC AUC : {auc:.4f}\n\n")

    f.write("Classification Report\n")
    f.write(classification_report(y_test,y_pred))

