import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#load dataset
train = pd.read_csv("data/KDDTrain+.txt", header=None)
test = pd.read_csv("data/KDDTest+.txt", header=None)
columns = [
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
    'dst_host_srv_rerror_rate',
    'attack',
    'difficulty'
]
train.columns = columns
test.columns = columns
print(train.duplicated().sum())
print("Missing values in train:")
print(train.isnull().sum())

print("\nMissing values in test:")
print(test.isnull().sum())
print(train.dtypes)



protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()

#to understand categorical features, we use fit transform
train['protocol_type'] = protocol_encoder.fit_transform(train['protocol_type'])
test['protocol_type'] = protocol_encoder.transform(test['protocol_type'])

train['service'] = service_encoder.fit_transform(train['service'])
test['service'] = service_encoder.transform(test['service'])

train['flag'] = flag_encoder.fit_transform(train['flag'])
test['flag'] = flag_encoder.transform(test['flag'])

train['attack'] = train['attack'].apply(
    lambda x: 1 if x == 'normal' else 0
)

test['attack'] = test['attack'].apply(
    lambda x: 1 if x == 'normal' else 0
)

X_train = train.drop(columns=['attack', 'difficulty'])
y_train = train['attack']

X_test = test.drop(columns=['attack', 'difficulty'])
y_test = test['attack']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




import joblib

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(protocol_encoder, "model/protocol_encoder.pkl")
joblib.dump(service_encoder, "model/service_encoder.pkl")
joblib.dump(flag_encoder, "model/flag_encoder.pkl")



X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

X_train_df.to_csv("data/X_train_processed.csv", index=False)
X_test_df.to_csv("data/X_test_processed.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)