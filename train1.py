import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import os

train_csv_zip_path=
test_csv_zip_path=
train_csv_path=
test_csv_path=

if not os.path.exists(train_csv_path):
    with ZipFile(train_csv_zip_path, 'r') as zip_ref:
      zip_ref.extractall()
    
if not os.path.exists(test_csv_path):
    with ZipFile(test_zip_path, 'r') as zip_ref:
      zip_ref.extractall()
    
train_df=pd.read_csv(train_csv_path)
test_df=pd.read_csv(test_csv_path)

X_train= train_df.drop(columns=['label']).values
y_train = train_df['label'].values
X_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

X_train = X_train / 255.0
X_test = X_test / 255.0

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
