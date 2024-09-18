# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import accuracy_score
from pgmpy.inference import VariableElimination

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data"

columns = ['target', 'T3', 'T4', 'TSH', 'TSH Measured', 'T3 Measured', 'T4 Measured', 'Goiter', 
           'Tumor', 'Pregnant', 'Thyroid Surgery', 'I131 Treatment', 'Query Hypothyroid']

data = pd.read_csv(url, delim_whitespace=True, header=None, names=columns, na_values="?")

data = data[['target', 'T3', 'T4', 'TSH', 'Goiter', 'Tumor', 'Pregnant', 'Thyroid Surgery', 'I131 Treatment', 'Query Hypothyroid']]

for col in ['T3', 'T4', 'TSH']:
    data[col].fillna(data[col].mean(), inplace=True)

categorical_columns = ['Goiter', 'Tumor', 'Pregnant', 'Thyroid Surgery', 'I131 Treatment', 'target']

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)

model = BayesianNetwork([('T3', 'target'), ('T4', 'target'), ('TSH', 'target'),
                         ('Goiter', 'target'), ('Tumor', 'target'), ('Pregnant', 'target'),
                         ('Thyroid Surgery', 'target'), ('I131 Treatment', 'target'),
                         ('Query Hypothyroid', 'target')])


model.fit(train_data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

y_pred = []
for index, row in X_test.iterrows():
    q = inference.map_query(variables=['target'], evidence=row.to_dict())
    y_pred.append(q['target'])

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")