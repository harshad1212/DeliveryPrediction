import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("mode_of_delivery_dataset.csv")

# Selected Features
selected_features = ['Maternal Age', 'Gestational Age', 'Fetal Heart Rate', 'Maternal Blood Pressure', 'Previous C-Section', 'Birth Weight', 'Labor Induced']

# Preprocessing function
def preprocess_data(df, selected_features):
    # Handle missing values
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numeric_imputer = SimpleImputer(strategy='mean')

    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    label_encoders = {}
    if 'Mode of Delivery' in df.columns:
        le = LabelEncoder()
        df['Mode of Delivery'] = le.fit_transform(df['Mode of Delivery'])
        label_encoders['Mode of Delivery'] = le

    X = df[selected_features]
    X = pd.get_dummies(X, drop_first=True)
    y = df['Mode of Delivery'] if 'Mode of Delivery' in df.columns else None

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders

X, y, scaler, label_encoders = preprocess_data(df.copy(), selected_features)

# Train model
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

log_reg = LogisticRegression(max_iter=10000, random_state=42)
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('Logistic Regression', log_reg),
    ('Random Forest', rand_forest)
], voting='hard')

voting_clf.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def predict_mode():
    if request.method == 'POST':
        input_data = {
            'Maternal Age': float(request.form['maternal_age']),
            'Gestational Age': float(request.form['gestational_age']),
            'Fetal Heart Rate': float(request.form['fetal_heart_rate']),
            'Maternal Blood Pressure': float(request.form['maternal_blood_pressure']),
            'Previous C-Section': request.form['previous_c_section'],
            'Birth Weight': float(request.form['birth_weight']),
            'Labor Induced': request.form['labor_induced']
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=pd.get_dummies(df[selected_features], drop_first=True).columns, fill_value=0)

        input_scaled = scaler.transform(input_df)
        prediction = voting_clf.predict(input_scaled)[0]

        # Decode if label encoder was used
        prediction_label = label_encoders['Mode of Delivery'].inverse_transform([prediction])[0]

        return render_template('form.html', prediction=prediction_label)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
