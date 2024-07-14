from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
housing = pd.read_csv("data.csv")

# Define the train_test_split function
def train_test_split(data, test_split_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_split_ratio)
    test_index = shuffled[:test_set_size]
    train_index = shuffled[test_set_size:]
    return data.iloc[train_index], data.iloc[test_index]

# Split the dataset
train_set, test_set = train_test_split(housing, 0.2)

# Stratified Shuffle Split
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    stratified_train_set = housing.loc[train_index]
    stratified_test_set = housing.loc[test_index]

# Prepare the data for training
housing = stratified_train_set.drop(columns=["MEDV", "NOX", "CHAS", "RAD", "TAX", "B"])
housing_label = stratified_train_set["MEDV"].copy()

# Define the pipeline
housing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Train the model
housing_numpy_array = housing_pipeline.fit_transform(housing)
model = RandomForestRegressor()
model.fit(housing_numpy_array, housing_label)

# Define input validation function
def validate_input(data):
    try:
        return [float(value) for value in data]
    except ValueError:
        raise ValueError("Invalid input: Please enter valid numerical values.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        data = [
            request.form.get('crim', ''),
            request.form.get('zn', ''),
            request.form.get('indus', ''),
            request.form.get('rm', ''),
            request.form.get('age', ''),
            request.form.get('dist', ''),
            request.form.get('ptr', ''),
            request.form.get('lstat', '')
        ]
        
        try:
            input_data = np.array([validate_input(data)])
            input_data_prepared = housing_pipeline.transform(input_data)
            prediction = model.predict(input_data_prepared)
            price = prediction[0] * 1000 * 83.52
            return render_template('test.html', crim=data[0], zn=data[1], indus=data[2], rm=data[3], age=data[4], dist=data[5], ptr=data[6], lstat=data[7], price=price)
        except ValueError as e:
            return render_template('test.html', error_message=str(e))
    else:
        return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)
