import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# ... existing code ...
# Update this line
file_path = '/Users/dariocoding/Downloads/cardio_train.csv'
try:
    # First try to open the file to test permissions
    with open(file_path, 'r') as f:
        pass
    # If that works, read with pandas
    df = pd.read_csv(file_path, delimiter=';')
except PermissionError:
    print(f"\nPermission Error: Cannot access {file_path}")
    print("Please try the following:")
    print("1. Open System Preferences > Security & Privacy > Privacy")
    print("2. Select 'Full Disk Access' from the left sidebar")
    print("3. Click the '+' button and add your Python/IDE application")
    print("4. Restart your Python/IDE application")
    exit(1)
except FileNotFoundError:
    print(f"\nFile not found: {file_path}")
    print("Please verify the file exists and the path is correct")
    exit(1)
# ... existing code ...

# Data preprocessing
df['age'] = df['age'] // 365  # Convert age from days to years
df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]  # Remove invalid blood pressure values
df = df[(df['height'] > 100) & (df['height'] < 220)]  # Remove invalid height values
df = df[(df['weight'] > 30) & (df['weight'] < 200)]  # Remove invalid weight values

# Create BMI feature
df['bmi'] = df['weight'] / ((df['height']/100) ** 2)

# Prepare features and target
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
X = df[features]
y = df['cardio']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Create Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Heart Attack Risk Analysis Dashboard'),
    
    # Feature Importance Plot
    html.Div([
        html.H2('Feature Importance'),
        dcc.Graph(
            id='feature-importance-plot',
            figure=px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance in Predicting Heart Attack Risk'
            )
        )
    ]),
    
    # Age vs Blood Pressure Scatter Plot
    html.Div([
        html.H2('Age vs Blood Pressure Distribution'),
        dcc.Graph(
            id='age-bp-scatter',
            figure=px.scatter(
                df,
                x='age',
                y='ap_hi',
                color='cardio',
                title='Age vs Systolic Blood Pressure',
                labels={'cardio': 'Cardiovascular Disease'}
            )
        )
    ]),
    
    # BMI Distribution
    html.Div([
        html.H2('BMI Distribution by Cardiovascular Disease'),
        dcc.Graph(
            id='bmi-dist',
            figure=px.histogram(
                df,
                x='bmi',
                color='cardio',
                marginal='box',
                title='BMI Distribution by Cardiovascular Disease Status',
                labels={'cardio': 'Cardiovascular Disease'}
            )
        )
    ]),
    
    # Risk Prediction Input Form
    html.Div([
        html.H2('Heart Attack Risk Predictor'),
        html.Div([
            html.Label('Age:'),
            dcc.Input(id='age-input', type='number', value=50),
            html.Label('Gender (1-female, 2-male):'),
            dcc.Input(id='gender-input', type='number', value=1),
            html.Label('Height (cm):'),
            dcc.Input(id='height-input', type='number', value=165),
            html.Label('Weight (kg):'),
            dcc.Input(id='weight-input', type='number', value=70),
            html.Label('Systolic Blood Pressure:'),
            dcc.Input(id='ap-hi-input', type='number', value=120),
            html.Label('Diastolic Blood Pressure:'),
            dcc.Input(id='ap-lo-input', type='number', value=80),
            html.Label('Cholesterol (1-normal, 2-above normal, 3-well above normal):'),
            dcc.Input(id='cholesterol-input', type='number', value=1),
            html.Label('Glucose (1-normal, 2-above normal, 3-well above normal):'),
            dcc.Input(id='glucose-input', type='number', value=1),
            html.Label('Smoking (0-no, 1-yes):'),
            dcc.Input(id='smoke-input', type='number', value=0),
            html.Label('Alcohol (0-no, 1-yes):'),
            dcc.Input(id='alcohol-input', type='number', value=0),
            html.Label('Physical Activity (0-no, 1-yes):'),
            dcc.Input(id='active-input', type='number', value=1),
            html.Button('Predict Risk', id='predict-button', n_clicks=0)
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px', 'maxWidth': '300px'}),
        html.Div(id='prediction-output')
    ])
])

# Callback for risk prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('age-input', 'value'),
     State('gender-input', 'value'),
     State('height-input', 'value'),
     State('weight-input', 'value'),
     State('ap-hi-input', 'value'),
     State('ap-lo-input', 'value'),
     State('cholesterol-input', 'value'),
     State('glucose-input', 'value'),
     State('smoke-input', 'value'),
     State('alcohol-input', 'value'),
     State('active-input', 'value')]
)
def predict_risk(n_clicks, age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alcohol, active):
    if n_clicks > 0:
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Prepare input data
        input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, 
                               cholesterol, glucose, smoke, alcohol, active, bmi]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = rf_model.predict_proba(input_scaled)[0]
        
        risk_percentage = prediction[1] * 100
        
        return html.Div([
            html.H3(f'Risk of Cardiovascular Disease: {risk_percentage:.1f}%'),
            html.P('Risk Level: ' + ('High' if risk_percentage > 50 else 'Low'))
        ])
    
    return ''

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)


