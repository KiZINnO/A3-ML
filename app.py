import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import mlflow
import os

# functions
def load_meta_data():
    filename = './model_training/model/a3_prediction.model'
    meta = pickle.load(open(filename, 'rb'))

    scaler = meta['scaler']
    year_default = 2009
    mileage_default = 21.79
    max_power_default = 67.05
    classes = meta['classes']
    
    return (scaler, year_default, mileage_default, max_power_default, classes)

def load_model3():

    mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
    mlflow.set_experiment(experiment_name="st124090-a3")

    # Load model from the model registry.
    model_name = "st124090-a3-model"
    model_version = 2

    # load a specific model version
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # load the latest version of a model in that stage.
    # model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

    return model

# --- Create the Dash app ---
app = dash.Dash(__name__)
app.title = "Car Price Prediction"

year_default = 2009
mileage_default = 21.79
max_power_default = 67.05

# Layout of the app
app.layout = html.Div(
    style={'backgroundColor': '#f4f4f4', 'padding': '40px'},
    children=[
        # Header Section with title and explanation
        html.Div([
            html.H1("Car Price Range Prediction", 
                    style={'textAlign': 'center', 'color': '#333', 'marginBottom': '10px'}),
            html.P(
                "Welcome to the Car Price Range Prediction app! This tool uses a machine learning classification model "
                " (loaded directly from ML flow server) to predict the price range of a car based on its year, mileage and max power. "
                " Experiment with the inputs below to see how different specifications affect the estimated price range.",
                style={'textAlign': 'center', 'fontSize': '18px', 'color': '#666', 'marginBottom': '20px', 'lineHeight': '1.5'}
            )
        ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '20px', 
                  'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '40px'}),
        
        # Input Form Section
        html.Div([
            html.Div([
                html.Label("Year", style={'fontSize': '16px', 'color': '#555'}),
                dcc.Input(id='year-input', type='number', placeholder='Enter year', value=year_default,
                          style={'width': '80%', 'padding': '10px', 'margin': '5px 0', 
                                 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Mileage", style={'fontSize': '16px', 'color': '#555'}),
                dcc.Input(id='mileage-input', type='number', placeholder='Enter mileage', value=mileage_default,
                          style={'width': '100%', 'padding': '10px', 'margin': '5px 0', 
                                 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Max Power", style={'fontSize': '16px', 'color': '#555'}),
                dcc.Input(id='maxpower-input', type='number', placeholder='Enter max power', value=max_power_default,
                          style={'width': '100%', 'padding': '10px', 'margin': '5px 0', 
                                 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'marginBottom': '20px'}),
            
            html.Button("Predict", id='predict-button', n_clicks=0,
                        style={'width': '100%', 'padding': '15px', 'backgroundColor': '#28a745',
                               'color': 'white', 'fontSize': '18px', 'border': 'none', 
                               'borderRadius': '5px', 'cursor': 'pointer'}),
            
            # Prediction Output
            html.Div(id='prediction-output', 
                     style={'marginTop': '30px', 'padding': '20px', 'textAlign': 'center', 
                            'fontWeight': 'bold', 'fontSize': '20px', 'borderRadius': '10px', 
                            'backgroundColor': '#0074D9', 'color': 'white'})
        ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px', 
                  'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
    ]
)

# Callback: When the Predict button is clicked, compute and display the prediction.
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('year-input', 'value')],
     State('mileage-input', 'value'),
     State('maxpower-input', 'value')
)
def predict_price(n_clicks, year, mileage, max_power):
    # Only predict if the button has been clicked at least once
    if n_clicks is None or n_clicks == 0:
        return ""
    
    # Load model from MLflow server
    model = load_model3()
    
    # Load metadata (scaler, default values, and price classes)
    scaler, default_year, default_mileage, default_max_power, classes = load_meta_data()
    
    # Use default values if any input is missing
    if year is None:
        year = default_year
    if mileage is None:
        mileage = default_mileage
    if max_power is None:
        max_power = default_max_power
        
    # Prepare input features.
    # Note: The order here is [max_power, mileage, year] followed by the encoded brand vector.
    input_features = np.array([[year, mileage, max_power]])
    # Scale only the numeric features (columns 0 to 2)
    input_features[:, 0:3] = scaler.transform(input_features[:, 0:3])
    # Add an intercept term if the model was trained with one.
    input_features = np.insert(input_features, 0, 1, axis=1)
    
    # Predict using the loaded model
    predicted_class = model.predict(input_features)[0]
    
    # Convert the predicted class index into a price range string.
    prediction = f" The prediction of the car price range is between ${classes[predicted_class]} - ${classes[predicted_class + 1]}"
    
    return prediction

if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', port=8888)