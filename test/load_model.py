import pickle
import mlflow
import os

filename = './model_training/model/a3_prediction.model'

loaded_model = pickle.load(open(filename, 'rb'))
model = loaded_model
scaler = loaded_model['scaler']
year_default = 2014
mileage_default = 21.14
max_power_default = 103.52

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