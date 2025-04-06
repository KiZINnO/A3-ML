# test_model.py
import numpy as np
import pytest
from load_model import load_model3, load_meta_data


def test_model_accepts_expected_input():
    """
    Test that the model accepts the expected input format.
    This constructs a sample input from default values and verifies that 
    the model returns a prediction without error.
    """
    model = load_model3()
    scaler, year_default, mileage_default, max_power_default, classes = load_meta_data()
    
    # For this test, defautl values
    year = year_default
    mileage = mileage_default 
    max_power = max_power_default
   
    
    # Construct input features in the same order as used during training:
    # [max_power, mileage, year] followed by the encoded brand vector.
    input_features = np.array([[max_power, mileage, year]])
    # Scale the numeric part (columns 0:3)
    input_features[:, 0:3] = scaler.transform(input_features[:, 0:3])
    # Insert an intercept term (if needed)
    input_features = np.insert(input_features, 0, 1, axis=1)
    
    # Predict – should return an array of shape (1,)
    prediction = model.predict(input_features)[0]
    
    # assert prediction is not None
    # assert prediction.shape == (1,)

def test_model_output_shape():
    """
    Test that the model's output is as expected.
    For a classification model, we expect a single prediction (class index)
    which should be convertible to an integer.
    """
    model = load_model3()
    scaler, year_default, mileage_default, max_power_default, classes = load_meta_data()
    
    year = year_default
    mileage = mileage_default 
    max_power = max_power_default
    
    input_features = np.array([[max_power, mileage, year]])
    input_features[:, 0:3] = scaler.transform(input_features[:, 0:3])
    input_features = np.insert(input_features, 0, 1, axis=1)
    
    prediction = model.predict(input_features)
    
    try:
        # Check that the first element can be interpreted as an integer class index.
        _ = int(prediction[0])
    except Exception:
        pytest.fail("Model output is not an integer class index as expected.")
