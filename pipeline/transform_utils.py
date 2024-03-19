import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Preprocessing function."""
    # Define preprocessing logic using TFT functions
    transformed_features = {}
    
    # Example transformations
    transformed_features['Pregnancies'] = tft.scale_to_z_score(inputs['Pregnancies'], elementwise=True)
    transformed_features['Glucose'] = tft.scale_to_z_score(inputs['Glucose'], elementwise=True)
    transformed_features['BloodPressure'] = tft.scale_to_z_score(inputs['BloodPressure'], elementwise=True)
    transformed_features['SkinThickness'] = tft.scale_to_z_score(tft.log(inputs['SkinThickness']), elementwise=True)
    transformed_features['Insulin'] = tft.scale_to_z_score(tft.log(inputs['Insulin']), elementwise=True)
    transformed_features['BMI'] = tft.scale_to_z_score(inputs['BMI'], elementwise=True)
    transformed_features['DiabetesPedigreeFunction'] = tft.scale_to_z_score(inputs['DiabetesPedigreeFunction'], elementwise=True)
    transformed_features['Age'] = tft.scale_to_z_score(inputs['Age'], elementwise=True)


    # Add more transformations as needed
    
    return transformed_features