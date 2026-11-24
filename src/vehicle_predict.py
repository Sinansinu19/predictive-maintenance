import joblib
import json
import pandas as pd
import shap

def predict_vehicle_failure(vehicle_type, sensor_rpm, sensor_speed, sensor_temp):
    model = joblib.load('models/vehicle/vehicle_failure_model.joblib')
    with open('models/vehicle/threshold.json', 'r') as f:
        threshold = json.load(f)['threshold']

    # Prepare input dataframe
    input_data = {
        'sensor_rpm': [sensor_rpm],
        'sensor_speed': [sensor_speed],
        'sensor_temp': [sensor_temp],
        'vehicle_type_heavy_truck': [1 if vehicle_type == 'heavy_truck' else 0],
        'vehicle_type_passenger_car': [1 if vehicle_type == 'passenger_car' else 0]
    }
    
    # Ensure column order matches the trained model exactly
    features = model.get_booster().feature_names
    df = pd.DataFrame(input_data)[features]

    # Run inference
    prob = model.predict_proba(df)[0][1]
    is_failure = bool(prob >= threshold)
    
    risk_level = "LOW"
    if prob > 0.4: risk_level = "MODERATE"
    if prob >= threshold: risk_level = "HIGH"

    # Generate SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)[0]
    contributions = dict(zip(features, shap_values))
    
    # Translate SHAP values into Mechanic Recommendations
    recommendations = []
    if is_failure:
        if contributions.get('sensor_rpm', 0) > 0:
            recommendations.append("Engine RPM is critically high. Inspect engine load and throttle body.")
        if contributions.get('sensor_temp', 0) > 0:
            recommendations.append("Sensor temperature is driving failure risk. Check cooling system and radiator.")
        if contributions.get('sensor_speed', 0) > 0:
            recommendations.append("Vehicle speed is abnormally high for current conditions. Advise operator speed regulation.")
        if not recommendations:
            recommendations.append("Multiple sensors indicate degradation. Perform a full system diagnostic.")
    else:
        recommendations.append("Sensors are within normal operating ranges. No immediate maintenance required.")

    return {
        "vehicle_type": vehicle_type,
        "failure_probability": round(float(prob), 4),
        "threshold": threshold,
        "failure_predicted": is_failure,
        "risk_level": risk_level,
        "shap_contributions": {k: round(float(v), 4) for k, v in contributions.items()},
        "mechanic_recommendations": recommendations
    }

if __name__ == "__main__":
    # Testing Raashid's extreme edge case to prove it is fixed
    result = predict_vehicle_failure("heavy_truck", 45000, 1200, 800)
    print(json.dumps(result, indent=2))