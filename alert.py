import numpy as np 


COASTAL_CITIES = [
    {"name": "Chennai", "lat": 13.0827, "lng": 80.2707},
    {"name": "Visakhapatnam", "lat": 17.6868, "lng": 83.2185},
    {"name": "Bhubaneswar", "lat": 20.2961, "lng": 85.8245},
    {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639},
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
    {"name": "Kochi", "lat": 9.9312, "lng": 76.2673},
]

CONFIDENCE_RADIUS_KM_BY_STEP = {1: 17, 2: 31, 3: 44, 4: 59, 5: 72, 6: 87}


def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def check_risk(predicted_lat, predicted_lng, confidence_radius_km, cities=COASTAL_CITIES, alert_threshold_km=200):
    at_risk = []
    for city in cities:
        dist = haversine_distance(predicted_lat, predicted_lng, city["lat"], city["lng"])
        if dist <= (confidence_radius_km + alert_threshold_km):
            at_risk.append({"city": city["name"], "distance_km": round(dist, 1)})
    return at_risk


def generate_alert(prediction, horizon_step):
    """
    prediction: one dict from predict_cyclone_path(), e.g. {"hours_ahead": 3, "lat": .., "lon": .., "wind": ..}
    horizon_step: 1-6, matching the position of this prediction in the returned list
    Returns: alert message string, or None if no city is at risk
    """
    radius_km = CONFIDENCE_RADIUS_KM_BY_STEP.get(horizon_step, 90)
    at_risk_cities = check_risk(prediction["lat"], prediction["lon"], radius_km)

    if not at_risk_cities:
        return None

    message = f"""⚠️ CYCLONE EARLY INDICATOR (AI-Assisted, Research Prototype)

Predicted storm position in {prediction['hours_ahead']}h: ({prediction['lat']:.2f}, {prediction['lon']:.2f})
Estimated wind speed: {prediction['wind']:.0f} knots
Model uncertainty radius: ~{radius_km} km

Cities within potential risk zone:
"""
    for c in at_risk_cities:
        message += f"  - {c['city']} (~{c['distance_km']} km from predicted center)\n"

    message += "\n⚠️ This is an AI research prototype, not an official warning. Please refer to IMD (mausam.imd.gov.in) or NDMA for verified alerts."
    return message


if __name__ == "__main__":
    from inference import load_cyclone_model, predict_cyclone_path

    model, checkpoint = load_cyclone_model('best_lstm_early.pt')
    example_input = [
        [21.3, 88.0, 89.0, 953.0],
        [22.0, 88.1, 89.0, 954.0],
        [22.7, 88.4, 88.0, 954.0],
    ]
    predictions = predict_cyclone_path(model, checkpoint, example_input)

    for step, pred in enumerate(predictions, start=1):
        alert = generate_alert(pred, step)
        if alert:
            print(alert)
            break
