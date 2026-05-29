def mine_unsafe_patterns(counterfactuals):
    """
    Extract simple unsafe behavioral patterns from counterfactual evidence.
    """

    patterns = []

    speed_distance_cases = []
    weather_distance_cases = []

    for cf in counterfactuals:
        original = cf["original"]

        speed = original["speed"]
        distance = original["distance"]
        weather_risk = original["weather_risk"]

        if speed > 14 and distance < 5:
            speed_distance_cases.append(original)

        if weather_risk > 0.6 and distance < 6:
            weather_distance_cases.append(original)

    if len(speed_distance_cases) > 0:
        patterns.append({
            "name": "high_speed_low_distance",
            "condition": "speed > 14 AND distance < 5",
            "support": len(speed_distance_cases),
            "description": "Unsafe behavior caused by high speed combined with insufficient distance."
        })

    if len(weather_distance_cases) > 0:
        patterns.append({
            "name": "weather_risk_low_distance",
            "condition": "weather_risk > 0.6 AND distance < 6",
            "support": len(weather_distance_cases),
            "description": "Unsafe behavior caused by risky weather and insufficient distance."
        })

    return patterns