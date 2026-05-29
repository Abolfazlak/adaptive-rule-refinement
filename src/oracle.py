def oracle_label(speed, distance, lane_offset, weather_risk, road_curvature):
    """
    Ground-truth behavior of the simulated autonomous driving system.
    """

    if distance < 2.2:
        return "Fail"

    if lane_offset > 1.5:
        return "Fail"

    if speed > 14 and distance < 5:
        return "Fail"

    if weather_risk > 0.85 and speed > 12:
        return "Fail"

    if road_curvature > 0.85 and speed > 16:
        return "Fail"

    if weather_risk > 0.6 and distance < 6:
        return "Fail"

    return "Pass"