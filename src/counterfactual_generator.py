import pandas as pd

from config_loader import load_config
from inconsistency_detector import detect_inconsistencies
from oracle import oracle_label


FEATURE_STEPS = {
    "speed": 0.5,
    "distance": 0.25,
    "lane_offset": 0.05,
    "weather_risk": 0.05,
    "road_curvature": 0.05
}


FEATURE_DIRECTIONS = {
    "speed": [-1, 1],
    "distance": [1, -1],
    "lane_offset": [-1, 1],
    "weather_risk": [-1, 1],
    "road_curvature": [-1, 1]
}


def try_feature_change(item, feature_name):
    original_label = item["actual_label"]

    original_values = {
        "speed": item["speed"],
        "distance": item["distance"],
        "lane_offset": item["lane_offset"],
        "weather_risk": item["weather_risk"],
        "road_curvature": item["road_curvature"]
    }

    step = FEATURE_STEPS[feature_name]

    for direction in FEATURE_DIRECTIONS[feature_name]:
        modified = original_values.copy()

        for _ in range(30):
            modified[feature_name] = round(
                modified[feature_name] + direction * step,
                2
            )

            if modified[feature_name] < 0:
                break

            new_label = oracle_label(
                speed=modified["speed"],
                distance=modified["distance"],
                lane_offset=modified["lane_offset"],
                weather_risk=modified["weather_risk"],
                road_curvature=modified["road_curvature"]
            )

            if new_label != original_label:
                return {
                    "original": item,
                    "counterfactual": {
                        **modified,
                        "label": new_label
                    },
                    "changed_feature": feature_name,
                    "delta": round(
                        modified[feature_name] -
                        original_values[feature_name],
                        2
                    )
                }

    return None


def generate_counterfactual(item):
    candidates = []

    for feature in FEATURE_STEPS.keys():
        cf = try_feature_change(item, feature)

        if cf:
            step = FEATURE_STEPS[feature]
            normalized_delta = abs(cf["delta"]) / step

            cf["normalized_delta"] = normalized_delta
            candidates.append(cf)

    if not candidates:
        return None

    return min(candidates, key=lambda x: x["normalized_delta"])


def main():
    config = load_config()

    df = pd.read_csv(config["dataset"]["output_path"])

    inconsistencies = detect_inconsistencies(
        df,
        config["rule"]["initial_rule"]
    )

    counterfactuals = []

    for item in inconsistencies:
        cf = generate_counterfactual(item)

        if cf:
            counterfactuals.append(cf)

    print(f"Generated {len(counterfactuals)} counterfactuals")

    print("\nExamples:\n")

    for cf in counterfactuals[:10]:
        print(cf)
        print()


if __name__ == "__main__":
    main()