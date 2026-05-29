import pandas as pd

from config_loader import load_config
from rule_evaluator import evaluate_rule


def detect_inconsistencies(df, rule=None):
    config = load_config()

    if rule is None:
        rule = config["rule"]["initial_rule"]

    inconsistencies = []

    for index, row in df.iterrows():
        prediction = evaluate_rule(row, rule)
        actual = row["label"]

        if prediction != "Inconclusive" and prediction != actual:
            inconsistencies.append({
                "index": int(index),
                "speed": float(row["speed"]),
                "distance": float(row["distance"]),
                "lane_offset": float(row["lane_offset"]),
                "weather_risk": float(row["weather_risk"]),
                "road_curvature": float(row["road_curvature"]),
                "rule_prediction": prediction,
                "actual_label": actual
            })

    return inconsistencies


def main():
    config = load_config()
    df = pd.read_csv(config["dataset"]["output_path"])

    rule = config["rule"]["initial_rule"]
    inconsistencies = detect_inconsistencies(df, rule)

    print(f"Rule: {rule}")
    print(f"Found {len(inconsistencies)} inconsistencies")

    for item in inconsistencies[:10]:
        print(item)


if __name__ == "__main__":
    main()