import pandas as pd

from config_loader import load_config


def normalize_rule(rule):
    return (
        rule.replace("AND", "and")
            .replace("OR", "or")
            .replace("NOT", "not")
    )


def evaluate_rule(row, rule=None):
    if rule is None:
        config = load_config()
        rule = config["rule"]["initial_rule"]

    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "speed": float(row["speed"]),
        "distance": float(row["distance"]),
        "lane_offset": float(row["lane_offset"]),
        "weather_risk": float(row["weather_risk"]),
        "road_curvature": float(row["road_curvature"])
    }

    try:
        result = eval(normalize_rule(rule), safe_globals, safe_locals)
        return "Pass" if result else "Inconclusive"
    except Exception:
        return "Invalid"


def evaluate_dataset(rule=None, dataset_path=None):
    config = load_config()

    if rule is None:
        rule = config["rule"]["initial_rule"]

    if dataset_path is None:
        dataset_path = config["dataset"]["output_path"]

    df = pd.read_csv(dataset_path)

    predictions = []
    mismatches = []

    for index, row in df.iterrows():
        prediction = evaluate_rule(row, rule)
        actual = row["label"]

        predictions.append(prediction)

        if prediction != "Inconclusive" and prediction != actual:
            mismatches.append({
                "index": int(index),
                "speed": float(row["speed"]),
                "distance": float(row["distance"]),
                "lane_offset": float(row["lane_offset"]),
                "weather_risk": float(row["weather_risk"]),
                "road_curvature": float(row["road_curvature"]),
                "rule_prediction": prediction,
                "actual_label": actual
            })

    decided_cases = [p for p in predictions if p != "Inconclusive"]

    accuracy = 0
    if decided_cases:
        accuracy = (len(decided_cases) - len(mismatches)) / len(decided_cases)

    return {
        "total_samples": len(df),
        "decided_cases": len(decided_cases),
        "mismatches": len(mismatches),
        "accuracy": accuracy,
        "mismatch_examples": mismatches[:10]
    }


def main():
    config = load_config()
    rule = config["rule"]["initial_rule"]

    result = evaluate_dataset(rule)

    print("Initial Rule Evaluation")
    print("-----------------------")
    print(f"Rule: {rule}")
    print(f"Total samples: {result['total_samples']}")
    print(f"Decided cases: {result['decided_cases']}")
    print(f"Mismatches: {result['mismatches']}")
    print(f"Rule accuracy on decided cases: {result['accuracy']:.2f}")

    print("\nExample mismatches:")
    for item in result["mismatch_examples"][:5]:
        print(item)


if __name__ == "__main__":
    main()