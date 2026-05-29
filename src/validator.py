import re
import pandas as pd

from config_loader import load_config
from rule_evaluator import normalize_rule, evaluate_rule


ALLOWED_WORDS = {"AND", "OR", "NOT"}
ALLOWED_CHARS_PATTERN = r"^[a-zA-Z0-9_\s\.\(\)<>=!]+$"


def get_allowed_variables():
    config = load_config()
    return set(config["features"].keys())


def empty_metrics():
    return {
        "decided": 0,
        "mismatches": 0,
        "invalid": 0,
        "accuracy": 0
    }


def validate_grammar(rule):
    if rule is None:
        return False, ["Rule is None"]

    if rule == "INVALID_RULE":
        return False, [
            "LLM output was rejected by sanitizer because it was malformed"
        ]

    allowed_variables = get_allowed_variables()

    if not re.match(ALLOWED_CHARS_PATTERN, rule):
        return False, ["Rule contains invalid characters"]

    tokens = re.findall(r"[a-zA-Z_]+", rule)
    errors = []

    for token in tokens:
        if token in ALLOWED_WORDS:
            continue

        if token.lower() in {"and", "or", "not"}:
            continue

        if token not in allowed_variables:
            errors.append(f"Unknown token or variable: {token}")

    try:
        compile(normalize_rule(rule), "<string>", "eval")
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")

    return len(errors) == 0, errors


def evaluate_refined_rule(rule, dataset_path=None):
    config = load_config()

    if dataset_path is None:
        dataset_path = config["dataset"]["output_path"]

    df = pd.read_csv(dataset_path)

    decided = 0
    mismatches = 0
    invalid = 0

    for _, row in df.iterrows():
        prediction = evaluate_rule(row, rule)
        actual = row["label"]

        if prediction == "Invalid":
            invalid += 1
            continue

        if prediction != "Inconclusive":
            decided += 1

            if prediction != actual:
                mismatches += 1

    accuracy = 0

    if decided > 0:
        accuracy = (decided - mismatches) / decided

    return {
        "decided": decided,
        "mismatches": mismatches,
        "invalid": invalid,
        "accuracy": accuracy
    }


def validate_rule(rule):
    config = load_config()
    min_decided_cases = config["validation"]["min_decided_cases"]

    grammar_ok, grammar_errors = validate_grammar(rule)

    if not grammar_ok:
        return {
            "valid": False,
            "errors": grammar_errors,
            "metrics": empty_metrics()
        }

    metrics = evaluate_refined_rule(rule)

    errors = []

    if metrics["invalid"] > 0:
        errors.append("Rule cannot be evaluated on some samples")

    if metrics["mismatches"] > 0:
        errors.append(f"Rule still has {metrics['mismatches']} mismatches")

    if metrics["decided"] < min_decided_cases:
        errors.append(
            f"Rule is too restrictive. It decides only {metrics['decided']} cases. "
            f"Minimum required decided cases is {min_decided_cases}."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "metrics": metrics
    }


def main():
    config = load_config()

    test_rules = [
        config["rule"]["initial_rule"],
        (
            "distance >= 2.2 AND lane_offset <= 1.5 AND weather_risk <= 0.7 "
            "AND NOT (speed > 14 AND distance < 5) "
            "AND NOT (weather_risk > 0.6 AND distance < 6)"
        ),
        (
            "distance >= 2.2 AND lane_offset <= 1.5 AND weather_risk <= 0.7 "
            "AND NOT ((speed > 14 AND distance < 5) OR (weather_risk > 0.6 AND distance < 6))"
        )
    ]

    for rule in test_rules:
        print("\nRule:")
        print(rule)

        result = validate_rule(rule)

        print("Validation result:")
        print(result)


if __name__ == "__main__":
    main()