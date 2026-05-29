import argparse
import json
import pandas as pd

from config_loader import load_config
from inconsistency_detector import detect_inconsistencies
from counterfactual_generator import generate_counterfactual
from ollama_refiner import refine_rule
from validator import validate_rule
from fallback_rule_generator import generate_fallback_rule
from pattern_miner import mine_unsafe_patterns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive LLM-based Safety Rule Refinement Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to config JSON file"
    )

    return parser.parse_args()


def build_feedback(refined_rule, validation):
    feedback = "\n".join(validation["errors"])

    metrics = validation.get("metrics", {})

    if metrics.get("mismatches", 0) > 0:
        feedback += """
The rule still allows unsafe cases.
Refine the rule by excluding unsafe regions using AND NOT clauses.
"""

    if "too restrictive" in feedback:
        feedback += """
The rule became too restrictive.
Preserve the original coverage as much as possible.
"""

    return feedback


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_path = config["dataset"]["output_path"]
    original_rule = config["rule"]["initial_rule"]
    max_iterations = config["ollama"]["max_iterations"]
    use_fallback = config["ollama"].get("use_fallback", True)

    print("Adaptive Rule Refinement Pipeline")
    print("---------------------------------")
    print(f"Config: {args.config}")

    df = pd.read_csv(dataset_path)

    inconsistencies = detect_inconsistencies(df, original_rule)
    print(f"Initial inconsistencies: {len(inconsistencies)}")

    counterfactuals = []

    for item in inconsistencies:
        cf = generate_counterfactual(item)

        if cf:
            counterfactuals.append(cf)

    print(f"Generated counterfactuals: {len(counterfactuals)}")

    mined_patterns = mine_unsafe_patterns(counterfactuals)

    print("\nMined unsafe patterns:")
    for pattern in mined_patterns:
        print(pattern)

    logs = {
        "config_path": args.config,
        "original_rule": original_rule,
        "mined_patterns": mined_patterns,
        "initial_inconsistencies": len(inconsistencies),
        "counterfactuals_count": len(counterfactuals),
        "iterations": [],
        "accepted_rule": None,
        "accepted_by": None
    }

    feedback = None
    best_rule = None

    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}")

        refined_rule = refine_rule(
            original_rule,
            counterfactuals,
            feedback
        )

        print("LLM proposed rule:")
        print(refined_rule)

        validation = validate_rule(refined_rule)

        print("Validation:")
        print(validation)

        logs["iterations"].append({
            "iteration": iteration,
            "proposed_rule": refined_rule,
            "validation": validation,
            "feedback_used": feedback,
            "source": "llm"
        })

        if validation["valid"]:
            best_rule = refined_rule
            logs["accepted_rule"] = best_rule
            logs["accepted_by"] = "llm"

            print("\nAccepted refined rule:")
            print(best_rule)
            break

        feedback = build_feedback(refined_rule, validation)

        invalid_count = sum(
            1 for item in logs["iterations"]
            if item["proposed_rule"] == "INVALID_RULE"
        )

        should_use_fallback = False

        if invalid_count >= 2:
            should_use_fallback = True

        if iteration >= 3 and validation["metrics"]["mismatches"] >= len(inconsistencies):
            should_use_fallback = True

        if use_fallback and should_use_fallback:
            print("\nUsing evidence-based fallback rule generator...")

            fallback_rule = generate_fallback_rule(counterfactuals)
            fallback_validation = validate_rule(fallback_rule)

            print("Fallback rule:")
            print(fallback_rule)

            print("Fallback validation:")
            print(fallback_validation)

            logs["iterations"].append({
                "iteration": f"{iteration}-fallback",
                "proposed_rule": fallback_rule,
                "validation": fallback_validation,
                "feedback_used": "Evidence-based fallback after repeated invalid or ineffective LLM outputs",
                "source": "fallback"
            })

            if fallback_validation["valid"]:
                best_rule = fallback_rule
                logs["accepted_rule"] = best_rule
                logs["accepted_by"] = "fallback"

                print("\nAccepted fallback refined rule:")
                print(best_rule)
                break

    if best_rule is None:
        print("\nNo valid rule found.")

    with open("results/logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("\nLogs saved to results/logs.json")


if __name__ == "__main__":
    main()