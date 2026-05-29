import json

from openai import OpenAI

from config_loader import load_config
from rule_sanitizer import sanitize_rule
from pattern_miner import mine_unsafe_patterns


def build_prompt(original_rule, counterfactuals, feedback=None):

    return f"""
You are a senior safety engineer responsible for refining symbolic safety rules.

Current rule:

{original_rule}

The rule incorrectly classifies some unsafe situations as PASS.

Your task is to refine the rule.

Requirements:

1. Keep the original rule intact.
2. Add only the minimum additional constraints needed.
3. Focus on eliminating unsafe PASS classifications.
4. Prefer adding NOT(...) exclusion regions.
5. Produce a single final rule.
6. Do not explain your reasoning.
7. Output ONLY the final rule.

Observed unsafe regions:

- speed > 14 AND distance < 5
- weather_risk > 0.6 AND distance < 6

Examples of unsafe cases:

- speed=19.4, distance=3.44, weather_risk=0.25
- speed=21.9, distance=3.95, weather_risk=0.14
- speed=22.47, distance=3.15, weather_risk=0.08
- weather_risk=0.68, distance=3.67
- weather_risk=0.64, distance=2.73

The refined rule should reject these unsafe regions while preserving the original behavior as much as possible.

Return only the final rule.
"""

def refine_rule(original_rule, counterfactuals, feedback=None):
    config = load_config()

    api_config = config["openai_api"]

    client = OpenAI(
        base_url=api_config["base_url"],
        api_key=api_config["api_key"],
        timeout=180.0,
        max_retries=3
    )

    prompt = build_prompt(
        original_rule,
        counterfactuals,
        feedback
    )

    response = client.chat.completions.create(
        model=api_config["model"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You generate strict formal safety rules."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_rule = response.choices[0].message.content.strip()
    
    print("\nRAW GPT OUTPUT:")
    print(raw_rule)
    print()
    sanitized_rule = sanitize_rule(raw_rule)

    return sanitized_rule


def main():
    from config_loader import load_config
    from counterfactual_generator import generate_counterfactual
    from inconsistency_detector import detect_inconsistencies

    import pandas as pd

    config = load_config()

    original_rule = config["rule"]["initial_rule"]

    df = pd.read_csv(config["dataset"]["output_path"])

    inconsistencies = detect_inconsistencies(
        df,
        original_rule
    )

    counterfactuals = []

    for item in inconsistencies:
        cf = generate_counterfactual(item)

        if cf:
            counterfactuals.append(cf)

    refined_rule = refine_rule(
        original_rule,
        counterfactuals
    )

    print("GPT refined rule:")
    print(refined_rule)


if __name__ == "__main__":
    main()