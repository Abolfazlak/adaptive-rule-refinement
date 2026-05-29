import json

from openai import OpenAI

from config_loader import load_config
from pattern_miner import mine_unsafe_patterns
from rule_sanitizer import sanitize_rule


def build_symbolic_rule(base_rule, patterns):
    rule = base_rule

    for pattern in patterns:
        condition = pattern["condition"]
        clause = f"AND NOT ({condition})"

        if clause not in rule:
            rule += f" {clause}"

    return rule


def build_fallback_prompt(base_rule, patterns):
    return f"""
You are a senior safety rule synthesis assistant.

Generate exactly one complete Pass-condition rule.

Base rule:
{base_rule}

Unsafe patterns mined from counterfactual evidence:
{json.dumps(patterns, indent=2)}

Instructions:
1. Preserve the base rule exactly.
2. Add one exclusion for each unsafe pattern.
3. Prefer this form:
   AND NOT (<unsafe_condition>)
4. Do not explain.
5. Do not use markdown.
6. Output only the final rule.

Valid example:
{base_rule} AND NOT (speed > 14 AND distance < 5) AND NOT (weather_risk > 0.6 AND distance < 6)

Final rule:
"""


def generate_fallback_rule(counterfactuals=None):
    config = load_config()

    base_rule = config["rule"]["initial_rule"]
    api_config = config["openai_api"]

    if counterfactuals is None:
        counterfactuals = []

    patterns = mine_unsafe_patterns(counterfactuals)

    symbolic_rule = build_symbolic_rule(base_rule, patterns)

    client = OpenAI(
        base_url=api_config["base_url"],
        api_key=api_config["api_key"],
        timeout=180.0,
        max_retries=3
    )

    prompt = build_fallback_prompt(base_rule, patterns)

    try:
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
        sanitized_rule = sanitize_rule(raw_rule)

        if sanitized_rule == "INVALID_RULE":
            return symbolic_rule

        for pattern in patterns:
            condition = pattern["condition"]

            direct_clause = f"NOT ({condition})"

            if direct_clause not in sanitized_rule:
                return symbolic_rule

        return sanitized_rule

    except Exception:
        return symbolic_rule