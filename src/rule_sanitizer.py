import re

from config_loader import load_config


REQUIRED_BASE_CONSTRAINTS = [
    "distance >= 2.2",
    "lane_offset <= 1.5",
    "weather_risk <= 0.7"
]


def normalize_not_or(rule):
    """
    Convert:
    AND NOT((A) OR (B))
    into:
    AND NOT (A) AND NOT (B)
    """

    pattern = r"AND\s+NOT\s*\(\s*\((.*?)\)\s+OR\s+\((.*?)\)\s*\)"

    match = re.search(pattern, rule)

    if not match:
        return rule

    first = match.group(1).strip()
    second = match.group(2).strip()

    replacement = f"AND NOT ({first}) AND NOT ({second})"

    return re.sub(pattern, replacement, rule)


def extract_rule_from_text(text):
    """
    Extracts the rule-like substring from verbose LLM output.
    """

    if text is None:
        return None

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    start = text.find("distance >=")

    if start == -1:
        return text

    candidate = text[start:]

    # Do NOT stop on "." because decimal numbers contain dots, e.g. 2.2 and 0.7
    stop_phrases = [
        " This rule",
        " This preserves",
        " respectively",
        " Explanation",
        " Based on"
    ]

    stop_positions = []

    for phrase in stop_phrases:
        pos = candidate.find(phrase)
        if pos != -1:
            stop_positions.append(pos)

    if stop_positions:
        candidate = candidate[:min(stop_positions)]

    return candidate.strip()


def sanitize_rule(rule):
    """
    Cleans, extracts, and repairs raw LLM output before validation.
    """

    config = load_config()
    base_rule = config["rule"]["initial_rule"]

    if rule is None:
        return "INVALID_RULE"

    rule = rule.strip()

    # Remove markdown/code fences
    rule = rule.replace("```", "")
    rule = rule.replace("\n", " ")

    # Normalize whitespace
    rule = re.sub(r"\s+", " ", rule).strip()

    # Remove common prefixes
    prefixes = [
        "refined rule:",
        "final refined rule:",
        "here is the refined pass-condition rule:",
        "here's the refined pass-condition rule:",
        "based on the given information, i can generate a refined pass-condition rule as follows:"
    ]

    lowered = rule.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            rule = rule[len(prefix):].strip()
            break

    # Extract rule if LLM returned explanation + rule
    rule = extract_rule_from_text(rule)

    if rule is None or len(rule.strip()) == 0:
        return "INVALID_RULE"

    rule = rule.strip()

    # Remove outer parentheses around the base rule
    rule = rule.replace(
        "(distance >= 2.2 AND lane_offset <= 1.5 AND weather_risk <= 0.7)",
        "distance >= 2.2 AND lane_offset <= 1.5 AND weather_risk <= 0.7"
    )

    # Add spaces after NOT if GPT returns NOT(...)
    rule = re.sub(r"NOT\s*\(", "NOT (", rule)

    # Normalize: AND NOT ((A) OR (B)) -> AND NOT (A) AND NOT (B)
    rule = normalize_not_or(rule)

    # Reject remaining OR only if it was not normalized
    if " OR " in rule:
        return "INVALID_RULE"

    # If model outputs only extension:
    if rule.startswith("AND "):
        rule = base_rule + " " + rule

    # Ensure mandatory original constraints exist
    missing_constraints = []

    for constraint in REQUIRED_BASE_CONSTRAINTS:
        if constraint not in rule:
            missing_constraints.append(constraint)

    if missing_constraints:
        prefix = " AND ".join(missing_constraints)
        rule = prefix + " AND " + rule

    # Normalize spaces again
    rule = re.sub(r"\s+", " ", rule).strip()

    # Reject malformed start
    if rule.startswith("AND") or rule.startswith("NOT"):
        return "INVALID_RULE"

    return rule