import json
import matplotlib.pyplot as plt


def main():
    with open("results/logs.json", "r") as f:
        logs = json.load(f)

    labels = ["Initial"]
    mismatches = [logs["initial_inconsistencies"]]
    accuracies = [0.85]

    for item in logs["iterations"]:
        source = item.get("source", "llm")
        iteration = item["iteration"]

        labels.append(f"{source}-{iteration}")

        metrics = item["validation"]["metrics"]
        mismatches.append(metrics["mismatches"])
        accuracies.append(metrics["accuracy"])

    plt.figure(figsize=(11, 5))
    plt.bar(labels, mismatches)
    plt.xlabel("Refinement Step")
    plt.ylabel("Mismatch Count")
    plt.title("Mismatch Reduction Across Adaptive Refinement")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/mismatch_plot.png")
    print("Saved: results/mismatch_plot.png")

    plt.figure(figsize=(11, 5))
    plt.bar(labels, accuracies)
    plt.xlabel("Refinement Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Adaptive Refinement")
    plt.xticks(rotation=30)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("results/accuracy_plot.png")
    print("Saved: results/accuracy_plot.png")


if __name__ == "__main__":
    main()