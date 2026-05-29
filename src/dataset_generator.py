import random
import pandas as pd

from config_loader import load_config
from oracle import oracle_label


def generate_dataset(config):
    dataset_config = config["dataset"]
    features_config = config["features"]

    num_samples = dataset_config["num_samples"]
    output_path = dataset_config["output_path"]
    random_seed = dataset_config["random_seed"]

    random.seed(random_seed)

    data = []

    for _ in range(num_samples):
        speed = round(random.uniform(
            features_config["speed"]["min"],
            features_config["speed"]["max"]
        ), 2)

        distance = round(random.uniform(
            features_config["distance"]["min"],
            features_config["distance"]["max"]
        ), 2)

        lane_offset = round(random.uniform(
            features_config["lane_offset"]["min"],
            features_config["lane_offset"]["max"]
        ), 2)

        weather_risk = round(random.uniform(
            features_config["weather_risk"]["min"],
            features_config["weather_risk"]["max"]
        ), 2)

        road_curvature = round(random.uniform(
            features_config["road_curvature"]["min"],
            features_config["road_curvature"]["max"]
        ), 2)

        label = oracle_label(
            speed=speed,
            distance=distance,
            lane_offset=lane_offset,
            weather_risk=weather_risk,
            road_curvature=road_curvature
        )

        data.append({
            "speed": speed,
            "distance": distance,
            "lane_offset": lane_offset,
            "weather_risk": weather_risk,
            "road_curvature": road_curvature,
            "label": label
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    return df


def main():
    config = load_config()
    df = generate_dataset(config)

    print("Dataset generated successfully!")
    print(f"Samples: {len(df)}")
    print(df.head())
    print("\nLabel distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()