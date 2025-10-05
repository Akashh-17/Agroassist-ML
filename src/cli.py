import argparse, json
from pathlib import Path
import pandas as pd
from src.helpers import clean_data, build_features, validate_columns


def main():
    parser = argparse.ArgumentParser(description="Crop data CLI")
    parser.add_argument("command", choices=["fetch", "process", "features", "metrics"])
    parser.add_argument("--input", "-i", default="data/Crop_recommendation.csv")
    parser.add_argument("--out", "-o", default="artifacts")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        print("Fetch step assumes file exists locally.")
    else:
        df = pd.read_csv(args.input)
        validate_columns(df)
        if args.command == "process":
            df_clean = clean_data(df)
            df_clean.to_csv(out_dir / "clean.csv", index=False)
            print("Saved clean.csv")
        elif args.command == "features":
            df_feat = build_features(clean_data(df))
            df_feat.to_csv(out_dir / "features.csv", index=False)
            print("Saved features.csv")
        elif args.command == "metrics":
            metrics = {"rows": len(df), "cols": len(df.columns)}
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print("Saved metrics.json")


if __name__ == "__main__":
    main()
