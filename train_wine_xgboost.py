import argparse

import numpy as np
import wandb
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost multiclass model on a dataset different from the lab notebook."
    )
    parser.add_argument("--project", default="Lab1-alternative-data")
    parser.add_argument("--run-name", default="xgboost-wine")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb-mode",
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="Use online after running `wandb login`; disabled is best for a quick local smoke test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=dataset.target,
    )

    params = {
        "objective": "multi:softmax",
        "eta": 0.1,
        "max_depth": 4,
        "num_class": len(dataset.target_names),
        "eval_metric": "mlogloss",
        "nthread": 4,
        "seed": args.seed,
    }

    run_config = {
        **params,
        "dataset": "sklearn.load_wine",
        "features": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "test_size": args.test_size,
        "rounds": args.rounds,
    }
    run = None
    callbacks = []

    if args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.project,
            name=args.run_name,
            mode=args.wandb_mode,
            config=run_config,
        )
        callbacks.append(wandb.xgboost.WandbCallback(log_model=True))

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=dataset.feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=dataset.feature_names)

    model = xgb.train(
        params,
        dtrain,
        args.rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
        callbacks=callbacks,
    )

    predictions = model.predict(dtest).astype(int)
    error_rate = float(np.mean(predictions != y_test))
    accuracy = float(accuracy_score(y_test, predictions))

    print(f"Test accuracy = {accuracy:.4f}")
    print(f"Test error = {error_rate:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    if run is not None:
        run.summary["Accuracy"] = accuracy
        run.summary["Error Rate"] = error_rate
        wandb.sklearn.plot_confusion_matrix(
            y_test,
            predictions,
            labels=list(dataset.target_names),
        )
        run.finish()


if __name__ == "__main__":
    main()
