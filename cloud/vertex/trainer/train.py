import os
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib


def load_data_from_bigquery(project: str, table: str) -> pd.DataFrame:
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{table}`"
    return client.query(query).to_dataframe()


def main() -> None:
    project = os.environ.get("PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("PROJECT or GOOGLE_CLOUD_PROJECT must be set")

    # Table expected: dataset.table with a 'label' column
    bq_table = os.environ.get("BQ_TABLE", "trading.features_daily")
    model_dir = os.environ.get("AIP_MODEL_DIR", "/gcs/models")

    df = load_data_from_bigquery(project, bq_table)
    if "label" not in df.columns:
        raise RuntimeError("Expected a 'label' column in the BigQuery table for supervised training")

    y = df.pop("label")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(
        n_estimators=int(os.environ.get("N_ESTIMATORS", "300")),
        max_depth=int(os.environ.get("MAX_DEPTH", "6")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "0.1")),
        subsample=float(os.environ.get("SUBSAMPLE", "1.0")),
        colsample_bytree=float(os.environ.get("COLSAMPLE_BYTREE", "1.0")),
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, output_path)
    print({"saved_model": output_path, "rows": len(df)})


if __name__ == "__main__":
    main()


