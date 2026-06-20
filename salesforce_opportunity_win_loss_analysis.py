from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_URLS = {
    "sales_pipeline": "https://raw.githubusercontent.com/DiogoSoares3/CRM-AI-Analysis/main/data/sales_pipeline.csv",
    "accounts": "https://raw.githubusercontent.com/DiogoSoares3/CRM-AI-Analysis/main/data/accounts.csv",
    "sales_teams": "https://raw.githubusercontent.com/DiogoSoares3/CRM-AI-Analysis/main/data/sales_teams.csv",
    "products": "https://raw.githubusercontent.com/DiogoSoares3/CRM-AI-Analysis/main/data/products.csv",
}

NUMERIC_FEATURES = [
    "estimated_amount",
    "log_company_revenue",
    "log_employee_count",
    "account_age_at_engage",
    "is_subsidiary",
]

CATEGORICAL_FEATURES = [
    "product",
    "product_series",
    "industry",
    "sales_region",
    "office_location",
    "engage_month",
    "engage_quarter",
    "engage_year",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_public_data() -> dict[str, Path]:
    ensure_directories()
    local_files: dict[str, Path] = {}
    for name, url in DATA_URLS.items():
        local_path = DATA_DIR / f"{name}.csv"
        if not local_path.exists():
            urlretrieve(url, local_path)
        local_files[name] = local_path
    return local_files


def load_source_tables() -> dict[str, pd.DataFrame]:
    local_files = download_public_data()
    return {name: pd.read_csv(path) for name, path in local_files.items()}


def prepare_opportunity_frame(source_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pipeline_df = source_tables["sales_pipeline"].copy()
    accounts_df = source_tables["accounts"].copy()
    teams_df = source_tables["sales_teams"].copy()
    products_df = source_tables["products"].copy()

    pipeline_df["product"] = pipeline_df["product"].replace({"GTXPro": "GTX Pro"})

    merged = (
        pipeline_df.merge(accounts_df, on="account", how="left")
        .merge(teams_df, on="sales_agent", how="left")
        .merge(products_df, on="product", how="left")
    )

    closed = merged.loc[merged["deal_stage"].isin(["Won", "Lost"])].copy()
    closed["engage_date"] = pd.to_datetime(closed["engage_date"], errors="coerce")
    closed["close_date"] = pd.to_datetime(closed["close_date"], errors="coerce")
    closed["is_won"] = (closed["deal_stage"] == "Won").astype(int)

    closed["industry"] = (
        closed["sector"]
        .replace({"technolgy": "technology"})
        .fillna("unknown")
        .str.title()
    )
    closed["sales_region"] = closed["regional_office"].fillna("Unknown")
    closed["product_series"] = closed["series"].fillna("Unknown")
    closed["office_location"] = closed["office_location"].fillna("Unknown")
    closed["engage_month"] = closed["engage_date"].dt.strftime("%b").fillna("Unknown")
    closed["engage_quarter"] = (
        "Q" + closed["engage_date"].dt.quarter.fillna(0).astype(int).astype(str)
    ).replace({"Q0": "Unknown"})
    closed["engage_year"] = (
        closed["engage_date"].dt.year.fillna(-1).astype(int).astype(str).replace("-1", "Unknown")
    )
    closed["account_age_at_engage"] = (
        closed["engage_date"].dt.year - closed["year_established"]
    )
    closed["is_subsidiary"] = closed["subsidiary_of"].notna().astype(int)
    closed["log_company_revenue"] = np.log1p(closed["revenue"])
    closed["log_employee_count"] = np.log1p(closed["employees"])

    # The public opportunity table exposes won revenue, not a pre-close quote amount.
    # Product list price is the cleanest public proxy for deal size without leaking the outcome.
    closed["estimated_amount"] = closed["sales_price"]

    keep_columns = [
        "opportunity_id",
        "sales_agent",
        "manager",
        "account",
        "product",
        "product_series",
        "industry",
        "sales_region",
        "office_location",
        "deal_stage",
        "is_won",
        "engage_date",
        "close_date",
        "estimated_amount",
        "revenue",
        "employees",
        "account_age_at_engage",
        "is_subsidiary",
        "log_company_revenue",
        "log_employee_count",
        "engage_month",
        "engage_quarter",
        "engage_year",
    ]
    return closed[keep_columns].copy()


def summarize_win_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            opportunities=("opportunity_id", "count"),
            wins=("is_won", "sum"),
        )
        .reset_index()
    )
    summary["losses"] = summary["opportunities"] - summary["wins"]
    summary["win_rate"] = summary["wins"] / summary["opportunities"]
    return summary.sort_values(["win_rate", "opportunities"], ascending=[False, False])


def build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def fit_logistic_regression(df: pd.DataFrame) -> dict[str, object]:
    X = df[MODEL_FEATURES]
    y = df["is_won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "baseline_accuracy": y_test.mean(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).T
    report_df.index.name = "label"

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "classification_report": report_df,
    }


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "raw_feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )

    coef_df["feature_group"] = coef_df["raw_feature"].str.extract(r"^(?:num|cat)__([^_]+)")
    coef_df["feature_label"] = coef_df["raw_feature"].apply(pretty_feature_name)
    coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()

    return coef_df.sort_values("coefficient")


def pretty_feature_name(raw_feature: str) -> str:
    if raw_feature.startswith("num__"):
        clean = raw_feature.replace("num__", "")
        return clean.replace("_", " ").title()

    clean = raw_feature.replace("cat__", "")
    field_name, _, value = clean.partition("_")
    label_map = {
        "product": "Product",
        "product_series": "Product Series",
        "industry": "Industry",
        "sales_region": "Sales Region",
        "office_location": "Office Location",
        "engage_month": "Engage Month",
        "engage_quarter": "Engage Quarter",
        "engage_year": "Engage Year",
    }
    label = label_map.get(field_name, field_name.replace("_", " ").title())
    return f"{label} = {value}"


def key_factor_table(coef_df: pd.DataFrame) -> pd.DataFrame:
    allowed_groups = {
        "estimated_amount",
        "log_company_revenue",
        "log_employee_count",
        "account_age_at_engage",
        "is_subsidiary",
        "product",
        "industry",
        "sales_region",
        "engage_quarter",
        "engage_year",
    }
    filtered = coef_df.loc[coef_df["feature_group"].isin(allowed_groups)].copy()
    return filtered.sort_values("absolute_coefficient", ascending=False).head(12)


def plot_closed_outcomes(df: pd.DataFrame) -> Path:
    counts = df["deal_stage"].value_counts().reindex(["Won", "Lost"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=["#2E8B57", "#C44E52"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Closed Opportunity Outcomes")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Opportunity Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 25, f"{value:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "closed_outcomes.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_win_rate(summary_df: pd.DataFrame, x_col: str, title: str, filename: str) -> Path:
    plot_df = summary_df.copy()
    plot_df["win_rate_pct"] = plot_df["win_rate"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x=x_col, y="win_rate_pct", color="#4C72B0", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Win Rate (%)")
    ax.tick_params(axis="x", rotation=30)
    for patch, value in zip(ax.patches, plot_df["win_rate_pct"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(y_test: pd.Series, predictions: np.ndarray) -> Path:
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Lost", "Won"])
    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Logistic Regression Confusion Matrix")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_key_coefficients(key_factors: pd.DataFrame) -> Path:
    plot_df = key_factors.sort_values("coefficient")
    colors = np.where(plot_df["coefficient"] >= 0, "#2E8B57", "#C44E52")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["feature_label"], plot_df["coefficient"], color=colors)
    ax.set_title("Largest Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "top_coefficients.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_findings(
    metrics: dict[str, float],
    win_rate_by_industry: pd.DataFrame,
    win_rate_by_product: pd.DataFrame,
    win_rate_by_region: pd.DataFrame,
    key_factors: pd.DataFrame,
) -> list[str]:
    best_industry = win_rate_by_industry.iloc[0]
    lowest_industry = win_rate_by_industry.iloc[-1]
    best_product = win_rate_by_product.iloc[0]
    lowest_product = win_rate_by_product.iloc[-1]
    best_region = win_rate_by_region.iloc[0]

    positive_factors = key_factors.loc[key_factors["coefficient"] > 0, "feature_label"].head(3).tolist()
    negative_factors = key_factors.loc[key_factors["coefficient"] < 0, "feature_label"].head(3).tolist()

    findings = [
        (
            f"The holdout accuracy came in at {metrics['accuracy']:.3f} versus a closed-deal win baseline "
            f"of {metrics['baseline_accuracy']:.3f}. In other words, the public fields carry some signal, "
            f"but not enough to beat a simple always-win guess on raw accuracy."
        ),
        (
            f"Industry mattered: {best_industry.iloc[0]} led the closed-deal win table at "
            f"{best_industry['win_rate']:.1%}, while {lowest_industry.iloc[0]} trailed at "
            f"{lowest_industry['win_rate']:.1%}."
        ),
        (
            f"Product mix mattered too: {best_product.iloc[0]} posted the highest win rate at "
            f"{best_product['win_rate']:.1%}, while {lowest_product.iloc[0]} was the weakest product in "
            f"the closed-deal sample at {lowest_product['win_rate']:.1%}."
        ),
        (
            f"Region was a smaller but still visible factor. The {best_region.iloc[0]} region had the best "
            f"win rate at {best_region['win_rate']:.1%}, and the coefficient table also leans positive for that region."
        ),
        (
            "The largest positive model coefficients point to "
            + ", ".join(positive_factors[:3])
            + "."
        ),
        (
            "The largest negative model coefficients point to "
            + ", ".join(negative_factors[:3])
            + "."
        ),
    ]
    return findings


def save_analysis_outputs(
    prepared_df: pd.DataFrame,
    win_rate_by_industry: pd.DataFrame,
    win_rate_by_product: pd.DataFrame,
    win_rate_by_region: pd.DataFrame,
    model_results: dict[str, object],
    coef_df: pd.DataFrame,
    key_factors: pd.DataFrame,
    findings: list[str],
) -> dict[str, str]:
    prepared_df.to_csv(OUTPUT_DIR / "closed_opportunities_modeling_frame.csv", index=False)
    win_rate_by_industry.to_csv(OUTPUT_DIR / "win_rate_by_industry.csv", index=False)
    win_rate_by_product.to_csv(OUTPUT_DIR / "win_rate_by_product.csv", index=False)
    win_rate_by_region.to_csv(OUTPUT_DIR / "win_rate_by_region.csv", index=False)
    coef_df.to_csv(OUTPUT_DIR / "logistic_coefficients.csv", index=False)
    key_factors.to_csv(OUTPUT_DIR / "key_factors.csv", index=False)
    model_results["classification_report"].to_csv(OUTPUT_DIR / "classification_report.csv")

    metrics_path = OUTPUT_DIR / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(model_results["metrics"], handle, indent=2)

    findings_path = OUTPUT_DIR / "analysis_summary.txt"
    with findings_path.open("w", encoding="utf-8") as handle:
        for line in findings:
            handle.write(f"- {line}\n")

    plot_paths = {
        "closed_outcomes": str(plot_closed_outcomes(prepared_df)),
        "industry_win_rate": str(
            plot_win_rate(
                win_rate_by_industry,
                "industry",
                "Win Rate by Industry",
                "win_rate_by_industry.png",
            )
        ),
        "product_win_rate": str(
            plot_win_rate(
                win_rate_by_product,
                "product",
                "Win Rate by Product",
                "win_rate_by_product.png",
            )
        ),
        "confusion_matrix": str(
            plot_confusion_matrix(model_results["y_test"], model_results["predictions"])
        ),
        "top_coefficients": str(plot_key_coefficients(key_factors)),
    }
    return plot_paths


def run_analysis() -> dict[str, object]:
    sns.set_theme(style="whitegrid")

    source_tables = load_source_tables()
    prepared_df = prepare_opportunity_frame(source_tables)

    win_rate_by_industry = summarize_win_rates(prepared_df, "industry")
    win_rate_by_product = summarize_win_rates(prepared_df, "product")
    win_rate_by_region = summarize_win_rates(prepared_df, "sales_region")

    model_results = fit_logistic_regression(prepared_df)
    coef_df = coefficient_table(model_results["model"])
    key_factors = key_factor_table(coef_df)

    findings = build_findings(
        model_results["metrics"],
        win_rate_by_industry,
        win_rate_by_product,
        win_rate_by_region,
        key_factors,
    )

    plot_paths = save_analysis_outputs(
        prepared_df,
        win_rate_by_industry,
        win_rate_by_product,
        win_rate_by_region,
        model_results,
        coef_df,
        key_factors,
        findings,
    )

    return {
        "prepared_df": prepared_df,
        "win_rate_by_industry": win_rate_by_industry,
        "win_rate_by_product": win_rate_by_product,
        "win_rate_by_region": win_rate_by_region,
        "model_results": model_results,
        "coefficient_table": coef_df,
        "key_factors": key_factors,
        "findings": findings,
        "plot_paths": plot_paths,
    }


def main() -> None:
    results = run_analysis()
    metrics = results["model_results"]["metrics"]

    print("Salesforce Opportunity Win-Loss Analysis")
    print("=" * 44)
    print(f"Closed opportunities modeled: {len(results['prepared_df']):,}")
    print(
        "Metrics: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1_score']:.3f}, "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )
    print("\nKey findings:")
    for line in results["findings"]:
        print(f"- {line}")


if __name__ == "__main__":
    main()
