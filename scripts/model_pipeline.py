"""
Shared classification-modelling helpers for Report 3, reused across all
three predictive tasks (14_predict_weapon.py, 15_predict_arrest.py,
16_predict_race.py) instead of rewriting the train/test/CV/evaluate/plot
loop three times.

Preprocessing note: numeric features are always scaled (StandardScaler).
This is redundant for the tree-based models (Decision Tree, Random Forest
are invariant to per-feature monotonic scaling -- their splits don't
change), but harmless, and it means one shared ColumnTransformer serves
Logistic Regression and the tree models alike rather than needing two
separate preprocessing paths.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
RANDOM_STATE = 44


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])


def get_models():
    """class_weight="balanced" on all three: every target in this report is
    imbalanced (weapon-found ~1.2%, arrest ~6.1%), and without it a model can
    score near-100% accuracy just by predicting the majority class every
    time -- as happened with Random Forest's default threshold in early runs
    (near-zero recall despite ~99% accuracy). Balancing trades some accuracy
    for a model that actually engages with the minority class."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                                    class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE,
                                                  class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
                                                  n_jobs=-1, class_weight="balanced"),
    }


def get_feature_importance(pipeline, model_name, feature_names_out):
    model = pipeline.named_steps["model"]
    if model_name == "Logistic Regression":
        coef = model.coef_
        raw = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
    elif hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    else:
        return None
    imp = pd.Series(raw, index=feature_names_out)
    # collapse one-hot dummy columns (e.g. "cat__pct_14") back to their source feature
    def base_name(name):
        name = name.replace("num__", "").replace("cat__", "")
        for sep in ["_"]:
            pass
        return name
    imp.index = [base_name(n) for n in imp.index]
    # re-split off the one-hot category suffix (original col name is always a
    # prefix ending right before the first underscore that follows it -- since
    # we control the source column list, match against it instead of guessing)
    return imp


def collapse_importance_by_source(imp_series, source_cols, agg="sum"):
    """One-hot expands e.g. 'pct' into 'pct_1', 'pct_2', ... -- collapse those
    back onto 'pct' so importance is reported per original variable.

    agg="sum" (Random Forest / tree models): each dummy's impurity-reduction
    is a genuinely additive quantity, so summing across a variable's dummies
    correctly gives that variable's total contribution.

    agg="mean" (Logistic Regression): a coefficient is a log-odds effect size
    on one indicator column, not an additive quantity -- summing across many
    dummies would mechanically inflate any high-cardinality categorical's
    score regardless of whether any individual category matters (verified:
    76 modest per-precinct coefficients, none above ~1.2, summed to ~25 --
    far above single-column flags like cs_bulge at 0.32 -- purely because
    there were 76 terms being added, not because precinct actually dominated
    the model). The mean absolute per-category coefficient reports the
    *typical* size of one category's effect, which is the fair comparison to
    a single 0/1 flag's own coefficient -- and it's more robust than taking
    the max, which would just spotlight whichever single category happened
    to have the noisiest/smallest-sample estimate."""
    total = {c: 0.0 for c in source_cols}
    count = {c: 0 for c in source_cols}
    for idx, val in imp_series.items():
        match = max((c for c in source_cols if idx == c or idx.startswith(c + "_")), key=len, default=None)
        if match:
            total[match] += val
            count[match] += 1
    if agg == "sum":
        out = total
    else:
        out = {c: (total[c] / count[c] if count[c] else 0.0) for c in source_cols}
    return pd.Series(out).sort_values(ascending=False)


def run_classification_suite(df, numeric_cols, categorical_cols, target, task_label,
                              fig_prefix, class_names=None, cv_folds=5):
    """Fits Dummy + LogReg + Decision Tree + Random Forest on one feature/
    target configuration; returns a metrics DataFrame and a dict of fitted
    pipelines, and saves a confusion-matrix figure + feature-importance figure."""
    feature_cols = numeric_cols + categorical_cols
    data = df[feature_cols + [target]].copy()
    X, y = data[feature_cols], data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    is_binary = y.nunique() == 2
    avg = "binary" if is_binary else "macro"

    preproc = build_preprocessor(numeric_cols, categorical_cols)
    models = get_models()
    models["Baseline (stratified guess)"] = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)

    # sklearn's confusion_matrix sorts labels alphabetically/numerically by
    # default -- class_names must be supplied in that same order, since it's
    # used purely as display text against whatever order this produces.
    label_order = sorted(y.unique().tolist())

    rows, fitted, importances = [], {}, {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        pipe = Pipeline([("preproc", preproc), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro" if not is_binary else "f1", n_jobs=-1)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average=avg, zero_division=0),
            "Recall": recall_score(y_test, preds, average=avg, zero_division=0),
            "F1-Score": f1_score(y_test, preds, average=avg, zero_division=0),
            "CV F1 (mean)": cv_scores.mean(),
            "CV F1 (std)": cv_scores.std(),
        })
        fitted[name] = pipe

        if name != "Baseline (stratified guess)":
            try:
                feat_names_out = pipe.named_steps["preproc"].get_feature_names_out()
                raw_imp = get_feature_importance(pipe, name, feat_names_out)
                agg = "mean" if name == "Logistic Regression" else "sum"
                importances[name] = collapse_importance_by_source(raw_imp, feature_cols, agg=agg)
            except Exception:
                pass

    metrics_df = pd.DataFrame(rows).set_index("Model")

    if class_names is not None and list(class_names) != label_order:
        assert len(class_names) == len(label_order), (
            f"class_names length {len(class_names)} != number of classes {len(label_order)}")

    _plot_confusion_matrices(fitted, X_test, y_test, class_names, label_order, task_label, fig_prefix)
    if importances:
        _plot_feature_importance(importances, task_label, fig_prefix)

    return metrics_df, fitted, importances, (X_test, y_test)


def _plot_confusion_matrices(fitted, X_test, y_test, class_names, label_order, task_label, fig_prefix):
    plot_models = {k: v for k, v in fitted.items() if k != "Baseline (stratified guess)"}
    fig, axes = plt.subplots(1, len(plot_models), figsize=(5 * len(plot_models), 4.5))
    if len(plot_models) == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, plot_models.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test), labels=label_order)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names if class_names is not None else label_order)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=",",
                   xticks_rotation=30 if class_names and len(class_names) > 2 else 0)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"Confusion matrices -- {task_label}", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{fig_prefix}_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(importances, task_label, fig_prefix, top_n=12):
    fig, axes = plt.subplots(1, len(importances), figsize=(6 * len(importances), 5.5))
    if len(importances) == 1:
        axes = [axes]
    for ax, (name, imp) in zip(axes, importances.items()):
        top = imp.head(top_n).sort_values()
        ax.barh(top.index, top.values, color="#2a78d6")
        ax.set_title(name, fontsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"Top {top_n} feature importances -- {task_label}", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{fig_prefix}_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)