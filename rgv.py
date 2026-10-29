print("Script started...")
print("Importing matplotlib...")
import matplotlib
matplotlib.use('Agg')
print("Importing pandas...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# File paths (Local)
train_path = "pp5i_train.gr.csv"
test_path = "pp5i_test.gr.csv"
class_path = "pp5i_train_class.txt"

def run_analysis():
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    class_df = pd.read_csv(class_path)
    class_np = class_df.to_numpy().ravel()

    print("Preprocessing data...")
    ttdf_rem = train_df.iloc[:, 1:].clip(20, 16000)
    tsdf_rem = test_df.iloc[:, 1:].clip(20, 16000)
    ttdf_cal = ttdf_rem.max(axis=1) / ttdf_rem.min(axis=1)
    del_ind = ttdf_cal[ttdf_cal < 2].index
    train_tdf = train_df.drop(del_ind)
    test_tdf = test_df.drop(del_ind)
    X_train_full = train_tdf.drop('SNO', axis=1).T
    X_test_full = test_tdf.drop('SNO', axis=1).T
    if X_train_full.shape[0] != len(class_np):
        min_len = min(X_train_full.shape[0], len(class_np))
        X_train_full = X_train_full.iloc[:min_len, :]
        class_np = class_np[:min_len]
    f_vals, _ = f_classif(X_train_full, class_np)
    gene_scores = pd.Series(f_vals, index=X_train_full.columns)
    top_genes = gene_scores.sort_values(ascending=False).head(25).index
    X_train = X_train_full[top_genes]
    X_test = X_test_full[top_genes]
    le = LabelEncoder()
    y_train = le.fit_transform(class_np)

    print("Comparing classifiers...")
    results = {}
    classifiers = {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=350, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "MLP": MLPClassifier(hidden_layer_sizes=(25, 25), max_iter=250, random_state=42)
    }
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        results[name] = clf.score(X_train, y_train)
    print(pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"]))

    print("Running cross-validation...")
    classifiers_cv = {
        "GNB": GaussianNB(),
        "DTC": DecisionTreeClassifier(),
        "KNC": KNeighborsClassifier(n_neighbors=3),
        "NN": MLPClassifier(hidden_layer_sizes=(25,25), max_iter=250, random_state=42),
        "ETC": ExtraTreesClassifier(n_estimators=350, random_state=42)
    }
    N_values = range(2, 31, 2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_cv = {name: [] for name in classifiers_cv.keys()}
    for N in N_values:
        top_genes_N = gene_scores.sort_values(ascending=False).head(N).index
        X_sel = X_train_full[top_genes_N]
        for name, clf in classifiers_cv.items():
            scores = []
            for train_idx, test_idx in cv.split(X_sel, class_np):
                X_tr, X_te = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
                y_tr, y_te = class_np[train_idx], class_np[test_idx]
                clf.fit(X_tr, y_tr)
                scores.append(accuracy_score(y_te, clf.predict(X_te)))
            results_cv[name].append(1 - np.mean(scores))
    
    all_scores = []
    for name in classifiers_cv.keys():
        for N, err in zip(N_values, results_cv[name]):
            all_scores.append((1-err, N, name))
    best_acc, best_N, best_clf = max(all_scores)
    print(f"Best N: {best_N}, Best Classifier: {best_clf}, Best Accuracy: {best_acc:.2%}")

    # Gradio interface
    def experiment(classifier_name, n_genes, show_chart):
        top_genes_exp = gene_scores.sort_values(ascending=False).head(n_genes).index
        X_train_exp = X_train_full[top_genes_exp]
        X_test_exp = X_test_full[top_genes_exp]
        clf_exp = classifiers[classifier_name]
        clf_exp.fit(X_train_exp, y_train)
        acc_exp = clf_exp.score(X_train_exp, y_train)
        preds_exp = le.inverse_transform(clf_exp.predict(X_test_exp))
        result = f"✅ Classifier: {classifier_name}\n✅ Accuracy: {acc_exp:.2%}\n✅ N Genes: {n_genes}\n\n📌 Predictions:\n"
        result += "\n".join([f"Sample {i+1}: {p}" for i,p in enumerate(preds_exp)])
        fig = None
        if show_chart:
            fig, ax = plt.subplots(figsize=(6,4))
            gene_scores.sort_values(ascending=False).head(30).plot(kind="bar", ax=ax)
            ax.set_title("Top 30 Gene Scores")
            plt.tight_layout()
        return result, fig

    demo = gr.Interface(
        fn=experiment,
        inputs=[
            gr.Dropdown(list(classifiers.keys()), label="Choose Classifier"),
            gr.Slider(5, 50, value=25, step=1, label="Select Top N Genes"),
            gr.Checkbox(label="Show Gene Score Chart?", value=False)
        ],
        outputs=[gr.Textbox(label="Output"), gr.Plot(label="Chart")],
        title="🧬 Genetic Disease Prediction"
    )
    print("Launching Gradio...")
    demo.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    print("Main block reached")
    # run_analysis()
