import json 
import os
from networkx import efficiency
import numpy as np
import matplotlib.pyplot as plt
from .config_loader import load_config

groups = {
        "Instructions": ["top_down", "bottom_up"],
        "Questions": ["learner_questions", "teacher_questions"],
        "Dialogic TD": ["mixed_top-down_learner_questions", "mixed_top-down_teacher_questions"],
        "Dialogic BU": ["mixed_bottom-up_learner_questions", "mixed_bottom-up_teacher_questions"],
        "Dialogic Qs": ["mixed_learner_questions", "mixed_teacher_questions"],
        "Expert" : ["expert", "expert_with_text"]
    }

wings_groups = {
        "top_down" : ["top_down", "top_down_2", "top_down_3"],
        "mixed_learner_questions": ["mixed_learner_questions", "mixed_learner_questions_2", "mixed_learner_questions_3"],
        "learner_questions": ["learner_questions", "learner_questions_2", "learner_questions_3"],
        "bottom_up": ["bottom_up", "bottom_up_2", "bottom_up_3"],
        "expert": ["expert", "expert_2", "expert_3"]
    }

label_map = {
    "top_down": "TD",
    "bottom_up": "BU",
    "learner_questions": "LQ",
    "teacher_questions": "TQ",
    "mixed_top-down_learner_questions": "mix-TD-LQ",
    "mixed_top-down_teacher_questions": "mix-TD-TQ",
    "mixed_bottom-up_learner_questions": "mix-BU-LQ",
    "mixed_bottom-up_teacher_questions": "mix-BU-TQ",
    "mixed_learner_questions": "mix-LQ",
    "mixed_teacher_questions": "mix-TQ",
    "expert": "Expert"
}

def factorize(ontology):
    entities = [e.lower() for e in ontology.keys()]
    features = _extract_all_features(ontology)
    values = _extract_all_values(ontology)
    features_values = _extract_feature_values(ontology)
    entities_features = _extract_entities_features(ontology)
    triplets = _extract_triplets(ontology)
    return {
        "entities": entities,
        "features": features,
        "values": values,
        "features_values": features_values,
        "entities_features": entities_features,
        "triplets": triplets
    }

def _extract_feature_values(ontology):
    """Extract feature values for each entity."""
    feature_values = set()
    for _, entity_data in ontology.items():
        if isinstance(entity_data, dict):
            for feature, value in entity_data.items():
                if isinstance(value, (str)):
                    feature_values.add((feature.lower(), value.lower()))
    return feature_values

def _extract_entities_features(ontology):
    """Extract feature values for each entity."""
    entities_values = set()
    for entity, entity_data in ontology.items():
        if isinstance(entity_data, dict):
            for feature, _ in entity_data.items():
                    entities_values.add((entity.lower(), feature.lower()))
    return entities_values

def _extract_triplets(ontology):
    """Extract triplets (entity, feature, value) from the ontology."""
    triplets = set()
    for entity, entity_data in ontology.items():
        if isinstance(entity_data, dict):
            for feature, value in entity_data.items():
                if isinstance(value, (str)):
                    triplets.add((entity.lower(), feature.lower(), value.lower()))
    return triplets
def _extract_all_features(ontology):
    """Extract all unique features from the ontology."""
    features = set()
    for entity_data in ontology.values():
        if isinstance(entity_data, dict):
            features.update(entity_data.keys())
    return [f.lower() for f in features]

def _extract_all_values(ontology):
    """Extract all unique values from the ontology."""
    values = set()
    for entity_data in ontology.values():
        if isinstance(entity_data, dict):
            for value in entity_data.values():
                if isinstance(value, (str, int, float)):
                    values.add(str(value))
    return [v.lower() for v in values]


def get_data(input_path, groups=groups):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")
    data = {}
    avg_res = {}
    var_res = {}
    std_res = {}

    # Load data for all strategies
    for _, strategies in groups.items():
        for strategy in strategies:
            file_path = os.path.join(input_path, f"alignment_{strategy}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Results file not found: {file_path}")
            data[strategy] = json.load(open(file_path, "r", encoding="utf-8"))

    # Compute averages and variances
    for strategy, results in data.items():
        print(f"[Loading Data] Loaded alignment results for strategy: {strategy}")
        sums = {}
        sums_sq = {}

        # Aggregate sums and squared sums
        for _, run in results.items():
            for key, metrics in run.items():
                if key not in sums:
                    sums[key] = {m: 0.0 for m in metrics}
                    sums_sq[key] = {m: 0.0 for m in metrics}
                for metric, value in metrics.items():
                    sums[key][metric] += value
                    sums_sq[key][metric] += value ** 2

        n_run = len(results)
        avg_res[strategy] = {}
        var_res[strategy] = {}
        std_res[strategy] = {}

        for key in sums:
            avg_res[strategy][key] = {}
            var_res[strategy][key] = {}
            std_res[strategy][key] = {} 
            for metric in sums[key]:
                mean = sums[key][metric] / n_run
                # Sample variance (use n-1 for unbiased estimate)
                var = ((sums_sq[key][metric] - n_run * mean**2) / (n_run - 1)) if n_run > 1 else 0.0
                std = np.sqrt(var)
                std_res[strategy][key][metric] = std
                avg_res[strategy][key][metric] = mean
                var_res[strategy][key][metric] = var

    return avg_res, var_res, std_res

def heatmap_with_std(avg_data, std_data, title, output_path, extremes = False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # --- Convert nested dicts → DataFrames ---
    avg_df = pd.DataFrame(avg_data).T
    std_df = pd.DataFrame(std_data).T

    # Align orientation (metrics as rows, strategies as columns)
    avg_df = avg_df.T
    std_df = std_df.T

    # Remove unwanted rows (like ground_truth)
    if "ground_truth" in avg_df.index:
        avg_df = avg_df.drop("ground_truth")
        std_df = std_df.drop("ground_truth", errors="ignore")

    # Apply optional label mapping
   
    avg_df = avg_df.rename(columns=label_map)
    std_df = std_df.rename(columns=label_map)

    # --- Normalize rows (for consistent coloring) ---
    normalized = avg_df.copy()
    for row in avg_df.index:
        row_min, row_max = avg_df.loc[row].min(), avg_df.loc[row].max()
        if row_max > row_min:
            normalized.loc[row] = (avg_df.loc[row] - row_min) / (row_max - row_min)
        else:
            normalized.loc[row] = 0.0  # handle constant rows

    # --- Build annotated strings: "mean ± std" ---
    annotated = avg_df.copy().astype(object)  # avoid dtype warnings
    for r in avg_df.index:
        for c in avg_df.columns:
            m = avg_df.loc[r, c]
            s = std_df.loc[r, c] if c in std_df.columns else np.nan
            annotated.loc[r, c] = f"{m:.2f} ± {s:.2f}" if not np.isnan(s) else f"{m:.2f}"

    # --- Plot heatmap ---
    n_rows, n_cols = normalized.shape
    if extremes:
        fig_width = max(6, n_cols * 1.2)
        fig_height = max(8, n_rows * 0.8)
    else:
        fig_width = max(12, n_cols * 1.2)
        fig_height = max(6, n_rows * 0.6)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        normalized.astype(float),
        annot=annotated,
        fmt="",  # already formatted strings
        cmap="YlGnBu",
        cbar=False,
        annot_kws={"fontsize": 8}
    )

    plt.title(f"Heatmap of {title} (mean ± std, row-normalized)", fontsize=14, pad=10)
    plt.ylabel("Metrics")
    plt.xlabel("Strategies")
    plt.xticks(rotation=45, ha="right")

    # --- Save or show ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def heatmap(data, title, output_path, extremes=False): 
    import pandas as pd 
    import seaborn as sns 
    import os 
    # Convert nested dict → DataFrame with metrics as rows, strategies as columns 
    df = pd.DataFrame(data).T 
    df = df.T 
    if "ground_truth" in df.index: 
        df = df.drop("ground_truth") 
    # Apply label mapping if provided
    if label_map: df = df.rename(columns=label_map)

    # Row-wise normalization (scale each row independently to 0–1)
    normalized = df.copy()
    for row in df.index:
        row_min = df.loc[row].min()
        row_max = df.loc[row].max()
        if row_max > row_min:
            normalized.loc[row] = (df.loc[row] - row_min) / (row_max - row_min)
        else:
            normalized.loc[row] = 0  # constant row

    # Plot normalized heatmap but keep original values as annotations
    n_rows, n_cols = normalized.shape
    if extremes:
        fig_width = max(6, n_cols * 1.2)
        fig_height = max(8, n_rows * 0.8)
    else:
        fig_width = max(12, n_cols * 1.2)
        fig_height = max(6, n_rows * 0.6)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        normalized,
        annot=df,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False
    )
    plt.title("Heatmap of " + title + " (Row-normalized)")
    plt.ylabel("Metrics")
    plt.xlabel("Strategies")
    plt.xticks(rotation=45, ha="right")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_table(data, title="Table", output_path=None, extremes=False):

    import pandas as pd

    df = pd.DataFrame(data).T
    df = df.T

    if "ground_truth" in df.index:
        df = df.drop("ground_truth")

    # Apply label mapping if provided
    if label_map:
        df = df.rename(columns=label_map)
    
    if extremes:
        fig, ax = plt.subplots(figsize=(4, 6))
    else: 
        fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=np.round(df.values, 2), colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        plt.close()

def build_ontology(facts_dict, fill_missing=True, fill_value="unknown", check_consistency=True):

    ontology = {}

    for entity, feature, value in facts_dict.get("triplets", []):
        ontology.setdefault(entity, {})[feature] = value

    if fill_missing:
        entities = facts_dict.get("entities", [])
        features = facts_dict.get("features", [])
        for entity in entities:
            ontology.setdefault(entity, {})
            for feature in features:
                ontology[entity].setdefault(feature, fill_value)

    if check_consistency:
        ef_pairs = facts_dict.get("entities_features", set())
        fv_pairs = facts_dict.get("features_values", set())

        for e, f, v in facts_dict.get("triplets", []):
            if (e, f) not in ef_pairs:
                print(f"Inconsistency: ({e}, {f}) not in entities_features")
            if (f, v) not in fv_pairs:
                print(f"Inconsistency: ({f}, {v}) not in features_values")

    return ontology

def majority_ontology(ontologies):
    """Combine multiple ontologies by majority voting."""
    o_facts = {}
    for i, ontology in ontologies.items():
        o_facts[i] = factorize(ontology)

    facts = o_facts[0].keys()
    combined = {}
    for f in facts:
        majority = [v[f] for v in o_facts.values()]
        combined[f] = set.intersection(*majority) if majority else set()

    return combined

def main():

    # Load configuration
    config = load_config("config.yml")
    model_conf = config["model"]
    model_name = model_conf["name"]
    save_dir = config.get("save_dir", "results")
    INPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests", "align")
    PLOT_PATH = os.path.join(save_dir, model_name, "plots")
    # data = {strategy : {fact : { missing, correct, extra}}}
    avg, var, std_res = get_data(INPUT_PATH)

    missing_data = {strategy: {fact: values.get("missing", 0) for fact, values in metrics.items()} for strategy, metrics in avg.items()} # FN
    correct_data = {strategy: {fact: values.get("correct", 0) for fact, values in metrics.items()} for strategy, metrics in avg.items()} # TP
    extra_data = {strategy: {fact: values.get("extra", 0) for fact, values in metrics.items()} for strategy, metrics in avg.items()} # FP

    #Heatmap of averages
    #heatmap(correct_data, "Correct Elements", os.path.join(PLOT_PATH, f"heatmap_correct_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"))
    #heatmap(missing_data, "Missing Elements", os.path.join(PLOT_PATH, f"heatmap_missing_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"))
    #heatmap(extra_data, "Extra Elements", os.path.join(PLOT_PATH, f"heatmap_extra_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"))


    missing_std = {strategy: {fact: values.get("missing", 0)
                          for fact, values in metrics.items()}
               for strategy, metrics in std_res.items()}

    correct_std = {strategy: {fact: values.get("correct", 0)
                            for fact, values in metrics.items()}
                for strategy, metrics in std_res.items()}

    extra_std = {strategy: {fact: values.get("extra", 0)
                            for fact, values in metrics.items()}
                for strategy, metrics in std_res.items()}
    


    # Heatmap with std of correct alignments
    heatmap_with_std(correct_data, missing_std, "Correct Elements", os.path.join(PLOT_PATH, f"heatmap_std_correct_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"))
    heatmap_with_std(missing_data, correct_std, "Missing Elements", os.path.join(PLOT_PATH, f"heatmap_std_missing_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"))
    heatmap_with_std(extra_data, extra_std, "Extra Elements", os.path.join(PLOT_PATH, f"heatmap_std_extra_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"))

    # Compute precision, recall, F1
    precision_data = {}
    recall_data = {}
    f1_data = {}
    for strategy, metrics in avg.items():
        precision_data[strategy] = {}
        recall_data[strategy] = {}
        f1_data[strategy] = {}
        for fact, values in metrics.items():
            tp = values.get("correct", 0)
            fn = values.get("missing", 0)
            fp = values.get("extra", 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_data[strategy][fact] = precision * 100  # as percentage
            recall_data[strategy][fact] = recall * 100        # as percentage
            f1_data[strategy][fact] = f1 * 100                # as percentage


    #plot_table(precision_data, "Precision (%)", os.path.join(PLOT_PATH, f"precision_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"))
    #plot_table(recall_data, "Recall (%)", os.path.join(PLOT_PATH, f"recall_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"))
    #plot_table(f1_data, "F1 Score (%)", os.path.join(PLOT_PATH, f"f1_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"))

    print(f"[Main] Plots saved to {PLOT_PATH}")

def wings_main():
    from pprint import pprint
    # Load configuration
    config = load_config("config.yml")
    model_conf = config["model"]
    model_name = model_conf["name"]
    save_dir = config.get("save_dir", "results")
    INPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests", "align")
    PLOT_PATH = os.path.join(save_dir, model_name, "plots")
    # data = {strategy : {fact : { missing, correct, extra}}}
    data, var, std = get_data(INPUT_PATH, groups = wings_groups)

    missing_data = {}
    correct_data = {}
    extra_data = {}

    missing_std = {}
    correct_std = {}
    extra_std = {}

    for strategy, trials in wings_groups.items():
        # initialize nested dicts
        missing_data[strategy] = {}
        correct_data[strategy] = {}
        extra_data[strategy] = {}
        
        missing_std[strategy] = {}
        correct_std[strategy] = {}
        extra_std[strategy] = {}

        # get all fact keys from the first trial (e.g., 'entity', 'feature', etc.)
        facts = data[trials[0]].keys()

        for fact in facts:
            # skip the 'ground_truth' section
            if fact == "ground_truth":
                continue

            # collect values from all trials in the group
            missing_vals = [data[t][fact]["missing"] for t in trials]
            correct_vals = [data[t][fact]["correct"] for t in trials]
            extra_vals   = [data[t][fact]["extra"]   for t in trials]

            # compute averages
            missing_data[strategy][fact] = sum(missing_vals) / len(missing_vals)
            correct_data[strategy][fact] = sum(correct_vals) / len(correct_vals)
            extra_data[strategy][fact]   = sum(extra_vals) / len(extra_vals)

            # compute between-trial std (sample std, ddof=1)
            missing_std[strategy][fact] = np.std(missing_vals, ddof=1)
            correct_std[strategy][fact] = np.std(correct_vals, ddof=1)
            extra_std[strategy][fact]   = np.std(extra_vals, ddof=1)

    heatmap_with_std(correct_data, correct_std, "Correct Elements", os.path.join(PLOT_PATH, f"extremes_heatmap_std_correct_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"), extremes=True)
    heatmap_with_std(missing_data, missing_std, "Missing Elements", os.path.join(PLOT_PATH, f"extremes_heatmap_std_missing_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"), extremes=True)
    heatmap_with_std(extra_data, extra_std, "Extra Elements", os.path.join(PLOT_PATH, f"extremes_heatmap_std_extra_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"), extremes=True)

    # Heatmap of correct alignments
    #heatmap(correct_data, "Correct Elements", os.path.join(PLOT_PATH, f"extremes_heatmap_correct_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"), extremes=True)
    #heatmap(missing_data, "Missing Elements", os.path.join(PLOT_PATH, f"extremes_heatmap_missing_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"), extremes=True)
    #heatmap(extra_data, "Extra Elements", os.path.join(PLOT_PATH, f"extremes_heatmap_extra_{os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]}.pdf"), extremes=True)

    # Compute precision, recall, F1 
    precision_data = {}
    recall_data = {}
    f1_data = {}
    for strategy, metrics in data.items():
        precision_data[strategy] = {}
        recall_data[strategy] = {}
        f1_data[strategy] = {}
        for fact, values in metrics.items():
            tp = values.get("correct", 0)
            fn = values.get("missing", 0)
            fp = values.get("extra", 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_data[strategy][fact] = precision * 100  # as percentage
            recall_data[strategy][fact] = recall * 100        # as percentage
            f1_data[strategy][fact] = f1 * 100                # as percentage

    # Average precision, recall, F1 across trials in each group
    P = {}
    R = {}
    F1 = {}
    for strategy, trials in wings_groups.items():
        P[strategy] = {}
        R[strategy] = {}
        F1[strategy] = {}
        for fact in precision_data[trials[0]].keys():
            if fact == "ground_truth":
                continue
            p_vals = [precision_data[t][fact] for t in trials]
            r_vals = [recall_data[t][fact] for t in trials]
            f1_vals = [f1_data[t][fact] for t in trials]

            P[strategy][fact] = sum(p_vals) / len(p_vals)
            R[strategy][fact] = sum(r_vals) / len(r_vals)
            F1[strategy][fact] = sum(f1_vals) / len(f1_vals)
    
    #plot_table(P, "Precision (%)", os.path.join(PLOT_PATH, f"extremes_precision_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"), extremes=True)
    #plot_table(R, "Recall (%)", os.path.join(PLOT_PATH, f"extremes_recall_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"), extremes=True)
    #plot_table(F1, "F1 Score (%)", os.path.join(PLOT_PATH, f"extremes_f1_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf"), extremes=True)

    print(f"[Main] Plots saved to {PLOT_PATH}")

def corr_test():
    import scipy.stats as stats
    # Load configuration
    config = load_config("config.yml")
    model_conf = config["model"]
    model_name = model_conf["name"]
    save_dir = config.get("save_dir", "results")
    INPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests", "align")
    PLOT_PATH = os.path.join(save_dir, model_name, "plots")
    # data = {strategy : {fact : { missing, correct, extra}}}
    avg, _, _ = get_data(INPUT_PATH)
    missing_data = {strategy: {fact: values.get("missing", 0) for fact, values in metrics.items()} for strategy, metrics in avg.items()} # FN
    correct_data = {strategy: {fact: values.get("correct", 0) for fact, values in metrics.items()} for strategy, metrics in avg.items()} # TP
    extra_data = {strategy: {fact: values.get("extra", 0) for fact, values in metrics.items()} for strategy, metrics in avg.items()} # FP

    # Performance as global precision
    global_precision = {
    strategy: (
        sum(correct_data[strategy].values()) /
        (sum(correct_data[strategy].values()) + sum(extra_data[strategy].values()))
        if (sum(correct_data[strategy].values()) + sum(extra_data[strategy].values())) > 0 else 0
    )
    for strategy in avg.keys() if strategy not in ["expert", "expert_with_text"]
    } 

    print("Global Precision by Strategy:", global_precision)

    # Time of learning plateau
    
    INPUT_PATH_TRACK = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests", "track_info")
    learning_plateau = {}
    max_info = {}
    for strategy in global_precision.keys():
        if strategy in ["expert", "expert_with_text"]:
            continue  # skip expert strategy
        file_path = os.path.join(INPUT_PATH_TRACK, f"tracking_{strategy}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")
        data = json.load(open(file_path, "r", encoding="utf-8"))
        tkf = data["metrics"]["total_known_facts"]
        plateau = tkf.index(max(tkf))  # first occurrence of max known facts
        max_exchange = max(tkf)
        max_info[strategy] = max_exchange
        learning_plateau[strategy] = plateau

    print("Learning Plateau Steps:", learning_plateau)

    # Align strategies and extract paired lists
    strategies = global_precision.keys() & learning_plateau.keys()
    precisions = [global_precision[s] for s in strategies]
    plateaus = [learning_plateau[s] for s in strategies]
    maxs = [max_info[s] for s in strategies]

    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(plateaus, precisions)
    print(f"Pearson r = {pearson_corr:.3f} (p = {pearson_p:.3f})")

    # Spearman rank correlation (robust to nonlinearity)
    spearman_corr, spearman_p = stats.spearmanr(plateaus, precisions)
    print(f"Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3f})")

    # Correlation with information exchanged
    pearson_corr_info, pearson_p_info = stats.pearsonr(maxs, precisions)
    print(f"Pearson r (info) = {pearson_corr_info:.3f} (p = {pearson_p_info:.3f})")
    spearman_corr_info, spearman_p_info = stats.spearmanr(maxs, precisions)
    print(f"Spearman ρ (info) = {spearman_corr_info:.3f} (p = {spearman_p_info:.3f})")

    # 20 questions games
    avg_20q  = {
    "teacher_questions": 5.3,
    "top_down": 5.55,
    "mixed_learner_questions": 5.6,
    "mixed_top-down_learner_questions": 5.9,
    "mixed_bottom-up_learner_questions": 6.2,
    "bottom_up": 6.5,
    "learner_questions": 7.05,
    "mixed_top-down_teacher_questions": 6.7,
    "mixed_teacher_questions": 7.7,
    "mixed_bottom-up_teacher_questions": 9.35}

    games_20q = {s: avg_20q[s] for s in strategies if s in avg_20q}
    games = [games_20q[s] for s in strategies if s in avg_20q]
    pearson_corr_20q, pearson_p_20q = stats.pearsonr(games, maxs)
    print(f"Pearson r (20q vs info) = {pearson_corr_20q:.3f} (p = {pearson_p_20q:.3f})")
    spearman_corr_20q, spearman_p_20q = stats.spearmanr(games, maxs)
    print(f"Spearman ρ (20q vs info) = {spearman_corr_20q:.3f} (p = {spearman_p_20q:.3f})")
    # 20q vs plateau
    pearson_corr_20q_p, pearson_p_20q_p = stats.pearsonr(games, plateaus)
    print(f"Pearson r (20q vs plateau) = {pearson_corr_20q_p:.3f} (p = {pearson_p_20q_p:.3f})")
    spearman_corr_20q_p, spearman_p_20q_p = stats.spearmanr(games, plateaus)
    print(f"Spearman ρ (20q vs plateau) = {spearman_corr_20q_p:.3f} (p = {spearman_p_20q_p:.3f})")
    # 20q vs precision
    pearson_corr_20q_pr, pearson_p_20q_pr = stats.pearsonr(games, precisions)
    print(f"Pearson r (20q vs precision) = {pearson_corr_20q_pr:.3f} (p = {pearson_p_20q_pr:.3f})")
    spearman_corr_20q_pr, spearman_p_20q_pr = stats.spearmanr(games, precisions)
    print(f"Spearman ρ (20q vs precision) = {spearman_corr_20q_pr:.3f} (p = {spearman_p_20q_pr:.3f})")  
    


        

if __name__ == "__main__":
    #main()
    #wings_main()
    corr_test()


