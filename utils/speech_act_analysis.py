
import os
from collections import Counter
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.api_interface import call_model_api
import json
import pandas as pd
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

groups = {
        "Instructions": ["top_down", "bottom_up"],
        "Questions": ["learner_questions", "teacher_questions"],
        "Dialogic TD": ["mixed_top-down_learner_questions", "mixed_top-down_teacher_questions"],
        "Dialogic BU": ["mixed_bottom-up_learner_questions", "mixed_bottom-up_teacher_questions"],
        "Dialogic Qs": ["mixed_learner_questions", "mixed_teacher_questions"]
    }

def load_trials(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Classifier over Dialogue (helper) ---
def classifier_dialogue(dialogue, step, model, num_samples = 5, context_aware=False, ontology_json=""):

    if not context_aware:
        system_prompt = """
                        You are an expert linguist specializing in Speech Act Theory and Educational Dialogue Analysis. 
                        Your task is to classify the function of a specific turn in a dialogue.

                        You will receive the dialogue history and the current turn.
                        1. First, analyze the linguistic structure and intent of the turn inside your thought process.
                        2. Then, output a single Valid JSON object containing the classification tag.
                        3. Do NOT output Markdown formatting (like ```json). Output the label string only.
                        """
        user_prompt = f"""
                        ### Task
                        Classify the **Current Turn** into exactly one of the following tags based on its primary communicative function.

                        ### The Rubric
                        1. **Informing**: Conveying facts, answers, explanations, or evaluative feedback (e.g., "The answer is 5", "That is correct").
                        2. **Inquiring**: Asking questions or probing for information (e.g., "Why?", "Is it X?").
                        3. **Acknowledging**: Confirming receipt or simple agreement without adding new content (e.g., "Okay", "Right", "I see").
                        4. **Phatic**: Social pleasantries or channel management (e.g., "Hello", "Can you hear me?").

                        ### The Input
                        **Conversation History:**
                        {dialogue}

                        **Current Turn to Classify:**
                        {step}

                        ### Output Format
                        Output ONLY one of the following tags as a string: "Informing", "Inquiring", "Acknowledging", "Phatic", or "Not Related" if none apply.
                        """
    else:
        system_prompt = f"""
                        You are an expert linguist specializing in Educational Dialogue Analysis. 
                        Your task is to classify the function of a specific turn in a dialogue relative to a strict **Knowledge Domain**.

                        ### The Knowledge Domain (Ground Truth)
                        {ontology_json}

                        ### Instructions
                        1. Analyze the linguistic structure of the turn.
                        2. **Relevance Check:** Compare the content of the turn against the JSON Ontology above.
                        - If the user discusses these specific species and features, it is RELATED.
                        - If the user discusses unrelated topics, it is OFF-TOPIC.
                        3. Output a single Valid JSON object containing the classification tag.
                        4. Do NOT output Markdown. Output the label string only.
                        """   
        user_prompt = f"""
                        ### Task
                        Classify the **Current Turn** as either "Related" or "Off-Topic".

                        ### The Rubric
                        1. **Related**: 
                        - The turn is about the Alien Species, their traits, or the specific ontology structure provided.
                        - *Reasoning:* If the turn keeps the educational conversation going smoothly within the context of the alien world, it is Related.

                        2. **Off-Topic**: 
                        - The turn introduces concepts **outside the scope** of the provided alien universe.
                        - *Reasoning:* If the turn shifts focus to **tools** (Protégé, Git), **workflows**, or **unrelated domains**, it is Off-Topic.
                        - *Reasoning:* If the turn does not contain any content relevant to the alien species or ontology, it is Off-Topic.
                        - *Reasoning:* If the turn does not contain information about the alien species or their traits, it is Off-Topic.


                        ### The Input
                        **Conversation History:**
                        {dialogue}

                        **Current Turn to Classify:**
                        {step}

                        ### Output Format
                        Output ONLY one of the following tags as a string: "Related" or "Off-Topic".
                        """
   
    message = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    results = []

    for _ in range(num_samples):
        response = call_model_api(message, model)
        print(f"[Classifier] Model response: {response}")
        # Look for which tag appears in the LLM response
        found_tag = None
        for tag in [
            "Related",
            "Off-Topic"
        ]:
        
            if tag in response:
                found_tag = tag
                break  # stop at the first match

        # Default to "Not Related" if nothing matches
        results.append(found_tag or "No label found")

    response = Counter(results).most_common(1)[0][0]
    #print(f"Most common response: {response}, {type(response)}")
    return response

# --- Start Classifier over Dialogue (helper) ---
def start_classifier(dialogue, model_name, ontology_json):

    if not dialogue:
        raise ValueError("Dialogue cannot be empty or None")

    res = {}
    for i, step in enumerate(dialogue):
        print(f"[Classifier] Processing dialogue turn {i+1}/{len(dialogue)}")
        
        classification = classifier_dialogue(
            dialogue=dialogue[:i],
            step=step,
            model="gpt-oss-large",
            num_samples=5,
            context_aware=True,
            ontology_json=ontology_json
        )

        print(f"[Classifier] Classification for turn {i+1}: {classification}")
        res[i] = classification
    return res

# --- Main Function for Classifier ---
def main_classifier():
    from config_loader import load_config

    config = load_config("config.yml")
    #ontology_file = config["ontology"]["file"]
    save_dir = config.get("save_dir", "results")
    model_name = config["model"]["name"]
    strategy = config["Teacher"]["strategy"]
    ontology_file = config["ontology"]["file"]
    with open(ontology_file, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
        print("[Main] Loaded ontology from ", ontology_file)
    OUTPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests")

    TRAINING_LOGS_DIR = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "dialogue")
    print(f"[Main] Output path set to: {OUTPUT_PATH}")
    print(f"[Main] Training logs path set to: {TRAINING_LOGS_DIR}")
    os.makedirs(os.path.join(OUTPUT_PATH, "track_info"), exist_ok=True)

    res = {}
    for _, agent_keys in groups.items():
        for agent_key in agent_keys:
            #if agent_key == "bottom_up" or agent_key == "top_down":
            #    continue  # Skip these strategies
            print(f"[Main] Processing strategy: {agent_key}")
            strategy = agent_key
            filename = os.path.join(TRAINING_LOGS_DIR, f"{strategy}_train.json")
            if os.path.isfile(filename) and filename.endswith("_train.json"):
                print(f"[Main] Processing file: {filename}")
                conversation = load_trials(filename)[0]["conversation"]
                print(f"[Main] Converted conversation with {len(conversation)} turns.")
                print(f"[Main] Starting tracking for {filename}")
                #if agent_key == "bottom_up" or agent_key == "top_down":
                #    res[strategy] = start_classifier(conversation, model_name, ontology_json=ontology)
                #else:
                #    continue
                res[strategy] = start_classifier(conversation, model_name, ontology_json=ontology)
                print(f"[Main] Saving cumulative results...")
                with open(os.path.join(OUTPUT_PATH, "track_info", f"binary_classifier_results.json"), 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=4)

# --- Helper Functions ---
def split_into_sentences(text):
    # Splits by punctuation (. ! ?) followed by whitespace, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# --- Core Metric Calculation ---
def compute_density_for_strategy(strategy_name, dialogue, extracted_data, is_monolithic=False):
    """
    Computes Semantic Density. 
    If is_monolithic=True, splits the first dialogue turn into sentences first.
    """
    # 1. Build Global Vocabulary (The "Signal")
    vocab = set()
    facts = extracted_data.get('facts', {})
    
    def add_to_vocab(source_list):
        if not source_list: return
        # Check the final state [-1] for cumulative knowledge
        final_state = source_list[-1] 
        for item in final_state:
            for token in re.split(r'[\s-]+', str(item).lower()):
                if len(token) > 2: vocab.add(token)

    add_to_vocab(facts.get('entities', []))
    add_to_vocab(facts.get('features', []))
    add_to_vocab(facts.get('values', []))
    
    # 2. Prepare Units of Analysis (Turns vs Sentences)
    if is_monolithic:
        # Assuming the monolithic text is in the first (and only) turn
        # If it's a list of length 1, take index 0.
        full_text = dialogue[0] if isinstance(dialogue, list) and len(dialogue) > 0 else str(dialogue)
        analysis_units = split_into_sentences(full_text)
        # print(f"   -> Split monolithic '{strategy_name}' into {len(analysis_units)} sentences.")
    else:
        # Interactive: Filter for Active Phase Only
        raw_units = dialogue
        
        # Check for fact history to determine saturation
        fact_history = extracted_data.get('metrics', {}).get('total_known_facts', [])
        
        if fact_history and len(fact_history) > 0:
            # Convert to numpy for threshold detection
            F = np.array(fact_history)
            final_val = F[-1]
            
            if final_val > 0:
                # Saturation Threshold: 100% of final value
                threshold = 1 * final_val
                t_sat = np.argmax(F >= threshold)
                
                # Slice the dialogue: Keep turns 0 to t_sat
                analysis_units = raw_units[:t_sat + 1]
                # print(f"   -> Sliced '{strategy_name}' to {len(analysis_units)} turns (Saturation idx: {t_sat})")
            else:
                analysis_units = raw_units
        else:
            analysis_units = raw_units

    # 3. Compute Density per Unit
    records = []
    for i, unit_text in enumerate(analysis_units):
        tokens = re.findall(r'\b\w+\b', unit_text.lower())
        n_tokens = len(tokens)
        
        if n_tokens == 0:
            density = 0.0
        else:
            matches = sum(1 for t in tokens if t in vocab)
            density = matches / n_tokens
            
        records.append({
            "Strategy": strategy_name,
            "Turn": i,
            "Density": density
        })
        
    return records

# --- Main Function ---
def main_semantic_density(groups = groups):
    # Load Config
    from config_loader import load_config
    config = load_config("config.yml")
    save_dir = config.get("save_dir", "results")
    model_name = config["model"]["name"]
    ontology_file = config["ontology"]["file"]
    
    # Paths
    OUTPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(ontology_file))[0], "tests")
    TRACKING_DIR = os.path.join(OUTPUT_PATH, "track_info")
    TRAINING_LOGS_DIR = os.path.join(save_dir, "llama3", os.path.splitext(os.path.basename(ontology_file))[0], "dialogue")

    print(f"[Main] Output path: {OUTPUT_PATH}")
    
    # --- 1. Define Strategy Order (Baselines Top, Others Bottom) ---
    baselines = ["top_down", "bottom_up"]
    interactive_strategies = []
    
    # Flatten groups but exclude baselines (we add them manually to top)
    for group_name, strategies in groups.items():
        for s in strategies:
            if s not in baselines:
                interactive_strategies.append(s)
    
    # The final display order: Baselines -> Separator -> Interactive
    # We use "---" as a dummy strategy name for the separator
    display_order = baselines + ["---"] + interactive_strategies
    
    all_density_records = []

    # --- 2. Process All Strategies (Including Monolithic) ---
    # We create a temporary list of ALL strategies to loop through
    strategies_to_process = baselines + interactive_strategies

    for agent_key in strategies_to_process:
        print(f"[Main] Processing: {agent_key}")
        
        dialogue_path = os.path.join(TRAINING_LOGS_DIR, f"{agent_key}_train.json")
        tracking_path = os.path.join(TRACKING_DIR, f"tracking_{agent_key}.json")
        
        if os.path.isfile(dialogue_path) and os.path.isfile(tracking_path):
            try:
                # Load Data
                with open(dialogue_path, 'r', encoding='utf-8') as f:
                    d_data = json.load(f)
                    conversation = d_data[0]["conversation"] if isinstance(d_data, list) else d_data["conversation"]

                with open(tracking_path, 'r', encoding='utf-8') as f:
                    extracted_data = json.load(f)

                # Check if Monolithic
                is_mono = (agent_key in baselines)

                # Compute
                records = compute_density_for_strategy(agent_key, conversation, extracted_data, is_monolithic=is_mono)
                all_density_records.extend(records)

            except Exception as e:
                print(f"[Error] {agent_key}: {e}")
        else:
            print(f"[Warning] Files not found for {agent_key}")

    # --- 3. Visualization ---
    if all_density_records:
        df = pd.DataFrame(all_density_records)
        
        # Save raw data
        df.to_csv(os.path.join(TRACKING_DIR, "semantic_density_results_learning_phase.csv"), index=False)

        # ==========================================
        # PRINT AVERAGE DENSITY PER STRATEGY
        # ==========================================
        print("\n" + "="*40)
        print("AVERAGE SEMANTIC DENSITY BY STRATEGY")
        print("="*40)
        
        # Calculate mean density per strategy
        avg_densities = df.groupby("Strategy")["Density"].mean()
        
        # Reorder to match display order (excluding the separator)
        for strategy in baselines + interactive_strategies:
            if strategy in avg_densities.index:
                val = avg_densities[strategy]
                print(f"{strategy:<35} | {val:.4f}")
        print("="*40 + "\n")

        # Pivot
        heatmap_data = df.pivot(index="Strategy", columns="Turn", values="Density")
        
        # Add the Separator Row (NaNs)
        # We perform a reindex to insert the "---" row which doesn't exist in data, creating NaNs
        heatmap_data = heatmap_data.reindex(display_order)
        
        # Plot
        plt.figure(figsize=(15, 8)) # Taller to accommodate more rows
        
        # Use a mask for the separator to color it specifically (optional, or just let NaNs be white)
        ax = sns.heatmap(
            heatmap_data, 
            cmap="viridis", 
            linewidths=0.5, 
            linecolor='white',
            cbar_kws={'label': 'Semantic Density (Ontology Terms / Total Words)'},
            vmin=0, vmax=0.6,
            # Ensure NaNs (our separator and empty turns) are plotted as White
            mask=heatmap_data.isnull() 
        )
        
        # Set background to white/gray for NaNs (The separator)
        ax.set_facecolor('#f0f0f0') 

        plt.title("Semantic Density Profile", fontsize=16)
        plt.xlabel("Sequence Step (Sentence or Turn)", fontsize=12)
        plt.ylabel("Strategy", fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(TRACKING_DIR, "semantic_density_heatmap_learning_phase.png")
        plt.savefig(plot_path)
        print(f"[Main] Plot saved to {plot_path}")
        plt.show()
    else:
        print("[Main] No data.")

# --- Plotting Functions for Speech Act Analysis ---
def main_plotter(groups = groups):
    # Load classified data (Adjust path as needed)
    filename = "results/gpt-4o/ontology_aliens_10/tests/track_info/aware_classifier_results.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return

    # Flatten to a single list based on your 'groups' dictionary
    ordered_strategies = []
    for group_name, strategies in groups.items():
        ordered_strategies.extend(strategies)

    # ==========================================
    # 1. DEFINE ROBUST COLOR PALETTE
    # ==========================================
    # We define specific colors for ALL possible tags
    palette_map = {
        "Informing": "#3498db",      # Blue
        "Inquiring": "#e67e22",      # Orange
        "Acknowledging": "#2ecc71",  # Green
        "Phatic": "#95a5a6",         # Grey
        "Off-Topic": "#e74c3c",      # Red
        "No label found": "#34495e"  # Dark Blue/Black
    }
    
    # Define the strict order of tags for the legend and integers
    tag_order = ["Informing", "Inquiring", "Acknowledging", "Phatic", "Off-Topic", "No label found"]
    
    # Create the list of colors in the exact order of the tags (0 to 5)
    ordered_colors = [palette_map[tag] for tag in tag_order]

    # ==========================================
    # 2. PROCESS DATA FRAME
    # ==========================================
    records = []
    for strategy, turns in data.items():
        for turn_id, tag in turns.items():
            records.append({"Strategy": strategy, "Turn": int(turn_id), "Tag": tag})
    
    df = pd.DataFrame(records)

    # ==========================================
    # 3. PLOT 1: DNA HEATMAP
    # ==========================================
    plt.figure(figsize=(15, 6))

    # Prepare pivot table
    heatmap_data = df.pivot(index="Strategy", columns="Turn", values="Tag")
    
    # Reindex rows to enforce strategy order
    heatmap_data = heatmap_data.reindex(ordered_strategies)

    # Map tags to integers (0-5)
    tag_to_int = {tag: i for i, tag in enumerate(tag_order)}
    numeric_data = heatmap_data.replace(tag_to_int)

    # Create Discrete Colormap from our ordered list
    cmap = mcolors.ListedColormap(ordered_colors)
    
    # Determine bounds for the colorbar (0, 1, 2, 3, 4, 5, 6)
    bounds = range(len(tag_order) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Draw Heatmap
    ax1 = sns.heatmap(
        numeric_data, 
        cmap=cmap, 
        norm=norm,
        cbar=False, 
        linewidths=0.5, 
        linecolor='white',
        yticklabels=True, 
        xticklabels=True
    )

    plt.title("Dialogue DNA: Speech Acts Sequence", fontsize=16)
    plt.xlabel("Turn Number", fontsize=12)
    plt.ylabel("Strategy", fontsize=12)

    # Create Custom Legend
    patches = [mpatches.Patch(color=palette_map[tag], label=tag) for tag in tag_order]
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', title="Speech Act")

    plt.tight_layout()
    plt.savefig("aw_dna_heatmap.png")
    plt.show()

    # ==========================================
    # 4. PLOT 2: STACKED BAR DISTRIBUTION
    # ==========================================
    # Calculate counts
    df_counts = df.groupby(["Strategy", "Tag"], observed=False).size().unstack(fill_value=0)
    
    # CRITICAL FIX: Ensure ALL columns exist, even if count is 0
    df_counts = df_counts.reindex(columns=tag_order, fill_value=0)
    
    # Calculate Ratios
    df_ratios = df_counts.div(df_counts.sum(axis=1), axis=0) * 100

    # Reorder Rows (Reverse for Barh to match Heatmap top-down visual)
    df_ratios_ordered = df_ratios.reindex(ordered_strategies[::-1])

    # Plot
    ax2 = df_ratios_ordered.plot(
        kind='barh', 
        stacked=True, 
        color=ordered_colors, # Uses the exact same color list as heatmap
        edgecolor='black',
        linewidth=0.5,
        width=0.8,
        figsize=(15, 7)
    )

    plt.title("Speech Act Distribution by Strategy", fontsize=16)
    plt.xlabel("Percentage of Turns", fontsize=12)
    plt.ylabel("Strategy", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Speech Act")

    # Add Percentage Labels
    for c in ax2.containers:
        # Only label segments > 5% width for readability
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 5 else '' for v in c]
        ax2.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig("aw_distribution_plot.png")
    plt.show()

def main_plotter_binary(groups = groups):
    # Load classified data (Adjust path as needed)
    filename = "results/llama3/ontology_aliens_10/tests/track_info/binary_classifier_results.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return

    # Flatten to a single list based on your 'groups' dictionary
    ordered_strategies = []
    for group_name, strategies in groups.items():
        ordered_strategies.extend(strategies)

    # ==========================================
    # 1. DEFINE BINARY COLOR PALETTE
    # ==========================================
    palette_map = {
        "Related": "#EBD8D8",   # Green (Good/Safe)
        "Off-Topic": "#161514", # Red (Drift/Alert)
    }
    
    # Define the strict order of tags
    tag_order = ["Related", "Off-Topic"]
    
    # Create the list of colors in the exact order of the tags
    ordered_colors = [palette_map[tag] for tag in tag_order]

    # ==========================================
    # 2. PROCESS DATA FRAME
    # ==========================================
    records = []
    for strategy, turns in data.items():
        for turn_id, tag in turns.items():
            # Safety check: Ensure tag matches our binary expectation
            if tag not in tag_order:
                # Optional: Map old tags if mixing data versions, or skip
                continue 
            records.append({"Strategy": strategy, "Turn": int(turn_id), "Tag": tag})
    
    df = pd.DataFrame(records)

    if df.empty:
        print("No valid 'Related' or 'Off-Topic' tags found in data.")
        return

    # ==========================================
    # 3. PLOT 1: DNA HEATMAP (Binary Strip)
    # ==========================================
    plt.figure(figsize=(15, 6))

    # Prepare pivot table
    heatmap_data = df.pivot(index="Strategy", columns="Turn", values="Tag")
    heatmap_data = heatmap_data.reindex(ordered_strategies)

    # Map tags to integers (0=Related, 1=Off-Topic)
    tag_to_int = {tag: i for i, tag in enumerate(tag_order)}
    numeric_data = heatmap_data.replace(tag_to_int)

    # Create Discrete Colormap
    cmap = mcolors.ListedColormap(ordered_colors)
    bounds = range(len(tag_order) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Draw Heatmap
    ax1 = sns.heatmap(
        numeric_data, 
        cmap=cmap, 
        norm=norm,
        cbar=False, 
        linewidths=0.5, 
        linecolor='white',
        yticklabels=True, 
        xticklabels=True
    )

    plt.title("Dialogue Relevance: Semantic Drift Analysis", fontsize=16)
    plt.xlabel("Turn Number", fontsize=12)
    plt.ylabel("Strategy", fontsize=12)

    # Custom Legend
    patches = [mpatches.Patch(color=palette_map[tag], label=tag) for tag in tag_order]
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', title="Relevance")

    plt.tight_layout()
    plt.savefig("binary_relevance_dna_heatmap_llama3.png")
    plt.show()

    # ==========================================
    # 4. PLOT 2: STACKED BAR DISTRIBUTION
    # ==========================================
    # Calculate counts
    df_counts = df.groupby(["Strategy", "Tag"], observed=False).size().unstack(fill_value=0)
    
    # Ensure both columns exist
    df_counts = df_counts.reindex(columns=tag_order, fill_value=0)
    
    # Calculate Ratios
    df_ratios = df_counts.div(df_counts.sum(axis=1), axis=0) * 100

    # Reorder Rows (Reverse for Barh)
    df_ratios_ordered = df_ratios.reindex(ordered_strategies[::-1])

    # Plot
    ax2 = df_ratios_ordered.plot(
        kind='barh', 
        stacked=True, 
        color=ordered_colors, 
        edgecolor='black',
        linewidth=0.5,
        width=0.8,
        figsize=(15, 7)
    )

    plt.title("Topic Adherence by Strategy", fontsize=16)
    plt.xlabel("Percentage of Turns", fontsize=12)
    plt.ylabel("Strategy", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Relevance")

    # Add Labels
    for c in ax2.containers:
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 5 else '' for v in c]
        ax2.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig("binary_relevance_distribution_plot_llama3.png")
    plt.show()

if __name__ == "__main__":
    #main_classifier()
    #main_plotter_binary()
    main_semantic_density()

    """
    ========================================
    AVERAGE SEMANTIC DENSITY BY STRATEGY
    ========================================
    top_down                            | 0.3125
    bottom_up                           | 0.0376
    learner_questions                   | 0.0243
    teacher_questions                   | 0.0868
    mixed_top-down_learner_questions    | 0.0229
    mixed_top-down_teacher_questions    | 0.0616
    mixed_bottom-up_learner_questions   | 0.1243
    mixed_bottom-up_teacher_questions   | 0.1434
    mixed_learner_questions             | 0.0950
    mixed_teacher_questions             | 0.0370
    ========================================
    """

    """
    ========================================
    AVERAGE SEMANTIC DENSITY BY STRATEGY
    ========================================
    top_down                            | 0.3125
    bottom_up                           | 0.0376
    learner_questions                   | 0.0607
    teacher_questions                   | 0.0875
    mixed_top-down_learner_questions    | 0.0858
    mixed_top-down_teacher_questions    | 0.0632
    mixed_bottom-up_learner_questions   | 0.1230
    mixed_bottom-up_teacher_questions   | 0.2295
    mixed_learner_questions             | 0.0938
    mixed_teacher_questions             | 0.0381
    ========================================
    """

    """
    LLAMA3 Results :
    ========================================
    AVERAGE SEMANTIC DENSITY BY STRATEGY
    ========================================
    top_down                            | 0.2011
    bottom_up                           | 0.1324
    learner_questions                   | 0.0956
    teacher_questions                   | 0.0604
    mixed_top-down_learner_questions    | 0.0854
    mixed_top-down_teacher_questions    | 0.0103
    mixed_bottom-up_learner_questions   | 0.1078
    mixed_bottom-up_teacher_questions   | 0.1486
    mixed_learner_questions             | 0.0678
    mixed_teacher_questions             | 0.0945
    ========================================
    """