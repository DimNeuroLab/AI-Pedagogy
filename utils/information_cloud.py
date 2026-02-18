import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config_loader import load_config
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import re
import plotly.graph_objects as go
from scipy.spatial import distance


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

def plot_ontology_cloud(ontology_data, color_by_attribute=True):
    """
    Embeds raw triplets and plots them as a 2D semantic cloud.
    
    Args:
        ontology_data (dict): The dictionary containing species and their props.
        color_by_attribute (bool): If True, colors points by category (Diet, Habitat). 
                                   If False, uses a single uniform color.
    """
    # 1. Construct Raw Triplets
    triplets = []
    labels = []
    categories = []
    
    for species, props in ontology_data.items():
        for category, value in props.items():
            # STRICT TRIPLET FORMAT
            triplet_str = f"{species}, {category}, {value}"
            
            triplets.append(triplet_str)
            labels.append(f"{species}\n({value})") 
            categories.append(category)

    # 2. Embeddings
    print("Embedding triplets...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    embeddings = model.encode(triplets)
    
    # 3. Dimensionality Reduction
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate=200)
    coords = tsne.fit_transform(embeddings)
    
    # 4. Prepare Data
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "category": categories
    })
    
    # 5. Visualization Setup
    plt.figure(figsize=(14, 10))
    
    # Configure arguments based on the flag
    if color_by_attribute:
        scatter_kwargs = {
            "hue": "category",
            "style": "category",
            "palette": "bright"
        }
        title_suffix = "(Colored by Attribute)"
    else:
        scatter_kwargs = {
            "color": "#3498db" # Uniform Blue
        }
        title_suffix = "(Uniform Color)"

    # Plot the cloud
    sns.scatterplot(
        data=df, 
        x="x", y="y", 
        s=150, 
        edgecolor="black",
        alpha=0.8,
        **scatter_kwargs
    )
    
    # Add text labels
    for i in range(df.shape[0]):
        plt.text(
            df.x[i]+0.3, 
            df.y[i]+0.3, 
            df.label[i], 
            fontsize=8, 
            alpha=0.7
        )

    plt.title(f"Ontology Space: Raw Triplet Embeddings {title_suffix}", fontsize=16)
    plt.xlabel("Semantic Dimension 1")
    plt.ylabel("Semantic Dimension 2")
    
    # Only show legend if we are actually distinguishing categories
    if color_by_attribute:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Relation Type")
        
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("ontology_triplet_cloud.png", dpi=300)
    plt.show()

def plot_trajectory(ontology_data, conversation_log, strategy_name="Unknown Strategy"):
    # --- STEP A: Prepare Ontology Triplets ---
    ontology_sentences = []
    ontology_labels = []
    
    for species, props in ontology_data.items():
        for category, value in props.items():
            # STRICT TRIPLET: "Subject, Relation, Object"
            ontology_sentences.append(f"{species}, {category}, {value}")
            ontology_labels.append(f"{species}-{value}")

    # --- STEP B: Prepare Dialogue Turns ---
    dialogue_sentences = []
    dialogue_roles = []
    turn_indices = []
    
    for i, turn in enumerate(conversation_log):
        # Clean the turn (remove "Speaker: ") for better embedding
        if ":" in turn:
            role, content = turn.split(":", 1)
            content = content.strip()
        else:
            role = "Unknown"
            content = turn
            
        dialogue_sentences.append(content)
        dialogue_roles.append(role.strip())
        turn_indices.append(i)

    # --- STEP C: Unified Embedding ---
    # We must embed AND project them together to share the same space
    combined_text = ontology_sentences + dialogue_sentences
    
    print(f"Embedding {len(combined_text)} items (Ontology + Dialogue)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(combined_text)
    
    # --- STEP D: Dimensionality Reduction ---
    # Using t-SNE to map high-dimensional meaning to 2D
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate=200)
    coords = tsne.fit_transform(embeddings)
    
    # --- STEP E: Separate Data back into Groups ---
    n_ontology = len(ontology_sentences)
    
    ont_x = coords[:n_ontology, 0]
    ont_y = coords[:n_ontology, 1]
    
    diag_x = coords[n_ontology:, 0]
    diag_y = coords[n_ontology:, 1]

    # --- STEP F: Visualization ---
    plt.figure(figsize=(12, 8))
    
    # 1. Plot Ontology Cloud (Background)
    plt.scatter(ont_x, ont_y, c='lightgrey', s=100, alpha=0.5, label='Ontology Facts')
    # Optional: Label a few ontology points contextually if needed, but keeping it clean for now
    
    # 2. Plot Dialogue Path (The Line)
    plt.plot(diag_x, diag_y, c='black', alpha=0.3, linestyle='--', linewidth=1)
    
    # 3. Plot Dialogue Turns (The Points)
    # Color code by Speaker
    colors = ['blue' if r == 'Teacher' else 'red' for r in dialogue_roles]
    
    scatter = plt.scatter(diag_x, diag_y, c=colors, s=150, edgecolors='black', zorder=10)
    
    # 4. Annotate the Turns (Numbering)
    for i, txt in enumerate(turn_indices):
        plt.annotate(txt, (diag_x[i], diag_y[i]), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    # Create Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', markersize=10),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
    
    plt.legend(custom_lines, ['Ontology Fact', 'Teacher Turn', 'Learner Turn'], loc='upper left')

    plt.title(f"Conversational Trajectory: {strategy_name}", fontsize=16)
    plt.xlabel("Semantic Dimension 1")
    plt.ylabel("Semantic Dimension 2")
    plt.tight_layout()
    plt.savefig(f"trajectory_{strategy_name}.png", dpi=300)
    plt.show()

def plot_trajectory_umap(ontology_data, conversation_log, strategy_name="Unknown Strategy"):
    
    # --- STEP A: Prepare Ontology (Reference Cloud) ---
    ontology_sentences = []
    for species, props in ontology_data.items():
        for category, value in props.items():
            ontology_sentences.append(f"{species}, {category}, {value}")

    # --- STEP B: Prepare Dialogue Data (Conditional Logic) ---
    dialogue_sentences = []
    dialogue_roles = []
    turn_indices = []

    # Check if we need to "explode" the turns into sentences
    should_chunk = strategy_name in ["top_down", "bottom_up"]
    if should_chunk:
        print(f"Strategy '{strategy_name}' detected as Dense. Applying Sentence Chunking...")
        for i, turn in enumerate(conversation_log):
            if ":" in turn:
                role, content = turn.split(":", 1)
                content = content.strip()
            else:
                role = "Unknown"
                content = turn
            
            # Chunk Teacher turns if they are long (> 30 words)
            word_count = len(content.split())
            if role == "Teacher" and word_count > 30:
                # Split by sentence delimiters (., !, ?) 
                chunks = re.split(r'(?<=[.!?]) +', content)
                for j, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 5: # Filter out noise
                        dialogue_sentences.append(chunk)
                        dialogue_roles.append(role)
                        # Use decimals to show sub-steps (Turn 0.1, 0.2...)
                        turn_indices.append(f"{i}.{j+1}")
            else:
                # Keep short turns or Learner turns whole
                dialogue_sentences.append(content)
                dialogue_roles.append(role)
                turn_indices.append(str(i))
                
    else:
        # Standard Processing (1 Turn = 1 Point)
        print(f"Strategy '{strategy_name}' detected as Interactive. Using Standard Turns.")
        for i, turn in enumerate(conversation_log):
            if ":" in turn:
                role, content = turn.split(":", 1)
                content = content.strip()
            else:
                role = "Unknown"
                content = turn
            dialogue_sentences.append(content)
            dialogue_roles.append(role.strip())
            turn_indices.append(str(i))

    # --- STEP C: Unified Embedding ---
    combined_text = ontology_sentences + dialogue_sentences
    print(f"Embedding {len(combined_text)} items ({len(dialogue_sentences)} dialogue points)...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(combined_text)
    
    # --- STEP D: UMAP Projection ---
    # Using 'cosine' metric is critical for semantic trajectories
    reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    # --- STEP E: Separate Coordinates ---
    n_ontology = len(ontology_sentences)
    ont_x = coords[:n_ontology, 0]
    ont_y = coords[:n_ontology, 1]
    diag_x = coords[n_ontology:, 0]
    diag_y = coords[n_ontology:, 1]

    # --- STEP F: Visualization ---
    plt.figure(figsize=(12, 8))
    
    # 1. Background: Ontology Cloud
    plt.scatter(ont_x, ont_y, c='lightgrey', s=100, alpha=0.5, label='Ontology Facts')
    
    # 2. Path: Trajectory Line
    # Dashed for standard, Dotted for chunked (visual cue)
    line_style = ':' if should_chunk else '--'
    plt.plot(diag_x, diag_y, c='black', alpha=0.3, linestyle=line_style, linewidth=1)
    
    # 3. Points: Turns
    colors = ['blue' if r == 'Teacher' else 'red' for r in dialogue_roles]
    # Smaller dots for chunks, larger for full turns
    sizes = [60 if '.' in idx else 150 for idx in turn_indices]
    
    plt.scatter(diag_x, diag_y, c=colors, s=sizes, edgecolors='black', zorder=10)
    
    # 4. Annotations
    # To avoid clutter in chunked plots, only label whole numbers or the first chunk
    for k, txt in enumerate(turn_indices):
        is_major_step = '.' not in txt or txt.endswith('.1')
        
        # Only label if it's a standard turn OR a major chunk step
        if not should_chunk or (should_chunk and is_major_step):
             plt.annotate(txt, (diag_x[k], diag_y[k]), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', markersize=10),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
    plt.legend(custom_lines, ['Ontology Fact', 'Teacher Turn', 'Learner Turn'], loc='upper left')

    plt.title(f"Conversational Trajectory (UMAP): {strategy_name}", fontsize=16)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(f"trajectory_{strategy_name}.png", dpi=300)
    plt.show()

def plot_3d_trajectory(ontology_data, conversation_log, strategy_name="Unknown Strategy"):
    
    # --- STEP A: Prepare Ontology ---
    ontology_sentences = []
    ont_labels = []
    for species, props in ontology_data.items():
        for category, value in props.items():
            ontology_sentences.append(f"{species}, {category}, {value}")
            ont_labels.append(f"{species}: {value}")

    # --- STEP B: Prepare Dialogue (With Chunking Logic) ---
    dialogue_sentences = []
    dialogue_roles = []
    turn_indices = []
    hover_texts = [] # Store actual text for the tooltip

    should_chunk = strategy_name in ["top_down", "bottom_up"]
    
    for i, turn in enumerate(conversation_log):
        if ":" in turn:
            role, content = turn.split(":", 1)
            content = content.strip()
        else:
            role = "Unknown"
            content = turn
            
        word_count = len(content.split())
        
        # Chunking Logic for dense strategies
        if should_chunk and role == "Teacher" and word_count > 30:
            chunks = re.split(r'(?<=[.!?]) +', content)
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) > 5:
                    dialogue_sentences.append(chunk)
                    dialogue_roles.append(role)
                    turn_indices.append(f"{i}.{j+1}")
                    hover_texts.append(f"<b>Turn {i}.{j+1}</b><br>{chunk[:100]}...")
        else:
            dialogue_sentences.append(content)
            dialogue_roles.append(role)
            turn_indices.append(str(i))
            hover_texts.append(f"<b>Turn {i}</b><br>{content[:100]}...")

    # --- STEP C: Unified Embedding ---
    combined_text = ontology_sentences + dialogue_sentences
    print(f"Embedding {len(combined_text)} items...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(combined_text)
    
    # --- STEP D: 3D UMAP Projection ---
    print("Running 3D UMAP...")
    reducer = umap.UMAP(
        n_components=3,  # <--- CRITICAL CHANGE: 3 Dimensions
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    
    # --- STEP E: Separate Data ---
    n_ont = len(ontology_sentences)
    ont_x, ont_y, ont_z = coords[:n_ont, 0], coords[:n_ont, 1], coords[:n_ont, 2]
    diag_x, diag_y, diag_z = coords[n_ont:, 0], coords[n_ont:, 1], coords[n_ont:, 2]

    # --- STEP F: Plotly Visualization ---
    fig = go.Figure()

    # 1. Ontology Cloud (Grey Points)
    fig.add_trace(go.Scatter3d(
        x=ont_x, y=ont_y, z=ont_z,
        mode='markers',
        marker=dict(size=4, color='green', opacity=1),
        name='Ontology Facts',
        text=ont_labels,
        hoverinfo='text'
    ))

    # 2. Trajectory Line (The Path)
    fig.add_trace(go.Scatter3d(
        x=diag_x, y=diag_y, z=diag_z,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Trajectory'
    ))

    # 3. Dialogue Points (Teacher = Blue, Learner = Red)
    # Map colors
    colors = ['blue' if r == 'Teacher' else 'red' for r in dialogue_roles]
    
    fig.add_trace(go.Scatter3d(
        x=diag_x, y=diag_y, z=diag_z,
        mode='markers',
        marker=dict(size=6, color=colors, opacity=1),
        name='Dialogue Turns',
        text=hover_texts,
        hoverinfo='text'
    ))

    # Layout Settings
    fig.update_layout(
        title=f"3D Conversational Trajectory: {strategy_name}",
        scene=dict(
            xaxis_title='Semantic Dim 1',
            yaxis_title='Semantic Dim 2',
            zaxis_title='Semantic Dim 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save as HTML (Open this file in any browser to rotate/zoom)
    output_filename = f"trajectory_3d_{strategy_name}.html"
    fig.write_html(output_filename)
    print(f"Saved interactive 3D plot to {output_filename}")
    
    # If running in Jupyter, this will display inline
    fig.show()

def build_fixed_ontology_space(ontology_data):
    """
    Creates a fixed 3D semantic space based ONLY on the ontology.
    Returns the reducer (the map) and the fixed ontology coordinates.
    """
    # 1. Prepare Ontology Text
    ontology_sentences = []
    ont_labels = []
    for species, props in ontology_data.items():
        for category, value in props.items():
            ontology_sentences.append(f"{species}, {category}, {value}")
            ont_labels.append(f"{species}: {value}")

    # 2. Embed Ontology
    print(f"Building fixed map from {len(ontology_sentences)} ontology facts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ont_embeddings = model.encode(ontology_sentences)

    # 3. Fit UMAP on Ontology ONLY
    # This defines the coordinate system based purely on truth/facts.
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    reducer.fit(ont_embeddings) # We only FIT here, we don't transform yet
    
    # Get the coordinates for the ontology itself
    ont_coords = reducer.transform(ont_embeddings)
    
    return {
        "reducer": reducer,
        "model": model, # We must use the exact same embedding model later
        "ont_coords": ont_coords,
        "ont_sentences": ontology_sentences,
        "ont_labels": ont_labels
    }

# ==========================================
# PART 2: PROJECT STRATEGY (Run for each log)
# ==========================================
CHUNK_STRATEGIES = ["top_down", "bottom_up"]

def plot_trajectory_in_fixed_space(fixed_space_data, conversation_log, strategy_name):
    """
    Projects a conversation into the pre-calculated ontology space.
    """
    reducer = fixed_space_data["reducer"]
    model = fixed_space_data["model"]
    ont_coords = fixed_space_data["ont_coords"]
    ont_labels = fixed_space_data["ont_labels"]

    # --- Prepare Dialogue (Chunking Logic) ---
    dialogue_sentences = []
    dialogue_roles = []
    hover_texts = []
    
    should_chunk = any(s in strategy_name for s in CHUNK_STRATEGIES)
    should_chunk = True
    
    for i, turn in enumerate(conversation_log):
        if ":" in turn:
            role, content = turn.split(":", 1)
            content = content.strip()
        else:
            role = "Unknown"
            content = turn
            
        word_count = len(content.split())
        
        if should_chunk and role == "Teacher" and word_count > 30:
            chunks = re.split(r'(?<=[.!?]) +', content)
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) > 5:
                    dialogue_sentences.append(chunk)
                    dialogue_roles.append(role)
                    hover_texts.append(f"<b>Turn {i}.{j+1}</b><br>{chunk[:100]}...")
        else:
            dialogue_sentences.append(content)
            dialogue_roles.append(role)
            hover_texts.append(f"<b>Turn {i}</b><br>{content[:100]}...")

    # --- Embed Dialogue ---
    # use the SAME model from the fixed space
    if len(dialogue_sentences) > 0:
        diag_embeddings = model.encode(dialogue_sentences)
        
        # --- TRANSFORM (Project into fixed space) ---
        # Critical: We use .transform(), NOT .fit_transform()
        # This forces the new points to respect the existing coordinate system
        diag_coords = reducer.transform(diag_embeddings)
        
        diag_x, diag_y, diag_z = diag_coords[:, 0], diag_coords[:, 1], diag_coords[:, 2]
    else:
        diag_x, diag_y, diag_z = [], [], []

    # --- Visualization ---
    fig = go.Figure()

    # 1. Fixed Ontology Cloud
    fig.add_trace(go.Scatter3d(
        x=ont_coords[:, 0], y=ont_coords[:, 1], z=ont_coords[:, 2],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.5),
        name='Ontology Facts (Fixed)',
        text=ont_labels,
        hoverinfo='text'
    ))

    # 2. Dialogue Trajectory
    if len(diag_x) > 0:
        fig.add_trace(go.Scatter3d(
            x=diag_x, y=diag_y, z=diag_z,
            mode='lines',
            line=dict(color='black', width=3, dash='solid'), # Solid line for visibility
            name='Trajectory'
        ))

        colors = ['blue' if r == 'Teacher' else 'red' for r in dialogue_roles]
        fig.add_trace(go.Scatter3d(
            x=diag_x, y=diag_y, z=diag_z,
            mode='markers',
            marker=dict(size=6, color=colors, opacity=1),
            name='Dialogue Turns',
            text=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"Fixed Space Trajectory: {strategy_name}",
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3',
            aspectmode='cube' # Keeps the box shape consistent
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_filename = f"fixed_trajectory_{strategy_name}.html"
    fig.write_html(output_filename)
    print(f"Saved {output_filename}")
    #fig.show()

def plot_surface_in_fixed_space(fixed_space_data, conversation_log, strategy_name):
    """
    Visualizes the conversation as a 3D Semantic Surface (Mesh/Hull).
    """
    reducer = fixed_space_data["reducer"]
    model = fixed_space_data["model"]
    ont_coords = fixed_space_data["ont_coords"]
    ont_labels = fixed_space_data["ont_labels"]

    # --- Prepare Dialogue (Chunking Logic) ---
    dialogue_sentences = []
    dialogue_roles = []
    hover_texts = []
    
    # We force chunking to get enough points to build a good surface
    should_chunk = True 
    
    for i, turn in enumerate(conversation_log):
        if ":" in turn:
            role, content = turn.split(":", 1)
            content = content.strip()
        else:
            role = "Unknown"
            content = turn
            
        word_count = len(content.split())
        
        if should_chunk and role == "Teacher" and word_count > 30:
            chunks = re.split(r'(?<=[.!?]) +', content)
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) > 5:
                    dialogue_sentences.append(chunk)
                    dialogue_roles.append(role)
                    hover_texts.append(f"<b>Turn {i}.{j+1}</b><br>{chunk[:100]}...")
        else:
            dialogue_sentences.append(content)
            dialogue_roles.append(role)
            hover_texts.append(f"<b>Turn {i}</b><br>{content[:100]}...")

    # --- Embed & Transform ---
    if len(dialogue_sentences) > 0:
        diag_embeddings = model.encode(dialogue_sentences)
        diag_coords = reducer.transform(diag_embeddings)
        dx, dy, dz = diag_coords[:, 0], diag_coords[:, 1], diag_coords[:, 2]
    else:
        dx, dy, dz = [], [], []

    # --- Visualization ---
    fig = go.Figure()

    # 1. Ontology Points (The Reference Skeleton)
    fig.add_trace(go.Scatter3d(
        x=ont_coords[:, 0], y=ont_coords[:, 1], z=ont_coords[:, 2],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.8),
        name='Ontology Facts',
        text=ont_labels,
        hoverinfo='text'
    ))

    # 2. Dialogue Surface (The Semantic Hull)
    if len(dx) > 4: # Need at least 4 points to make a 3D volume
        fig.add_trace(go.Mesh3d(
            x=dx, y=dy, z=dz,
            # alphahull: Determines the shape. 
            # -1 = Convex Hull (Standard shrink-wrap)
            # 0 or higher = Concave Hull (More fitted to the specific points)
            alphahull=-1, 
            opacity=0.2, # See-through so you can check if Ontology points are INSIDE
            color='blue',
            name='Semantic Volume'
        ))

    # 3. Dialogue Points (Optional: Keep them small to see density)
    if len(dx) > 0:
        colors = ['blue' if r == 'Teacher' else 'red' for r in dialogue_roles]
        fig.add_trace(go.Scatter3d(
            x=dx, y=dy, z=dz,
            mode='markers',
            marker=dict(size=3, color=colors, opacity=0.1), # Subtle markers
            name='Turns',
            text=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"Semantic Surface: {strategy_name}",
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_filename = f"surface_view_{strategy_name}.html"
    fig.write_html(output_filename)
    print(f"Saved {output_filename}")
    # fig.show()

def old_compute_metrics_per_strategy(fixed_space, groups, logs_directory="."):
    """
    Iterates through strategies in 'groups', loads their logs, 
    and computes Coverage, Proximity, and Dispersion in the fixed 3D space.
    
    Args:
        fixed_space (dict): The output from build_fixed_space() containing 'reducer', 'model', etc.
        groups (dict): Dictionary of strategy categories.
        logs_directory (str): Path to the folder containing your .json logs.
        
    Returns:
        pd.DataFrame: A ranked table of all strategies.
    """
    results = []
    
    # Radius for "Coverage" (1.0 is a standard heuristic for UMAP semantic similarity)
    COVERAGE_RADIUS = 1.0 

    print(f"Computing metrics for {sum(len(v) for v in groups.values())} strategies...")

    for group_name, strategy_list in groups.items():
        for strategy in strategy_list:
            
            # 1. LOAD THE LOG FILE
            # Construct filename (e.g., "learner_questions_train.json")
            filename = os.path.join(logs_directory, f"{strategy}_train.json")
            
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Assumes standard structure: [{"conversation": [...]}]
                        conversation_log = data[0]['conversation'][:-2]
                else:
                    print(f"⚠️ Warning: File not found for {strategy}. Skipping.")
                    continue
            except Exception as e:
                print(f"❌ Error loading {strategy}: {e}")
                continue

            # 2. PROCESS TEXT (No Chunking)
            text_data = conversation_log
            
            if not text_data:
                print(f"⚠️ Warning: No valid text found in {strategy}.")
                continue

            # 3. EMBED & PROJECT (Use .transform, NOT .fit)
            # We use the model and reducer from the FIXED space to ensure fair comparison
            embeddings = fixed_space["model"].encode(text_data)
            coords = fixed_space["reducer"].transform(embeddings)
            
            # 4. COMPUTE DISTANCES
            # Matrix: Rows=Dialogue Turns, Cols=Ontology Facts
            # d[i, j] = Distance from Turn i to Fact j
            dists = distance.cdist(coords, fixed_space["ont_coords"], metric='euclidean')
            
            # --- METRIC A: Semantic Coverage ---
            # For each ontology fact (column), find the CLOSEST dialogue turn.
            # If that distance < RADIUS, the fact is considered "covered".
            min_dist_per_fact = np.min(dists, axis=0)
            covered_count = np.sum(min_dist_per_fact < COVERAGE_RADIUS)
            coverage_pct = (covered_count / len(fixed_space["ont_coords"])) * 100
            
            # --- METRIC B: Semantic Proximity ---
            # For each dialogue turn (row), how close is it to the NEAREST valid fact?
            # Lower = Closer to truth. Higher = More drift/reasoning or hallucination.
            min_dist_per_turn = np.min(dists, axis=1)
            avg_proximity = np.mean(min_dist_per_turn)
            
            # --- METRIC C: Dispersion ---
            # Standard deviation of the dialogue points. 
            # High = Explored the space widely. Low = Stuck in one area.
            dispersion = np.std(coords)
            
            results.append({
                "Group": group_name,
                "Strategy": strategy,
                "Coverage (%)": round(coverage_pct, 1),
                "Avg Proximity": round(avg_proximity, 2),
                "Dispersion": round(dispersion, 2)
            })

    # 5. CREATE & RANK DATAFRAME
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Coverage (%)", ascending=False)
    
    return df

def compute_metrics_per_strategy(fixed_space, groups, logs_directory=".", chunking=False):
    """
    Computes 3D semantic metrics for strategy logs.
    
    Args:
        fixed_space (dict): The output from build_fixed_space() containing 'reducer', 'model', etc.
        groups (dict): Dictionary of strategy categories.
        logs_directory (str): Path to folder containing .json logs.
        chunking (bool): If True, splits LONG turns (>30 words) into individual sentences 
                         for finer granularity. If False, treats each turn as one point.
        
    Returns:
        pd.DataFrame: Ranked results.
    """
    results = []
    
    # Heuristic radius for "semantically touching" in UMAP space
    COVERAGE_RADIUS = 1.0 

    total_strategies = sum(len(v) for v in groups.values())
    print(f"Computing metrics for {total_strategies} strategies (Chunking={chunking})...")

    for group_name, strategy_list in groups.items():
        for strategy in strategy_list:
            
            # 1. LOAD FILE
            filename = os.path.join(logs_directory, f"{strategy}_train.json")
            
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Remove last 2 turns (evaluation phase) if needed, based on your previous logic
                        conversation_log = data[0]['conversation'][:-2]
                else:
                    print(f"⚠️ Warning: File not found for {strategy}. Skipping.")
                    continue
            except Exception as e:
                print(f"❌ Error loading {strategy}: {e}")
                continue

            # 2. PROCESS TEXT (Conditional Chunking)
            processed_text = []
            
            for turn in conversation_log:
                # Extract clean content
                if ":" in turn:
                    role, content = turn.split(":", 1)
                    content = content.strip()
                else:
                    content = turn.strip()

                # Apply Chunking Logic
                # If enabled AND text is long enough to be a paragraph (>30 words)
                if chunking and len(content.split()) > 30:
                    # Split by sentence delimiters (., !, ?)
                    sentences = re.split(r'(?<=[.!?]) +', content)
                    for s in sentences:
                        # Filter out noise (e.g. "Ok.", "Right.")
                        if len(s.strip()) > 5:
                            processed_text.append(s.strip())
                else:
                    # Standard Mode: Keep whole turn if it has content
                    if len(content) > 1:
                        processed_text.append(content)
            
            if not processed_text:
                print(f"⚠️ Warning: No valid text found in {strategy}.")
                continue

            # 3. EMBED & PROJECT
            # Critical: Use .transform() to project into the EXISTING fixed space
            embeddings = fixed_space["model"].encode(processed_text)
            coords = fixed_space["reducer"].transform(embeddings)
            
            # 4. COMPUTE METRICS
            # Calculate Euclidean distances between every Dialogue Point and every Ontology Fact
            dists = distance.cdist(coords, fixed_space["ont_coords"], metric='euclidean')
            
            # Metric A: Coverage (Completeness)
            # How many ontology facts have at least one dialogue point within radius?
            min_dist_per_fact = np.min(dists, axis=0)
            covered_count = np.sum(min_dist_per_fact < COVERAGE_RADIUS)
            coverage_pct = (covered_count / len(fixed_space["ont_coords"])) * 100
            
            # Metric B: Proximity (Relevance)
            # Average distance of dialogue points to the nearest truth
            min_dist_per_turn = np.min(dists, axis=1)
            avg_proximity = np.mean(min_dist_per_turn)
            
            # Metric C: Dispersion (Exploration)
            # Standard deviation of the dialogue cloud
            dispersion = np.std(coords)
            
            results.append({
                "Group": group_name,
                "Strategy": strategy,
                "Coverage (%)": round(coverage_pct, 1),
                "Avg Proximity": round(avg_proximity, 2),
                "Dispersion": round(dispersion, 2)
            })

    # 5. SORT & RETURN
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Coverage (%)", ascending=False)
    
    return df
# Run 
if __name__ == "__main__":
    config = load_config("config.yml")
    ontology_file = config["ontology"]["file"]
    save_dir = config.get("save_dir", "results")
    model_name = config["model"]["name"]
    strategy = config["Teacher"]["strategy"]

    with open(ontology_file, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
        print("[Main] Loaded ontology from ", ontology_file)
    #plot_ontology_cloud(ontology, color_by_attribute = False) 

    # Load conversation log
    TRAINING_LOGS_DIR = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "dialogue")
    filename = os.path.join(TRAINING_LOGS_DIR, f"{strategy}_train.json")
    conversation = load_trials(filename)[0]["conversation"][:-2]
    #plot_trajectory_umap(ontology, conversation, strategy_name= strategy)
    #plot_3d_trajectory(ontology, conversation, strategy_name= strategy)

    # Build Fixed Ontology Space
    fixed_space = build_fixed_ontology_space(ontology)
    plot_surface_in_fixed_space(fixed_space, conversation, strategy_name=strategy)
    # Compute Metrics Across Strategies
    metrics_df = compute_metrics_per_strategy(fixed_space, groups, logs_directory=TRAINING_LOGS_DIR, chunking=True )
    print("\n=== Strategy Metrics ===")
    print(metrics_df.to_string(index=False))


    """
    GPT 4o
    === Strategy Metrics === WITHOUT CHUNKING SENTENCES
      Group                          Strategy          Coverage (%)  Avg Proximity  Dispersion
    Dialogic TD  mixed_top-down_teacher_questions          68.0           0.83        3.88
    Dialogic BU mixed_bottom-up_teacher_questions          64.0           0.80        3.89
    Dialogic Qs         mixed_learner_questions_2          56.0           0.80        4.08
    Dialogic Qs         mixed_learner_questions_3          52.0           0.84        4.27
    Dialogic Qs mixed_teacher_questions Questions          52.0           0.83        3.85
                                learner_questions          46.0           0.87        4.06
    Dialogic Qs           mixed_learner_questions          44.0           0.82        4.03
    Dialogic BU mixed_bottom-up_learner_questions          40.0           0.81        3.75
    Dialogic TD  mixed_top-down_learner_questions          38.0           0.82        4.59
                      teacher_questions Questions          32.0           0.80        3.52
   """


    """
    GPT 4o
    === Strategy Metrics === WItH CHUNKING SENTENCES
       Group                       Strategy          Coverage (%)  Avg Proximity  Dispersion
     Dialogic TD  mixed_top-down_teacher_questions          92.0           0.83        4.06
     Dialogic BU mixed_bottom-up_learner_questions          90.0           0.82        3.91
     Dialogic Qs           mixed_teacher_questions          88.0           0.85        3.96
     Dialogic Qs           mixed_learner_questions          88.0           0.81        4.03
     Dialogic BU mixed_bottom-up_teacher_questions          88.0           0.83        3.99
       Questions                 teacher_questions          86.0           0.81        3.82
     Dialogic TD  mixed_top-down_learner_questions          86.0           0.88        4.32
       Questions                 learner_questions          78.0           0.88        4.05
    Instructions                         bottom_up          56.0           0.76        4.19
    Instructions                          top_down          42.0           0.79        3.87
    """

    """
    LLAMA 3=== Strategy Metrics ===
       Group                          Strategy  Coverage (%)  Avg Proximity  Dispersion
 Dialogic BU mixed_bottom-up_learner_questions          90.0           0.84        4.08
 Dialogic TD  mixed_top-down_learner_questions          88.0           0.84        4.06
 Dialogic BU mixed_bottom-up_teacher_questions          86.0           0.87        3.97
   Questions                 learner_questions          86.0           0.86        4.05
 Dialogic Qs           mixed_learner_questions          84.0           0.86        3.96
   Questions                 teacher_questions          80.0           0.95        3.80
 Dialogic Qs           mixed_teacher_questions          80.0           0.84        4.13
 Dialogic TD  mixed_top-down_teacher_questions          70.0           0.90        3.92
Instructions                         bottom_up          42.0           0.86        3.86
Instructions                          top_down          42.0           0.80        3.82

    """