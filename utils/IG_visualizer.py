import tkinter as tk
from tkinter import scrolledtext
import json
import threading
import time
import re
from utils.api_interface import call_local_llm

class OntologyViewer(tk.Tk):
    def __init__(self, ontology, conversation):
        super().__init__()
        self.title("Ontology Dialogue Highlighter")
        self.geometry("1200x800")

        # Validate inputs
        if not isinstance(ontology, dict) or not ontology:
            raise ValueError("Ontology must be a non-empty dictionary")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("Conversation must be a non-empty list")
        
        self.ontology = ontology
        self.conversation = conversation
        self.step = 0
        
        # Cache for extracted triples per step
        self.triples_cache = {}
        self.loading_triples = False
        self.current_extraction_thread = None
        
        # Extract structure from ontology for flexible processing
        self.entities = list(self.ontology.keys())
        self.all_features = self._extract_all_features()
        self.all_values = self._extract_all_values()

        self.setup_ui()
        self.setup_highlighting()
        self.update_buttons()
        self.show_step()

    def _extract_all_features(self):
        """Extract all unique features from the ontology."""
        features = set()
        for entity_data in self.ontology.values():
            if isinstance(entity_data, dict):
                features.update(entity_data.keys())
        return list(features)
    
    def _extract_all_values(self):
        """Extract all unique values from the ontology."""
        values = set()
        for entity_data in self.ontology.values():
            if isinstance(entity_data, dict):
                for value in entity_data.values():
                    if isinstance(value, (str, int, float)):
                        values.add(str(value))
        return list(values)

    def setup_ui(self):
        """Setup the user interface components."""
        # Ontology Text (left side)
        self.ontology_text = scrolledtext.ScrolledText(
            self, width=80, height=35, font=("Courier", 11)
        )
        self.ontology_text.grid(row=0, column=0, rowspan=3, padx=10, pady=10, sticky="nsew")

        # Dialogue Text (top right)
        self.dialogue_text = tk.Text(
            self, width=50, height=10, font=("Arial", 11), wrap="word", bg="#f0f0f0"
        )
        self.dialogue_text.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.dialogue_text.config(state="disabled")

        # Buttons (middle right)
        self.btn_frame = tk.Frame(self)
        self.btn_frame.grid(row=1, column=1, pady=10)
        
        self.back_btn = tk.Button(self.btn_frame, text="◀ Back", command=self.back_step, width=10)
        self.back_btn.grid(row=0, column=0, padx=5)
        
        self.next_btn = tk.Button(self.btn_frame, text="Next ▶", command=self.next_step, width=10)
        self.next_btn.grid(row=0, column=1, padx=5)
        
        self.reset_btn = tk.Button(self.btn_frame, text="🔄 Reset", command=self.reset_viewer, width=10)
        self.reset_btn.grid(row=0, column=2, padx=5)
        
        # Add manual cancel button during loading
        self.cancel_btn = tk.Button(self.btn_frame, text="❌ Cancel", command=self.cancel_extraction, width=10)
        self.cancel_btn.grid(row=0, column=3, padx=5)

        # Status and information panel (bottom right)
        info_frame = tk.Frame(self)
        info_frame.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="nsew")
        
        # Step info
        self.step_label = tk.Label(
            info_frame, text="Step 1 of 1", font=("Arial", 12, "bold")
        )
        self.step_label.pack(anchor="w", pady=(0, 10))
        
        # Loading status
        self.loading_label = tk.Label(
            info_frame, text="", font=("Arial", 10), fg="blue", bg="#fff8dc", padx=10, pady=5
        )
        self.loading_label.pack(anchor="w", fill="x", pady=(0, 10))

        # Extracted triples display
        tk.Label(info_frame, text="Extracted Triples:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.triples_frame = tk.Frame(info_frame, bg="white", relief="sunken", bd=1)
        self.triples_frame.pack(anchor="w", fill="both", expand=True, pady=(5, 0))
        
        self.triples_text = tk.Text(
            self.triples_frame, width=40, height=12, font=("Courier", 9), 
            wrap="word", bg="white", state="disabled"
        )
        self.triples_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Setup ontology display
        self.pretty_json = json.dumps(self.ontology, indent=4)
        self.ontology_text.insert("1.0", self.pretty_json)
        self.ontology_text.config(state="disabled")

        # Responsive layout weights
        self.grid_columnconfigure(0, weight=2)  # Ontology gets more space
        self.grid_columnconfigure(1, weight=1)  # Controls get less space
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_highlighting(self):
        """Setup text highlighting tags."""
        self.ontology_text.tag_config("highlight_feature", background="#ccffcc", foreground="darkgreen", font=("Courier", 11, "bold"))
        self.ontology_text.tag_config("highlight_value", background="#fff2aa", foreground="darkorange", font=("Courier", 11, "bold")) 
        self.ontology_text.tag_config("highlight_entity", background="#ffcccc", foreground="darkred", font=("Courier", 11, "bold"))
        # Add a new tag for mentioned entities (lighter highlighting)
        self.ontology_text.tag_config("mention_entity", background="#ffe6e6", foreground="darkred", font=("Courier", 11))

    def reset_viewer(self):
        """Reset to initial step and clear caches."""
        # Cancel any ongoing extraction
        if self.current_extraction_thread and self.current_extraction_thread.is_alive():
            self.loading_triples = False  # Signal thread to stop
            
        self.step = 0
        self.triples_cache.clear()
        self.loading_triples = False
        self.current_extraction_thread = None
        
        self.update_buttons()
        self.show_step()

    def on_close(self):
        """Handle window closing."""
        print("Closing Ontology Viewer...")
        self.loading_triples = False  # Signal threads to stop
        self.destroy()
        self.quit()

    def cancel_extraction(self):
        """Manually cancel ongoing extraction."""
        if self.loading_triples:
            print("Manually cancelling extraction")
            self.loading_triples = False
            self.loading_label.config(text="❌ Extraction cancelled", bg="#ffdddd")
            self.update_buttons()
            
            # Clear message after delay
            self.after(3000, lambda: self.loading_label.config(text="", bg="#f0f0f0"))

    def update_buttons(self):
        """Update button states based on current conditions."""
        if self.loading_triples:
            self.back_btn.config(state="disabled")
            self.next_btn.config(state="disabled", text="⏳ Loading...")
            self.reset_btn.config(state="disabled")
            self.cancel_btn.config(state="normal")  # Enable cancel during loading
        else:
            self.back_btn.config(state="normal" if self.step > 0 else "disabled")
            self.next_btn.config(
                state="normal" if self.step < len(self.conversation) - 1 else "disabled",
                text="Next ▶"
            )
            self.reset_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")  # Disable cancel when not loading

    def show_step(self):
        """Display the current conversation step."""
        current = self.conversation[self.step]
        role = current["role"].capitalize()
        content = current["content"]

        # Update step info
        self.step_label.config(text=f"Step {self.step + 1} of {len(self.conversation)}")

        # Update dialogue display
        self.dialogue_text.config(state="normal")
        self.dialogue_text.delete("1.0", "end")
        self.dialogue_text.insert("1.0", f"{role}: {content}")
        self.dialogue_text.config(state="disabled")

        # Clear previous highlighting
        self.clear_highlights()

        # Check if we have cached triples for this step
        if self.step in self.triples_cache:
            triples = self.triples_cache[self.step]
            self.highlight_triples(triples)
            self.display_triples(triples)
            # Also do fallback highlighting for mentioned entities
            self.highlight_mentioned_entities(content)
        else:
            # Extract triples in background thread
            self.extract_and_highlight_triples(content)

        self.update_buttons()

    def highlight_mentioned_entities(self, content):
        """Fallback highlighting for entities mentioned in text even without triples."""
        mentioned_entities = []
        content_lower = content.lower()
        
        for entity in self.entities:
            # Check for exact matches (case insensitive)
            if entity.lower() in content_lower:
                mentioned_entities.append(entity)
        
        if mentioned_entities:
            print(f"Found mentioned entities (fallback): {mentioned_entities}")
            self.ontology_text.config(state="normal")
            for entity in mentioned_entities:
                self.highlight_entity_name_light(entity)
            self.ontology_text.config(state="disabled")

    def highlight_entity_name_light(self, entity):
        """Light highlighting for mentioned entities."""
        try:
            pattern = f'"{entity}": {{'
            idx = self.ontology_text.search(pattern, "1.0", stopindex="end")
            if idx:
                end_idx = f"{idx}+{len(entity)}c"
                self.ontology_text.tag_add("mention_entity", f"{idx}+1c", end_idx)
                print(f"  Light highlighted entity name '{entity}' at {idx}")
        except Exception as e:
            print(f"Error light highlighting entity name {entity}: {e}")

    def extract_and_highlight_triples(self, content):
        """Extract triples using LLM in background thread."""
        if self.loading_triples:
            return  # Already loading

        self.loading_triples = True
        self.loading_label.config(text="🔄 Extracting triples...", bg="#fff8dc")
        self.clear_triples_display()
        self.update_buttons()

        # Cancel any existing thread
        if self.current_extraction_thread and self.current_extraction_thread.is_alive():
            print("Cancelling previous extraction thread")
            # The thread will check self.loading_triples and exit

        def extract_with_timeout():
            try:
                if not self.loading_triples:  # Check if cancelled
                    print("Extraction cancelled before starting")
                    return
                
                print(f"Starting triple extraction for: {content[:50]}...")
                start_time = time.time()
                
                # Add periodic checks during extraction
                triples = self.extract_triples_with_checks(content)
                
                elapsed = time.time() - start_time
                print(f"Triple extraction completed in {elapsed:.2f}s")
                
                # Schedule UI update in main thread only if still loading
                if self.loading_triples:
                    self.after(0, lambda: self.on_triples_extracted(triples, content))
                else:
                    print("Extraction completed but was cancelled - ignoring results")
                    
            except Exception as e:
                print(f"Error in triple extraction thread: {e}")
                import traceback
                traceback.print_exc()
                if self.loading_triples:
                    self.after(0, lambda: self.on_triples_extracted([], content))

        self.current_extraction_thread = threading.Thread(target=extract_with_timeout)
        self.current_extraction_thread.daemon = True
        self.current_extraction_thread.start()

        # Set a timeout to prevent indefinite loading
        self.after(20000, self.extraction_timeout)  # 20 second timeout

    def extraction_timeout(self):
        """Handle extraction timeout."""
        if self.loading_triples:
            print("Triple extraction timed out - forcing cancellation")
            self.loading_triples = False
            self.loading_label.config(text="⚠️ Extraction timed out", bg="#ffdddd")
            self.update_buttons()
            
            # Clear timeout message after delay
            self.after(5000, lambda: self.loading_label.config(text="", bg="#f0f0f0"))

    def on_triples_extracted(self, triples, content):
        """Handle completion of triple extraction."""
        if not self.loading_triples:
            return  # Was cancelled or timed out
            
        self.loading_triples = False
        self.loading_label.config(text="✅ Extraction complete", bg="#e8f5e8")
        
        # Process and validate triples
        processed_triples = self.process_triples_response(triples)
        
        # Cache the processed triples
        self.triples_cache[self.step] = processed_triples
        
        # Display and highlight
        self.display_triples(processed_triples)
        self.highlight_triples(processed_triples)
        
        # Also do fallback highlighting for mentioned entities if no triples found
        if not processed_triples:
            self.highlight_mentioned_entities(content)
        
        self.update_buttons()
        
        # Clear status after delay
        self.after(3000, lambda: self.loading_label.config(text="", bg="#f0f0f0"))

    def process_triples_response(self, response):
        """Process the LLM response to extract valid triples."""
        try:
            # If response is already a list, use it directly
            if isinstance(response, list):
                triples = response
            elif isinstance(response, str):
                # Try to parse as JSON
                response = response.strip()
                
                # Remove markdown code blocks if present
                if response.startswith('```'):
                    lines = response.split('\n')
                    response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
                if response.startswith('json'):
                    response = response[4:].strip()
                
                # Try to find JSON array in response
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    response = response[start_idx:end_idx+1]
                
                # Handle empty arrays or "no triples" responses
                if not response or response.strip() in ['[]', '']:
                    return []
                
                triples = json.loads(response)
            else:
                print(f"Unexpected response type: {type(response)}")
                return []
            
            # Validate and clean triples
            valid_triples = []
            for triple in triples:
                if isinstance(triple, list) and len(triple) == 3:
                    # Ensure all elements are strings or None
                    processed_triple = []
                    for item in triple:
                        if item is None or item == "None":
                            processed_triple.append(None)
                        else:
                            processed_triple.append(str(item))
                    valid_triples.append(processed_triple)
                else:
                    print(f"Invalid triple format: {triple}")
            
            return valid_triples
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {str(response)[:200]}...")
            return []
        except Exception as e:
            print(f"Error processing triples response: {e}")
            return []

    def clear_triples_display(self):
        """Clear the triples display area."""
        self.triples_text.config(state="normal")
        self.triples_text.delete("1.0", "end")
        self.triples_text.config(state="disabled")

    def display_triples(self, triples):
        """Display extracted triples in the UI."""
        self.triples_text.config(state="normal")
        self.triples_text.delete("1.0", "end")
        
        if not triples:
            self.triples_text.insert("1.0", "No triples extracted\n(Check for entity mentions)")
        else:
            triple_lines = []
            for i, triple in enumerate(triples):
                if len(triple) >= 3:
                    entity = triple[0] if triple[0] else "All"
                    feature = triple[1] if triple[1] else "?"
                    value = triple[2] if triple[2] else "?"
                    
                    # Handle incomplete triples
                    if triple[1] is None and triple[2] is None:
                        # Entity-only triple
                        triple_lines.append(f"{i+1}. {entity} (entity mentioned)")
                    elif triple[0] is None:
                        # Feature-value triple
                        triple_lines.append(f"{i+1}. All → {feature}: {value}")
                    else:
                        # Complete triple
                        triple_lines.append(f"{i+1}. {entity} → {feature}: {value}")
                else:
                    triple_lines.append(f"{i+1}. Invalid: {triple}")
            
            self.triples_text.insert("1.0", "\n".join(triple_lines))
        
        self.triples_text.config(state="disabled")

    def next_step(self):
        """Move to next conversation step."""
        if self.step < len(self.conversation) - 1 and not self.loading_triples:
            self.step += 1
            self.show_step()

    def back_step(self):
        """Move to previous conversation step."""
        if self.step > 0 and not self.loading_triples:
            self.step -= 1
            self.show_step()

    def clear_highlights(self):
        """Clear all highlighting from the ontology text."""
        self.ontology_text.config(state="normal")
        for tag in ["highlight_feature", "highlight_value", "highlight_entity", "mention_entity"]:
            self.ontology_text.tag_remove(tag, "1.0", "end")
        self.ontology_text.config(state="disabled")

    def highlight_triples(self, triples):
        """Highlight triples in the ontology display."""
        if not triples:
            return

        self.ontology_text.config(state="normal")

        try:
            for triple in triples:
                if len(triple) < 3:
                    continue
                    
                entity, feature, value = triple[0], triple[1], triple[2]
                
                # Handle incomplete triples - entity only
                if entity and feature is None and value is None:
                    print(f"Highlighting entity-only triple: {entity}")
                    self.highlight_entity_name(entity)
                # Handle feature-value pairs without specific entity
                elif entity is None or entity == "None":
                    # Partial triple: highlight all entities that have this feature-value pair
                    self.highlight_all_feature_value_pairs(feature, value)
                else:
                    # Full triple: highlight specific entity-feature-value
                    self.highlight_specific_triple(entity, feature, value)
                    
        except Exception as e:
            print(f"Error highlighting triples: {e}")
        finally:
            self.ontology_text.config(state="disabled")

    def highlight_all_feature_value_pairs(self, feature, value):
        """Highlight all (feature, value) pairs across all entities."""
        print(f"Highlighting all entities with {feature}→{value}")
        for entity in self.entities:
            if (entity in self.ontology and 
                isinstance(self.ontology[entity], dict) and
                self.ontology[entity].get(feature) == value):
                
                print(f"  Found match in entity: {entity}")
                self.highlight_specific_triple(entity, feature, value)

    def highlight_specific_triple(self, entity, feature, value):
        """Highlight a specific (entity, feature, value) triple."""
        try:
            print(f"Highlighting specific triple: {entity} → {feature}: {value}")
            
            # First highlight the entity name
            self.highlight_entity_name(entity)
            
            # Get the entity block to search within
            entity_block = self.get_entity_block(entity)
            if not entity_block:
                print(f"  Could not find entity block for {entity}")
                return

            block_start = f"{entity_block['start']}.0"
            block_end = f"{entity_block['end']}.0"
            print(f"  Entity block: lines {entity_block['start']} to {entity_block['end']}")

            # Highlight the feature
            feature_pattern = f'"{feature}"'
            feature_idx = self.ontology_text.search(feature_pattern, block_start, stopindex=block_end)
            if feature_idx:
                print(f"  Found feature '{feature}' at {feature_idx}")
                self.ontology_text.tag_add("highlight_feature", feature_idx, f"{feature_idx}+{len(feature_pattern)}c")

                # Highlight the value if this entity has this feature with this value
                if (entity in self.ontology and 
                    isinstance(self.ontology[entity], dict) and 
                    self.ontology[entity].get(feature) == value):
                    
                    value_pattern = f'"{value}"'
                    # Search for value after the feature
                    value_idx = self.ontology_text.search(value_pattern, feature_idx, stopindex=block_end)
                    if value_idx:
                        print(f"  Found value '{value}' at {value_idx}")
                        self.ontology_text.tag_add("highlight_value", value_idx, f"{value_idx}+{len(value_pattern)}c")
                    else:
                        print(f"  Could not find value '{value}' after feature")
                else:
                    print(f"  Entity {entity} does not have {feature}='{value}'")
            else:
                print(f"  Could not find feature '{feature}' in entity block")
                        
        except Exception as e:
            print(f"Error highlighting triple ({entity}, {feature}, {value}): {e}")

    def highlight_entity_name(self, entity):
        """Highlight the entity name in the ontology."""
        try:
            pattern = f'"{entity}": {{'
            idx = self.ontology_text.search(pattern, "1.0", stopindex="end")
            if idx:
                end_idx = f"{idx}+{len(entity)}c"
                self.ontology_text.tag_add("highlight_entity", f"{idx}+1c", end_idx)
                print(f"  Highlighted entity name '{entity}' at {idx}")
            else:
                print(f"  Could not find entity name pattern for '{entity}'")
        except Exception as e:
            print(f"Error highlighting entity name {entity}: {e}")

    def get_entity_block(self, entity):
        """Get the line range for an entity's JSON block."""
        try:
            pattern = f'"{entity}": {{'
            start = self.ontology_text.search(pattern, "1.0", stopindex="end")
            if not start:
                return None
            line_start = int(start.split('.')[0])
            lines = self.pretty_json.splitlines()
            brace_level = 0
            started = False
            for i in range(line_start - 1, len(lines)):
                line = lines[i]
                brace_level += line.count("{")
                brace_level -= line.count("}")
                if not started and "{" in line:
                    started = True
                if started and brace_level == 0:
                    return {"start": line_start, "end": i + 1}
        except Exception as e:
            print(f"Error getting entity block for {entity}: {e}")
        return None
    
    def extract_triples_with_checks(self, text):
        """Extract triples with periodic cancellation checks."""
        if not self.loading_triples:
            print("Extraction cancelled during checks")
            return []
            
        try:
            # Call the original extract_triples method
            response = self.extract_triples(text)
            
            # Check if cancelled during LLM call
            if not self.loading_triples:
                print("Extraction cancelled after LLM call")
                return []
                
            return response
            
        except Exception as e:
            print(f"Error in extract_triples_with_checks: {e}")
            return []

    def extract_triples(self, text):
        """ Extract (entity, feature, value) triples from the text.
        This method uses LLM.
        """
        # Use LLM to extract triples
        system_prompt = """You are a precise ontology information extractor. Your task is to identify explicit references to ontological concepts in dialogue and return them as structured triplets.

                            ONTOLOGY STRUCTURE:
                            - Entities: Top-level objects (e.g., "Zeyeqo", "Gokixi")  
                            - Features: Properties/attributes of entities (e.g., "Glimacule", "Nextor")
                            - Values: Specific feature assignments (e.g., "vorka", "throx")

                            TRIPLET FORMAT: [entity, feature, value]
                            - Use exact strings from the provided ontology
                            - Use null (JSON null, not string) for unspecified components
                            - Each triplet must have at least one non-null component

                            EXTRACTION PATTERNS:

                            Pattern 1: Complete Assignment → [entity, feature, value]
                            - "Zeyeqo's Glimacule is vorka" → ["Zeyeqo", "Glimacule", "vorka"]
                            - "The Notizi has Nextor value of twelve" → ["Notizi", "Nextor", "twelve"]

                            Pattern 2: Entity-Feature Reference → [entity, feature, null]
                            - "What is Zeyeqo's Glimacule?" → ["Zeyeqo", "Glimacule", null]
                            - "Tell me about Gokixi's Blumex" → ["Gokixi", "Blumex", null]

                            Pattern 3: Entity-Value Connection → [entity, null, value] 
                            - "Zeyeqo has the value vorka" → ["Zeyeqo", null, "vorka"]
                            - "I see 'twelve' associated with Notizi" → ["Notizi", null, "twelve"]

                            Pattern 4: Entity Mention → [entity, null, null]
                            - "Let's discuss Zeyeqo" → ["Zeyeqo", null, null]
                            - "The entities Gokixi and Notizi" → [["Gokixi", null, null], ["Notizi", null, null]]

                            Pattern 5: Feature-Value Relationship → [null, feature, value]
                            - "Glimacule can be vorka" → [null, "Glimacule", "vorka"]
                            - "The possible Nextor values include twelve" → [null, "Nextor", "twelve"]

                            Pattern 6: Feature Discussion → [null, feature, null]
                            - "What is Glimacule?" → [null, "Glimacule", null]
                            - "Let's examine the Nextor feature" → [null, "Nextor", null]

                            Pattern 7: Value Reference → [null, null, value]
                            - "The value 'vorka' appears in the data" → [null, null, "vorka"]
                            - "I notice 'twelve' is mentioned" → [null, null, "twelve"]

                            CRITICAL RULES:

                            1. Explicit Content Only: Extract only what is directly stated
                            - ❌ "it" → Must have explicit entity name in same sentence
                            - ❌ "that feature" → Must have explicit feature name
                            - ❌ Implied relationships → Only extract stated connections

                            2. No Implicit Inferences: Do not assume relationships not explicitly stated

                            3. Validation Requirements:
                            - Entity names must exist in ontology
                            - Feature names must exist in ontology
                            - Values must exist in ontology
                            - At least one component must be non-null per triplet

                            SPECIAL HANDLING:

                            Lists: "entities A, B, C" → [["A",null,null], ["B",null,null], ["C",null,null]]
                            Negations: Only extract if clear context → "X is not Y" needs clear feature
                            Ambiguous: "the blue one", "that entity" → No extraction (not ontological terms)

                            QUALITY ASSURANCE:
                            Before returning, verify each triplet:
                            ✓ At least one component is non-null
                            ✓ All non-null strings exist in provided ontology
                            ✓ No duplicate triplets in response
                            ✓ JSON array format is valid

                            OUTPUT FORMAT:
                            Return only a valid JSON array. No explanations, no markdown, no additional text.

                            Examples:
                            - Single: [["Zeyeqo", "Glimacule", "vorka"]]
                            - Multiple: [["Zeyeqo", null, null], [null, "Glimacule", null]]
                            - None: []"""

        
        user_prompt = f"""Ontology context:
                        {json.dumps(self.ontology, indent=2)[:1000]}...

                        Dialogue turn to analyze:
                        "{text}"

                        Extract (entity, feature, value) triples as JSON array:"""
        
        message = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        response = call_local_llm(message)

        return response

if __name__ == "__main__":
    # Example usage
    try:
        with open("data/ontologies/ontology_aliens_obfuscated.json", "r", encoding="utf-8") as f:
            ontology = json.load(f)

        # Example conversation for testing
        conversation = [
            {"role": "user", "content": "Great! Let's start by breaking down the ontology, which is essentially a structured way of representing knowledge. We'll begin with the concept of different \"categories\" or \"entities\" in the ontology. Each top-level category, such as \"Gokixi,\" \"Notizi,\" \"Zeyeqo,\" etc., represents a unique entity within this alien ontology.\n\nQuestion: Can you name one of the entities listed in the ontology?"},
            {"role": "assistant", "content": "Looking at the ontology, I can see several entities. One of them is \"Zeyeqo\"."},
            {"role": "user", "content": "Tell me about Zeyeqo's Glimacule"},
            {"role": "assistant", "content": "Zeyeqo's Glimacule is vorka"},
            {"role": "user", "content": "What are the possible Glimacule values?"},
            {"role": "assistant", "content": "Glimacule has values throx, vorka, and glent"}
        ]

        app = OntologyViewer(ontology, conversation)
        app.mainloop()
        
    except FileNotFoundError:
        print("Error: Could not find ontology file. Please check the path.")
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()