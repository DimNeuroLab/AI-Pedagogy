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

def get_facts(ontology):

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