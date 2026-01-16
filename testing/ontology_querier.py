import json


class OntologyQuerier:
    def __init__(self, ontology):
        """
        ontology: dict mapping species names to their feature dicts.
        Example:
        {
            "Zorblaxian": {
                "Habitat": "Crystal Caves",
                "Diet": "Luminous Spores",
                ...
            },
            ...
        }
        """
        self.ontology = ontology

    def query_with_features(self, feature_values):
        """
        Query the ontology for species matching all feature-value pairs.

        Args:
            feature_values (dict): e.g. {'Diet': 'Luminous Spores', 'Habitat': 'Crystal Caves'}

        Returns:
            List[str]: Names of matching species.
        """
        matches = []
        for name, features in self.ontology.items():
            if all(features.get(feature) == value for feature, value in feature_values.items()):
                matches.append(name)
        return matches
    
    def query_with_features_difference(self, feature_values):
        """
        Query the ontology for species that do NOT match all feature-value pairs.

        Args:
            feature_values (dict): e.g. {'Diet': 'Luminous Spores', 'Habitat': 'Crystal Caves'}

        Returns:
            List[str]: Names of species that do NOT match all provided features.
        """
        mismatches = []
        for name, features in self.ontology.items():
            if not all(features.get(feature) == value for feature, value in feature_values.items()):
                mismatches.append(name)
        return mismatches

    def list_feature_values(self):
        """
        Extract all possible values for each feature across the ontology.

        Returns:
            dict: {feature_name: sorted list of values}
        """
        feature_values = {}
        for features in self.ontology.values():
            for feature, value in features.items():
                feature_values.setdefault(feature, set()).add(value)
        return {k: sorted(v) for k, v in feature_values.items()}

    def discriminative_query(self, set1, set2):
        """
        Compare two sets of alien species and return features that are either common with a single value 
        or discriminative (different values in each set).
        Args:
            set1 (list[str]): List of species names for the first set.
            set2 (list[str]): List of species names for the second set.
        Returns:
            {
                "common_features": {feature: shared_value},
                "discriminative_features": {feature: {"set1": [...], "set2": [...]}},
            }
        """
        def get_features(species_names):
            return [
                self.ontology[name] for name in species_names
                if name in self.ontology
            ]

        set1_feats = get_features(set1)
        set2_feats = get_features(set2)

        if not set1_feats or not set2_feats:
            return {"common_features": {}, "discriminative_features": {}}

        all_keys = set(set1_feats[0]).intersection(set2_feats[0])

        common_features = {}
        discriminative_features = {}

        for feature in all_keys:
            values1 = {alien[feature] for alien in set1_feats if feature in alien}
            values2 = {alien[feature] for alien in set2_feats if feature in alien}

            if values1 == values2 and len(values1) == 1:
                # Same single value in both sets
                common_features[feature] = next(iter(values1))
            elif values1.isdisjoint(values2):
                # No overlapping values => discriminative
                discriminative_features[feature] = {
                    "set A": sorted(values1),
                    "set B": sorted(values2)
                }

        result = {
            "common_features": common_features,
            "discriminative_features": discriminative_features
        }

        return json.dumps(result, indent=2)

    def query_with_species(self, species_set):
        """
        Given a list of species names, return the features they all share with the same value.

        Args:
            species_set (list[str]): List of alien species names.

        Returns:
            dict: {feature_name: shared_value} for features that all species have in common.
        """
        if not species_set:
            return {}

        # Gather only valid species present in the ontology
        feature_dicts = [self.ontology[name] for name in species_set if name in self.ontology]

        if not feature_dicts:
            return {}

        # Find intersection of feature keys to be safe
        common_keys = set(feature_dicts[0].keys())
        for fd in feature_dicts[1:]:
            common_keys.intersection_update(fd.keys())

        common = {}
        for feature in common_keys:
            values = {alien[feature] for alien in feature_dicts}
            if len(values) == 1:
                common[feature] = values.pop()

        return common
    
    def query_with_species_difference(self, species_set):
        """
        Given a list of species names, return the features for which the species do NOT all share the same value.

        Args:
            species_set (list[str]): List of alien species names.

        Returns:
            dict: {feature_name: list_of_values} where values differ across species.
        """
        if not species_set:
            return {}

        # Gather only valid species present in the ontology
        feature_dicts = [self.ontology[name] for name in species_set if name in self.ontology]

        if not feature_dicts:
            return {}

        # Find intersection of feature keys to be safe
        common_keys = set(feature_dicts[0].keys())
        for fd in feature_dicts[1:]:
            common_keys.intersection_update(fd.keys())

        differing = {}
        for feature in common_keys:
            values = {alien[feature] for alien in feature_dicts}
            if len(values) > 1:
                differing[feature] = list(values)

        return differing


    

    
