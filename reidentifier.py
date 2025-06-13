import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

class ReIdentifier:
    def __init__(self, feature_extractor, threshold=0.5):
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.id_counter = 0
        self.database = {}  # id -> embedding

    def _match(self, embedding):
        best_id = None
        best_score = float('inf')
        for obj_id, saved_embedding in self.database.items():
            score = cosine(embedding, saved_embedding)
            if score < best_score:
                best_score = score
                best_id = obj_id
        if best_score < self.threshold:
            return best_id
        return None

    def reidentify(self, frame, tracked_objects):
        """
        Assigns persistent IDs using appearance-based matching.
        """
        output = {}
        for temp_id, bbox in tracked_objects.items():
            embedding = self.feature_extractor.extract(frame, bbox)
            match_id = self._match(embedding)
            if match_id is not None:
                output[match_id] = bbox
            else:
                output[self.id_counter] = bbox
                self.database[self.id_counter] = embedding
                self.id_counter += 1
        return output
