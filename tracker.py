import numpy as np
from collections import deque

class BasicTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}  # object_id: bbox
        self.disappeared = {}  # object_id: num_frames
        self.max_disappeared = max_disappeared

    def register(self, bbox):
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([
            [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            for bbox in [d['bbox'] for d in detections]
        ])

        if len(self.objects) == 0:
            for bbox in [d['bbox'] for d in detections]:
                self.register(bbox)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([
                [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
                for bbox in self.objects.values()
            ])

            distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = [int(x) for x in [
                    detections[col]['bbox'][0], detections[col]['bbox'][1],
                    detections[col]['bbox'][2], detections[col]['bbox'][3]
                ]]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_centroids))).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(len(input_centroids))).difference(used_cols)
            for col in unused_cols:
                self.register([int(x) for x in detections[col]['bbox']])

        return self.objects
