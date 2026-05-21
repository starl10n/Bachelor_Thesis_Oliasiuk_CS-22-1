from collections import deque
import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib("jet", buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = int(radius)
        self.thickness = int(thickness)

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i <= 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return frame

        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)

        max_i = len(self.buffer)
        for i, xy_i in enumerate(self.buffer):
            if len(xy_i) == 0:
                continue
            color = self.color_palette.by_idx(i)
            r = self.interpolate_radius(i, max_i)
            for center in xy_i:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=r,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return frame


class BallTracker:
    def __init__(self, buffer_size: int = 20):
        self.buffer = deque(maxlen=int(buffer_size))

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            self.buffer.append(np.empty((0, 2), dtype=np.float32))
            return detections

        xy = detections.get_anchors_coordinates(sv.Position.CENTER).astype(np.float32)
        self.buffer.append(xy)

        valid = [b for b in self.buffer if len(b) > 0]
        if not valid:
            return detections

        centroid = np.mean(np.concatenate(valid, axis=0), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = int(np.argmin(distances))
        return detections[[index]]

    def get_last_center(self):
        for b in reversed(self.buffer):
            if len(b) > 0:
                return b[0].astype(np.float32)
        return None

    def get_velocity(self):
        last = None
        prev = None
        for b in reversed(self.buffer):
            if len(b) > 0:
                if last is None:
                    last = b[0].astype(np.float32)
                else:
                    prev = b[0].astype(np.float32)
                    break
        if last is None or prev is None:
            return None
        return (last - prev).astype(np.float32)