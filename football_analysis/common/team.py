from typing import Generator, Iterable, List, TypeVar, Dict
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from transformers import SiglipVisionModel, SiglipImageProcessor
from collections import defaultdict, deque
import warnings

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(int(batch_size), 1)
    current_batch: List[V] = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = int(batch_size)
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.features_model.eval()
        self.processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, n_init="auto")

        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops_pil, self.batch_size)
        data = []

        use_amp = self.device.startswith("cuda")
        with torch.no_grad():
            for batch in batches:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.features_model(**inputs)
                else:
                    outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                data.append(embeddings.float().cpu().numpy())

        if not data:
            return np.array([])

        return np.concatenate(data, axis=0)

    def fit(self, crops: List[np.ndarray]) -> None:
        if len(crops) == 0:
            raise ValueError("Cannot fit with empty crops list")

        data = self.extract_features(crops)
        if len(data) == 0:
            raise ValueError("No features extracted from crops")

        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        if len(data) == 0:
            return np.array([])

        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


class TeamStabilizer:
    def __init__(self, history_size: int = 30, confidence_threshold: float = 0.7):
        self.history_size = int(history_size)
        self.confidence_threshold = float(confidence_threshold)
        self.team_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.stable_teams: Dict[int, int] = {}

    def update(self, tracker_ids: np.ndarray, predicted_teams: np.ndarray) -> np.ndarray:
        stabilized_teams = []

        for tracker_id, predicted_team in zip(tracker_ids, predicted_teams):
            tracker_id = int(tracker_id)
            predicted_team = int(predicted_team)
            self.team_history[tracker_id].append(predicted_team)

            history = list(self.team_history[tracker_id])

            if len(history) < 5:
                self.stable_teams[tracker_id] = predicted_team
                stabilized_teams.append(predicted_team)
                continue

            team_0_count = history.count(0)
            team_1_count = history.count(1)
            total = len(history)

            team_0_ratio = team_0_count / total
            team_1_ratio = team_1_count / total

            if team_0_ratio > self.confidence_threshold:
                stable_team = 0
            elif team_1_ratio > self.confidence_threshold:
                stable_team = 1
            elif tracker_id in self.stable_teams:
                stable_team = self.stable_teams[tracker_id]
            else:
                stable_team = 0 if team_0_ratio > team_1_ratio else 1

            self.stable_teams[tracker_id] = stable_team
            stabilized_teams.append(stable_team)

        return np.array(stabilized_teams, dtype=int)

    def get(self, tracker_ids: np.ndarray, default: int = 0) -> np.ndarray:
        out = []
        d = int(default)
        for tid in tracker_ids:
            tid = int(tid)
            out.append(int(self.stable_teams.get(tid, d)))
        return np.array(out, dtype=int)

    def cleanup_old_trackers(self, active_tracker_ids: set):
        all_tracker_ids = set(self.team_history.keys())
        inactive_ids = all_tracker_ids - set(active_tracker_ids)

        for tracker_id in inactive_ids:
            if tracker_id in self.team_history:
                del self.team_history[tracker_id]
            if tracker_id in self.stable_teams:
                del self.stable_teams[tracker_id]