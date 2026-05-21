import os
import re
import zipfile
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv


BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

PLAYER_IMGSZ = 960
PITCH_IMGSZ = 640

COLORS = ["#FF1493", "#00BFFF"]

FIT_STRIDE = 90
MAX_FIT_CROPS = 500
TEAM_UPDATE_BATCH = 48

PITCH_MARGIN_RATIO = 0.08
PITCH_CONF = 0.01
PITCH_CONF_THRESHOLDS = (0.70, 0.55, 0.40, 0.30)
PITCH_MIN_POINTS = 8
PITCH_MAX_POINTS = 24
PITCH_BORDER_MARGIN_RATIO = 0.03
PITCH_MIN_X_SPAN_RATIO = 0.35
PITCH_MIN_Y_SPAN_RATIO = 0.18
PITCH_RANSAC_REPROJ_THRESHOLD = 6.0

PLAYER_GROUND_ANCHOR_MIN_RATIO = 0.88
PLAYER_GROUND_ANCHOR_MAX_RATIO = 0.98
PLAYER_GROUND_ANCHOR_SLOPE = 0.36
GOALKEEPER_GROUND_ANCHOR_BONUS = 0.01
BALL_GROUND_ANCHOR_RATIO = 0.95


@st.cache_resource(show_spinner=False)
def _get_pitch_config():
    from football_analysis.configs.pitch_config import FootballPitchConfiguration
    return FootballPitchConfiguration()


@st.cache_resource(show_spinner=False)
def _get_annotators():
    ellipse = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        thickness=2,
    )
    label = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    return ellipse, label


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _enable_fast_torch():
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def load_models(player_path: str, pitch_path: str, device: str):
    _enable_fast_torch()
    from ultralytics import YOLO

    player_model = YOLO(player_path).to(device=device)
    pitch_model = YOLO(pitch_path).to(device=device)

    if str(device).startswith("cuda"):
        for model in (player_model, pitch_model):
            try:
                model.fuse()
            except Exception:
                pass
            try:
                if hasattr(model, "model") and model.model is not None:
                    model.model.half()
            except Exception:
                pass

    return player_model, pitch_model


def save_uploaded_file(uploaded_file, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def _natural_sort_key(value: str):
    value = os.path.basename(str(value)).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def generate_output_filename(original_filename: str, mode_prefix: str) -> str:
    name, ext = os.path.splitext(original_filename)
    return f"{name}_{mode_prefix}{ext}"


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    if len(detections) == 0:
        return []
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def _get_ground_anchor_points(detections: sv.Detections, frame_shape, default_class_id: Optional[int] = None) -> np.ndarray:
    if detections is None or len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)

    h_frame = float(frame_shape[0])
    xyxy = np.asarray(detections.xyxy, dtype=np.float32)
    class_ids = getattr(detections, "class_id", None)

    if class_ids is not None:
        class_ids = np.asarray(class_ids, dtype=np.int32)

    points = np.zeros((len(xyxy), 2), dtype=np.float32)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        bh = max(float(y2 - y1), 1.0)
        cls = int(class_ids[i]) if class_ids is not None else default_class_id

        if cls is None:
            cls = default_class_id

        if cls is not None and int(cls) < PLAYER_CLASS_ID:
            ratio = 0.98 - PLAYER_GROUND_ANCHOR_SLOPE * (bh / max(h_frame, 1.0)) + GOALKEEPER_GROUND_ANCHOR_BONUS
        else:
            ratio = 0.98 - PLAYER_GROUND_ANCHOR_SLOPE * (bh / max(h_frame, 1.0))

        ratio = float(np.clip(ratio, PLAYER_GROUND_ANCHOR_MIN_RATIO, PLAYER_GROUND_ANCHOR_MAX_RATIO))

        points[i, 0] = 0.5 * (x1 + x2)
        points[i, 1] = y1 + ratio * bh

    return points.astype(np.float32)


def _get_ball_ground_anchor_points(detections: sv.Detections) -> np.ndarray:
    if detections is None or len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xyxy = np.asarray(detections.xyxy, dtype=np.float32)
    points = np.zeros((len(xyxy), 2), dtype=np.float32)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        bh = max(float(y2 - y1), 1.0)
        points[i, 0] = 0.5 * (x1 + x2)
        points[i, 1] = y1 + BALL_GROUND_ANCHOR_RATIO * bh

    return points.astype(np.float32)


def resolve_goalkeepers_team_id(players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections) -> np.ndarray:
    if len(goalkeepers) == 0 or len(players) == 0:
        return np.array([])

    frame_h = max(
        float(
            np.max(
                [
                    goalkeepers.xyxy[:, 3].max() if len(goalkeepers) else 0,
                    players.xyxy[:, 3].max() if len(players) else 0,
                ]
            )
        ),
        1.0,
    )

    approx_frame_shape = (
        int(frame_h * 1.2),
        int(max(np.max(goalkeepers.xyxy[:, 2]) if len(goalkeepers) else 0, np.max(players.xyxy[:, 2]) if len(players) else 0, 1)),
        3,
    )

    goalkeepers_xy = _get_ground_anchor_points(goalkeepers, approx_frame_shape)
    players_xy = _get_ground_anchor_points(players, approx_frame_shape)

    team_0_players = players_xy[players_team_id == 0]
    team_1_players = players_xy[players_team_id == 1]

    if len(team_0_players) == 0 or len(team_1_players) == 0:
        default_team = 0 if len(team_0_players) > 0 else 1
        return np.array([default_team] * len(goalkeepers))

    team_0_centroid = team_0_players.mean(axis=0)
    team_1_centroid = team_1_players.mean(axis=0)

    result = []

    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        result.append(0 if dist_0 < dist_1 else 1)

    return np.array(result)


def _project_points(transformer, xy: np.ndarray):
    config = _get_pitch_config()

    if transformer is None or xy.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    transformed_xy = transformer.transform_points(points=xy.astype(np.float32))

    x_margin = config.length * PITCH_MARGIN_RATIO
    y_margin = config.width * PITCH_MARGIN_RATIO

    valid_mask = (
        np.isfinite(transformed_xy).all(axis=1)
        & (transformed_xy[:, 0] >= -x_margin)
        & (transformed_xy[:, 0] <= config.length + x_margin)
        & (transformed_xy[:, 1] >= -y_margin)
        & (transformed_xy[:, 1] <= config.width + y_margin)
    )

    return transformed_xy.astype(np.float32), valid_mask


def render_players_positioning(
    detections: sv.Detections,
    color_lookup: np.ndarray,
    transformer,
    ball_detections: Optional[sv.Detections] = None,
    frame_shape: Optional[tuple] = None,
) -> np.ndarray:
    from football_analysis.annotators.pitch_visualisation import draw_pitch, draw_points_on_pitch

    config = _get_pitch_config()

    if transformer is None or len(detections) == 0:
        return draw_pitch(config=config)

    if frame_shape is None:
        frame_shape = (720, 1280, 3)

    xy = _get_ground_anchor_points(detections, frame_shape).astype(np.float32)

    try:
        transformed_xy, valid_mask = _project_points(transformer, xy)
        players_positioning = draw_pitch(config=config)

        for i in range(len(COLORS)):
            mask = (color_lookup == i) & valid_mask
            if np.any(mask):
                players_positioning = draw_points_on_pitch(
                    config=config,
                    xy=transformed_xy[mask],
                    face_color=sv.Color.from_hex(COLORS[i]),
                    radius=20,
                    pitch=players_positioning,
                )

        if ball_detections is not None and len(ball_detections) > 0:
            ball_xy = _get_ball_ground_anchor_points(ball_detections).astype(np.float32)
            transformed_ball_xy, ball_valid = _project_points(transformer, ball_xy)

            if np.any(ball_valid):
                players_positioning = draw_points_on_pitch(
                    config=config,
                    xy=transformed_ball_xy[ball_valid],
                    face_color=sv.Color.from_hex("#FFFFFF"),
                    edge_color=sv.Color.from_hex("#000000"),
                    radius=15,
                    pitch=players_positioning,
                )

        return players_positioning
    except Exception:
        return draw_pitch(config=config)


def _get_target_orders(config):
    vertices = np.asarray(config.vertices, dtype=np.float32)
    orders = [("vertices", vertices)]
    labels = getattr(config, "labels", None)

    if labels:
        ordered = np.asarray([vertices[int(label) - 1] for label in labels], dtype=np.float32)
        orders.append(("labels", ordered))

    return orders


def _select_pitch_candidate_indices(boxes_xyxy: np.ndarray, frame_shape) -> np.ndarray:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.empty((0,), dtype=int)

    h, w = frame_shape[:2]
    frame_cx = w * 0.5
    frame_cy = h * 0.5

    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    centers_x = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    centers_y = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    dist = np.sqrt((centers_x - frame_cx) ** 2 + (centers_y - frame_cy) ** 2)
    score = areas - 0.35 * dist

    return np.argsort(score)[::-1]


def _get_pitch_predictions(result, frame_shape):
    try:
        keypoints = result.keypoints
        boxes = result.boxes

        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return []

        xy_all = keypoints.xy.detach().cpu().numpy().astype(np.float32)
        conf_all = None

        if keypoints.conf is not None:
            conf_all = keypoints.conf.detach().cpu().numpy().astype(np.float32)

        boxes_xyxy = None
        boxes_conf = None

        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == xy_all.shape[0]:
            boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            if boxes.conf is not None:
                boxes_conf = boxes.conf.detach().cpu().numpy().astype(np.float32)

        order = _select_pitch_candidate_indices(boxes_xyxy, frame_shape) if boxes_xyxy is not None else np.arange(xy_all.shape[0])
        candidates = []

        for idx in order.tolist():
            xy = xy_all[idx]
            conf = None if conf_all is None else conf_all[idx]
            bbox = None if boxes_xyxy is None else boxes_xyxy[idx]
            box_conf = None if boxes_conf is None else float(boxes_conf[idx])
            candidates.append((xy, conf, bbox, box_conf))

        return candidates
    except Exception:
        return []


def _diverse_topk(points: np.ndarray, conf: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.argsort(conf)[::-1]

    if len(idx) <= max_points:
        return idx

    selected = [int(idx[0])]
    remaining = idx[1:].tolist()

    while remaining and len(selected) < max_points:
        sel_pts = points[np.asarray(selected, dtype=int)]
        best_i = None
        best_score = -1.0

        for i in remaining:
            p = points[i]
            d = np.min(np.linalg.norm(sel_pts - p, axis=1))
            score = float(d) * 0.7 + float(conf[i]) * 100.0

            if score > best_score:
                best_score = score
                best_i = i

        selected.append(int(best_i))
        remaining.remove(best_i)

    return np.asarray(selected, dtype=int)


def _fit_homography_candidate(frame_shape, source_points, target_points):
    from football_analysis.common.view import ViewTransformer

    if len(source_points) < PITCH_MIN_POINTS:
        return None, float("inf"), None

    try:
        H, inliers = cv2.findHomography(source_points, target_points, cv2.RANSAC, PITCH_RANSAC_REPROJ_THRESHOLD)

        if H is None:
            return None, float("inf"), None

        if inliers is not None:
            mask = inliers.ravel().astype(bool)
            if np.count_nonzero(mask) >= 4:
                H_refined, _ = cv2.findHomography(source_points[mask], target_points[mask], 0)
                if H_refined is not None:
                    H = H_refined

        transformer = ViewTransformer.from_matrix(H.astype(np.float32))
        projected = transformer.transform_points(source_points.astype(np.float32))
        distances = np.linalg.norm(projected - target_points, axis=1)

        reproj_error_mean = float(np.mean(distances))
        reproj_error_median = float(np.median(distances))
        reproj_error_max = float(np.max(distances))

        h, w = frame_shape[:2]
        src_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst_corners = transformer.transform_points(src_corners)

        if not np.isfinite(dst_corners).all():
            return None, float("inf"), None

        config = _get_pitch_config()
        x_span = float(dst_corners[:, 0].max() - dst_corners[:, 0].min())
        y_span = float(dst_corners[:, 1].max() - dst_corners[:, 1].min())

        if x_span < config.length * PITCH_MIN_X_SPAN_RATIO or y_span < config.width * PITCH_MIN_Y_SPAN_RATIO:
            return None, float("inf"), None

        inlier_ratio = 1.0

        if inliers is not None and len(inliers) > 0:
            inlier_ratio = float(np.mean(inliers.ravel().astype(np.float32)))

        score = reproj_error_mean - 100.0 * inlier_ratio

        metrics = {
            "reprojection_error_mean_px": reproj_error_mean,
            "reprojection_error_median_px": reproj_error_median,
            "reprojection_error_max_px": reproj_error_max,
            "inlier_ratio": inlier_ratio,
            "keypoints_used": int(len(source_points)),
            "homography_score": float(score),
        }

        return transformer, score, metrics
    except Exception:
        return None, float("inf"), None

def _build_pitch_transformer(frame: np.ndarray, pitch_model):
    config = _get_pitch_config()
    result = pitch_model(frame, imgsz=PITCH_IMGSZ, conf=PITCH_CONF, verbose=False)[0]
    predictions = _get_pitch_predictions(result, frame.shape)

    if not predictions:
        return None, float("inf"), None, None

    target_orders = _get_target_orders(config)

    h, w = frame.shape[:2]
    margin_x = w * PITCH_BORDER_MARGIN_RATIO
    margin_y = h * PITCH_BORDER_MARGIN_RATIO

    best_transformer, best_score, best_order, best_metrics = None, float("inf"), None, None

    for source_points, kp_conf, bbox, box_conf in predictions[:3]:
        source_points = np.asarray(source_points, dtype=np.float32)

        if kp_conf is None:
            kp_conf = np.ones((len(source_points),), dtype=np.float32)
        else:
            kp_conf = np.nan_to_num(np.asarray(kp_conf, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        base_valid = np.isfinite(source_points).all(axis=1)
        base_valid &= (source_points[:, 0] >= 0) & (source_points[:, 0] <= (w - 1))
        base_valid &= (source_points[:, 1] >= 0) & (source_points[:, 1] <= (h - 1))

        edge_mask = (
            (source_points[:, 0] <= margin_x)
            | (source_points[:, 0] >= (w - 1 - margin_x))
            | (source_points[:, 1] <= margin_y)
            | (source_points[:, 1] >= (h - 1 - margin_y))
        )

        for order_name, target_points in target_orders:
            for threshold in PITCH_CONF_THRESHOLDS:
                valid = base_valid & (kp_conf >= threshold)
                valid &= (~edge_mask) | (kp_conf >= max(0.8, threshold + 0.15))

                if np.count_nonzero(valid) < PITCH_MIN_POINTS:
                    continue

                base_idx = np.where(valid)[0]
                selected = _diverse_topk(source_points[base_idx], kp_conf[base_idx], PITCH_MAX_POINTS)
                idx = base_idx[selected]

                transformer, score, metrics = _fit_homography_candidate(frame.shape, source_points[idx], target_points[idx])

                if transformer is not None and score < best_score:
                    best_transformer, best_score, best_order, best_metrics = transformer, score, order_name, metrics

            fallback = base_valid & (kp_conf >= 0.20)

            if np.count_nonzero(fallback) >= PITCH_MIN_POINTS:
                base_idx = np.where(fallback)[0]
                selected = _diverse_topk(source_points[base_idx], kp_conf[base_idx], PITCH_MAX_POINTS)
                idx = base_idx[selected]

                transformer, score, metrics = _fit_homography_candidate(frame.shape, source_points[idx], target_points[idx])

                if transformer is not None and score < best_score:
                    best_transformer, best_score, best_order, best_metrics = transformer, score, order_name, metrics

    return best_transformer, best_score, best_order, best_metrics


class CameraTracker:
    def __init__(self):
        self.prev_gray = None
        self.current_matrix = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=20, blockSize=7)

    def initialize(self, frame, initial_matrix):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_matrix = np.asarray(initial_matrix, dtype=np.float32).copy()

    def update(self, frame, detections):
        if self.prev_gray is None or self.current_matrix is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.ones(gray.shape, dtype=np.uint8) * 255

        if detections is not None and len(detections) > 0:
            xyxy = detections.xyxy.astype(int)
            for x1, y1, x2, y2 in xyxy:
                cv2.rectangle(
                    mask,
                    (max(0, x1 - 15), max(0, y1 - 15)),
                    (min(gray.shape[1], x2 + 15), min(gray.shape[0], y2 + 15)),
                    0,
                    -1,
                )

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)

        if p0 is not None and len(p0) > 10:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if len(good_new) >= 4:
                    M, inliers = cv2.estimateAffinePartial2D(
                        good_new,
                        good_old,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                    )

                    if M is not None:
                        H_step = np.eye(3, dtype=np.float32)
                        H_step[0:2, :] = M
                        self.current_matrix = self.current_matrix @ H_step
                        self.current_matrix /= self.current_matrix[2, 2]

        self.prev_gray = gray

        from football_analysis.common.view import ViewTransformer
        return ViewTransformer.from_matrix(self.current_matrix)


def _sample_team_fit_crops(video_path: str, player_model) -> List[np.ndarray]:
    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=FIT_STRIDE)
    crops: List[np.ndarray] = []

    for frame in frame_generator:
        result = player_model(frame, imgsz=PLAYER_IMGSZ, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        frame_crops = get_crops(frame, player_detections)

        if frame_crops:
            need = MAX_FIT_CROPS - len(crops)
            crops.extend(frame_crops[:need])

            if len(crops) >= MAX_FIT_CROPS:
                break

    return crops


def _init_video_writer(target_video_path: str, video_info: sv.VideoInfo):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(target_video_path, fourcc, video_info.fps, (video_info.width, video_info.height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"x264")
        out = cv2.VideoWriter(target_video_path, fourcc, video_info.fps, (video_info.width, video_info.height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(target_video_path, fourcc, video_info.fps, (video_info.width, video_info.height))

    return out


def process_players_positioning_mode(source_video_path: str, target_video_path: str, player_model, pitch_model, progress_bar):
    from football_analysis.common.ball import BallTracker, BallAnnotator
    from football_analysis.common.team import TeamClassifier, TeamStabilizer

    crops = _sample_team_fit_crops(source_video_path, player_model)

    if len(crops) == 0:
        raise ValueError("У відео не виявлено гравців")

    device = "cuda" if _torch_cuda_available() else "cpu"

    team_classifier = TeamClassifier(device=device, batch_size=48)
    team_classifier.fit(crops)

    team_stabilizer = TeamStabilizer(history_size=30, confidence_threshold=0.7)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=30)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = video_info.total_frames

    out = _init_video_writer(target_video_path, video_info)

    if not out.isOpened():
        raise RuntimeError("Не вдалося ініціалізувати запис відео")

    pending_ids: List[int] = []
    pending_crops: List[np.ndarray] = []

    ellipse_annotator, label_annotator = _get_annotators()

    camera_tracker = CameraTracker()
    current_transformer = None
    homography_metrics_rows = []
    successful_transform_frames = 0

    try:
        for idx, frame in enumerate(frame_generator):
            result = player_model(frame, imgsz=PLAYER_IMGSZ, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            ball_detections_raw = sv.Detections.from_ultralytics(result)
            ball_detections_raw = ball_detections_raw[ball_detections_raw.class_id == BALL_CLASS_ID]
            ball_detections = ball_tracker.update(ball_detections_raw)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            player_ids = players.tracker_id if players.tracker_id is not None else np.array([], dtype=int)

            if len(players) > 0 and players.tracker_id is not None:
                crops_now = get_crops(frame, players)

                for tid, crop in zip(player_ids.tolist(), crops_now):
                    tid = int(tid)
                    if tid not in team_stabilizer.stable_teams:
                        pending_ids.append(tid)
                        pending_crops.append(crop)

                if len(pending_crops) >= TEAM_UPDATE_BATCH:
                    preds = team_classifier.predict(pending_crops[:TEAM_UPDATE_BATCH])
                    ids_np = np.array(pending_ids[:TEAM_UPDATE_BATCH], dtype=int)
                    team_stabilizer.update(ids_np, preds)
                    pending_ids = pending_ids[TEAM_UPDATE_BATCH:]
                    pending_crops = pending_crops[TEAM_UPDATE_BATCH:]

                players_team_id = team_stabilizer.get(player_ids, default=0)
            else:
                players_team_id = np.array([], dtype=int)

            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

            detections_all = sv.Detections.merge([players, goalkeepers])

            homography_metrics = None
            homography_source = "tracking"

            if camera_tracker.current_matrix is None:
                transformer, score, _, homography_metrics = _build_pitch_transformer(frame, pitch_model)

                if transformer is not None:
                    camera_tracker.initialize(frame, transformer.m)
                    current_transformer = transformer
                    homography_source = "YOLO26-pose"
            else:
                current_transformer = camera_tracker.update(frame, detections_all)

            if current_transformer is not None:
                successful_transform_frames += 1

            if homography_metrics is not None:
                homography_metrics_rows.append(homography_metrics)

            color_lookup = np.array(players_team_id.tolist() + goalkeepers_team_id.tolist(), dtype=int)
            labels = [str(tracker_id) for tracker_id in detections_all.tracker_id] if detections_all.tracker_id is not None else []

            annotated_frame = frame.copy()
            annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
            annotated_frame = ellipse_annotator.annotate(annotated_frame, detections_all, custom_color_lookup=color_lookup)

            if labels:
                annotated_frame = label_annotator.annotate(
                    annotated_frame,
                    detections_all,
                    labels,
                    custom_color_lookup=color_lookup,
                )

            h, w, _ = frame.shape
            players_positioning = render_players_positioning(
                detections_all,
                color_lookup,
                current_transformer,
                ball_detections,
                frame.shape,
            )
            players_positioning = sv.resize_image(players_positioning, (w // 2, h // 2))
            players_positioning_h, players_positioning_w, _ = players_positioning.shape

            rect = sv.Rect(
                x=w // 2 - players_positioning_w // 2,
                y=h - players_positioning_h,
                width=players_positioning_w,
                height=players_positioning_h,
            )

            annotated_frame = sv.draw_image(annotated_frame, players_positioning, opacity=0.5, rect=rect)
            out.write(annotated_frame)

            if total_frames:
                progress_bar.progress((idx + 1) / total_frames)

            if idx % 120 == 0 and detections_all.tracker_id is not None:
                active_ids = set(map(int, detections_all.tracker_id.tolist()))
                team_stabilizer.cleanup_old_trackers(active_ids)
    finally:
        out.release()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = max(int(video_info.total_frames), 1)

    successful_frame_ratio = successful_transform_frames / total_frames if total_frames > 0 else 0.0

    if homography_metrics_rows:
        mean_reproj_error = float(np.mean([row["reprojection_error_mean_px"] for row in homography_metrics_rows]))
        median_reproj_error = float(np.mean([row["reprojection_error_median_px"] for row in homography_metrics_rows]))
        mean_inlier_ratio = float(np.mean([row["inlier_ratio"] for row in homography_metrics_rows]))
        mean_keypoints_used = float(np.mean([row["keypoints_used"] for row in homography_metrics_rows]))
    else:
        mean_reproj_error = 0.0
        median_reproj_error = 0.0
        mean_inlier_ratio = 0.0
        mean_keypoints_used = 0.0

    metrics_rows = [
        {"Показник": "Усього кадрів відеофрагмента", "Значення": int(total_frames)},
        {"Показник": "Кадрів з успішною просторовою трансформацією", "Значення": int(successful_transform_frames)},
        {"Показник": "Частка кадрів з успішною трансформацією", "Значення": round(float(successful_frame_ratio), 3)},
        {"Показник": "Середня похибка репроєкції", "Значення": round(mean_reproj_error, 3)},
        {"Показник": "Медіанна похибка репроєкції", "Значення": round(median_reproj_error, 3)},
        {"Показник": "Середня частка узгоджених ключових точок", "Значення": round(mean_inlier_ratio, 3)},
        {"Показник": "Середня кількість ключових точок для побудови гомографії", "Значення": round(mean_keypoints_used, 1)},
    ]

    return pd.DataFrame(metrics_rows)


def _save_dataset_uploads(uploaded_files, suffixes):
    root = tempfile.mkdtemp()

    if not uploaded_files:
        return root

    for uploaded_file in uploaded_files:
        filename = os.path.basename(uploaded_file.name)
        lower = filename.lower()

        if lower.endswith(".zip"):
            zip_path = os.path.join(root, filename)

            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(root)
            finally:
                try:
                    os.remove(zip_path)
                except Exception:
                    pass
        elif lower.endswith(suffixes):
            target_path = os.path.join(root, filename)

            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())

    return root


def _collect_files(root: str, suffixes):
    files = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(suffixes):
                files.append(os.path.join(dirpath, filename))

    return sorted(files, key=_natural_sort_key)


def _match_images_and_labels(image_files, label_files):
    labels_by_stem = {os.path.splitext(os.path.basename(path))[0]: path for path in label_files}
    pairs = []
    missing = []

    for image_path in image_files:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = labels_by_stem.get(stem)

        if label_path is None:
            missing.append(os.path.basename(image_path))
        else:
            pairs.append((image_path, label_path))

    if not pairs and len(image_files) == len(label_files):
        pairs = list(zip(sorted(image_files, key=_natural_sort_key), sorted(label_files, key=_natural_sort_key)))
        missing = []

    return pairs, missing


def _read_yolo_boxes(label_path: str, image_shape):
    h, w = image_shape[:2]
    boxes = []
    class_ids = []

    if not os.path.exists(label_path):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
            except Exception:
                continue

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls)

    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.asarray(boxes, dtype=np.float32), np.asarray(class_ids, dtype=np.int32)


def _read_yolo_pose(label_path: str, image_shape):
    h, w = image_shape[:2]
    boxes = []
    class_ids = []
    keypoints = []
    visibility = []

    if not os.path.exists(label_path):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 0, 2), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 8:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                kp_raw = [float(v) for v in parts[5:]]
            except Exception:
                continue

            if len(kp_raw) < 3:
                continue

            usable = (len(kp_raw) // 3) * 3
            kp_raw = kp_raw[:usable]
            kp_arr = np.asarray(kp_raw, dtype=np.float32).reshape(-1, 3)

            kp_xy = kp_arr[:, :2].copy()
            kp_xy[:, 0] *= w
            kp_xy[:, 1] *= h
            kp_v = kp_arr[:, 2].copy()

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls)
            keypoints.append(kp_xy)
            visibility.append(kp_v)

    if not boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 0, 2), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )

    min_kpts = min(len(k) for k in keypoints)
    keypoints = [k[:min_kpts] for k in keypoints]
    visibility = [v[:min_kpts] for v in visibility]

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(class_ids, dtype=np.int32),
        np.asarray(keypoints, dtype=np.float32),
        np.asarray(visibility, dtype=np.float32),
    )


def _box_iou_matrix(a: np.ndarray, b: np.ndarray):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter = inter_w * inter_h

    area_a = np.maximum((ax2 - ax1) * (ay2 - ay1), 0)
    area_b = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0)
    union = area_a + area_b - inter

    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _average_precision(scores, matches, total_gt):
    if total_gt <= 0 or len(scores) == 0:
        return 0.0

    scores = np.asarray(scores, dtype=np.float32)
    matches = np.asarray(matches, dtype=np.float32)

    order = np.argsort(scores)[::-1]
    matches = matches[order]

    tp = np.cumsum(matches)
    fp = np.cumsum(1.0 - matches)

    recall = tp / max(float(total_gt), 1e-9)
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])

    return float(ap)


def _empty_class_stats(class_ids):
    return {
        int(cls): {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "support": 0,
            "scores": [],
            "matches": [],
        }
        for cls in class_ids
    }


def _update_detection_stats(stats, gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf, class_ids, iou_threshold):
    for cls in class_ids:
        cls = int(cls)
        gt_mask = gt_cls == cls
        pred_mask = pred_cls == cls

        cls_gt = gt_boxes[gt_mask]
        cls_pred = pred_boxes[pred_mask]
        cls_conf = pred_conf[pred_mask]

        stats[cls]["support"] += int(len(cls_gt))

        if len(cls_gt) == 0 and len(cls_pred) == 0:
            continue

        if len(cls_gt) == 0:
            stats[cls]["fp"] += int(len(cls_pred))
            for conf in cls_conf:
                stats[cls]["scores"].append(float(conf))
                stats[cls]["matches"].append(0)
            continue

        if len(cls_pred) == 0:
            stats[cls]["fn"] += int(len(cls_gt))
            continue

        order = np.argsort(cls_conf)[::-1]
        cls_pred_ordered = cls_pred[order]
        cls_conf_ordered = cls_conf[order]
        ious = _box_iou_matrix(cls_pred_ordered, cls_gt)

        matched_gt = set()
        tp = 0
        fp = 0

        for row in range(ious.shape[0]):
            best_col = int(np.argmax(ious[row]))
            best_iou = float(ious[row, best_col])
            conf = float(cls_conf_ordered[row])

            if best_iou >= iou_threshold and best_col not in matched_gt:
                matched_gt.add(best_col)
                tp += 1
                stats[cls]["scores"].append(conf)
                stats[cls]["matches"].append(1)
            else:
                fp += 1
                stats[cls]["scores"].append(conf)
                stats[cls]["matches"].append(0)

        fn = len(cls_gt) - tp

        stats[cls]["tp"] += int(tp)
        stats[cls]["fp"] += int(fp)
        stats[cls]["fn"] += int(fn)


def _precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _stats_to_rows(stats, class_names, macro_label="Макро-середнє значення"):
    rows = []
    precision_values = []
    recall_values = []
    f1_values = []
    map_values = []
    total_support = 0

    for cls, name in class_names.items():
        item = stats[int(cls)]
        precision, recall, f1 = _precision_recall_f1(item["tp"], item["fp"], item["fn"])
        ap = _average_precision(item["scores"], item["matches"], item["support"])
        support = item["support"]

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        map_values.append(ap)
        total_support += support

        rows.append(
            {
                "Категорія": name,
                "Precision": round(float(precision), 3),
                "Recall": round(float(recall), 3),
                "F1-Score": round(float(f1), 3),
                "mAP": round(float(ap), 3),
                "Кількість зразків": int(support),
            }
        )

    rows.append(
        {
            "Категорія": macro_label,
            "Precision": round(float(np.mean(precision_values)) if precision_values else 0.0, 3),
            "Recall": round(float(np.mean(recall_values)) if recall_values else 0.0, 3),
            "F1-Score": round(float(np.mean(f1_values)) if f1_values else 0.0, 3),
            "mAP": round(float(np.mean(map_values)) if map_values else 0.0, 3),
            "Кількість зразків": int(total_support),
        }
    )

    return rows


def _evaluate_image_dataset(model, image_files, label_files, class_names, imgsz, conf_threshold, iou_threshold, progress_bar=None):
    pairs, missing = _match_images_and_labels(image_files, label_files)
    stats = _empty_class_stats(class_names.keys())
    processed = 0
    skipped = 0

    for idx, (image_path, label_path) in enumerate(pairs):
        image = cv2.imread(image_path)

        if image is None:
            skipped += 1
            continue

        gt_boxes, gt_cls = _read_yolo_boxes(label_path, image.shape)
        result = model(image, imgsz=int(imgsz), conf=float(conf_threshold), verbose=False)[0]

        if result.boxes is not None and result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            pred_cls = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
            pred_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_cls = np.empty((0,), dtype=np.int32)
            pred_conf = np.empty((0,), dtype=np.float32)

        _update_detection_stats(
            stats,
            gt_boxes,
            gt_cls,
            pred_boxes,
            pred_cls,
            pred_conf,
            list(class_names.keys()),
            float(iou_threshold),
        )

        processed += 1

        if progress_bar is not None and len(pairs) > 0:
            progress_bar.progress((idx + 1) / len(pairs))

    return _stats_to_rows(stats, class_names), processed, skipped, missing


def _extract_pose_predictions(result):
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 0, 2), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )

    pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    pred_cls = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    pred_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    if result.keypoints is None or result.keypoints.xy is None:
        return pred_boxes, pred_cls, pred_conf, np.empty((len(pred_boxes), 0, 2), dtype=np.float32), np.empty((len(pred_boxes), 0), dtype=np.float32)

    pred_kp = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)

    if result.keypoints.conf is not None:
        pred_kp_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
    else:
        pred_kp_conf = np.ones((pred_kp.shape[0], pred_kp.shape[1]), dtype=np.float32)

    return pred_boxes, pred_cls, pred_conf, pred_kp, pred_kp_conf


def _evaluate_pose_dataset(model, image_files, label_files, imgsz, conf_threshold, iou_threshold, kp_conf_threshold, kp_distance_ratio, progress_bar=None):
    pairs, missing = _match_images_and_labels(image_files, label_files)
    processed = 0
    skipped = 0

    field_tp = 0
    field_fp = 0
    field_fn = 0
    field_count = 0

    keypoint_tp = 0
    keypoint_fp = 0
    keypoint_fn = 0
    keypoint_count = 0

    field_scores = []
    field_matches = []

    keypoint_scores = []
    keypoint_matches = []

    for idx, (image_path, label_path) in enumerate(pairs):
        image = cv2.imread(image_path)

        if image is None:
            skipped += 1
            continue

        image_h, image_w = image.shape[:2]

        gt_boxes, gt_cls, gt_kp, gt_vis = _read_yolo_pose(label_path, image.shape)
        result = model(image, imgsz=int(imgsz), conf=float(conf_threshold), verbose=False)[0]
        pred_boxes, pred_cls, pred_conf, pred_kp, pred_kp_conf = _extract_pose_predictions(result)

        gt_mask = gt_cls == 0
        pred_mask = pred_cls == 0

        gt_boxes_cls = gt_boxes[gt_mask]
        pred_boxes_cls = pred_boxes[pred_mask]
        pred_conf_cls = pred_conf[pred_mask]

        gt_kp_cls = gt_kp[gt_mask] if len(gt_kp) else np.empty((0, 0, 2), dtype=np.float32)
        gt_vis_cls = gt_vis[gt_mask] if len(gt_vis) else np.empty((0, 0), dtype=np.float32)
        pred_kp_cls = pred_kp[pred_mask] if len(pred_kp) else np.empty((0, 0, 2), dtype=np.float32)
        pred_kp_conf_cls = pred_kp_conf[pred_mask] if len(pred_kp_conf) else np.empty((0, 0), dtype=np.float32)

        field_count += int(len(gt_boxes_cls))

        if len(gt_boxes_cls) == 0 and len(pred_boxes_cls) == 0:
            processed += 1
            if progress_bar is not None and len(pairs) > 0:
                progress_bar.progress((idx + 1) / len(pairs))
            continue

        if len(gt_boxes_cls) == 0:
            field_fp += int(len(pred_boxes_cls))
            for conf in pred_conf_cls:
                field_scores.append(float(conf))
                field_matches.append(0)

            if len(pred_kp_conf_cls):
                fp_count = int(np.sum(pred_kp_conf_cls >= kp_conf_threshold))
                keypoint_fp += fp_count
                for score in pred_kp_conf_cls[pred_kp_conf_cls >= kp_conf_threshold].ravel():
                    keypoint_scores.append(float(score))
                    keypoint_matches.append(0)

            processed += 1
            if progress_bar is not None and len(pairs) > 0:
                progress_bar.progress((idx + 1) / len(pairs))
            continue

        if len(pred_boxes_cls) == 0:
            field_fn += int(len(gt_boxes_cls))
            if len(gt_vis_cls):
                visible_count = int(np.sum(gt_vis_cls > 0))
                keypoint_fn += visible_count
                keypoint_count += visible_count
            processed += 1
            if progress_bar is not None and len(pairs) > 0:
                progress_bar.progress((idx + 1) / len(pairs))
            continue

        order = np.argsort(pred_conf_cls)[::-1]
        pred_boxes_ordered = pred_boxes_cls[order]
        pred_conf_ordered = pred_conf_cls[order]
        pred_kp_ordered = pred_kp_cls[order] if len(pred_kp_cls) else np.empty((0, 0, 2), dtype=np.float32)
        pred_kp_conf_ordered = pred_kp_conf_cls[order] if len(pred_kp_conf_cls) else np.empty((0, 0), dtype=np.float32)

        ious = _box_iou_matrix(pred_boxes_ordered, gt_boxes_cls)
        matched_gt = set()

        for row in range(ious.shape[0]):
            best_col = int(np.argmax(ious[row]))
            best_iou = float(ious[row, best_col])
            box_score = float(pred_conf_ordered[row])

            if best_iou >= iou_threshold and best_col not in matched_gt:
                matched_gt.add(best_col)
                field_tp += 1
                field_scores.append(box_score)
                field_matches.append(1)

                if len(gt_kp_cls):
                    gt_points = gt_kp_cls[best_col]
                    gt_visible = gt_vis_cls[best_col] > 0
                    visible_gt_count = int(np.sum(gt_visible))
                    keypoint_count += visible_gt_count

                    if visible_gt_count > 0:
                        if len(pred_kp_ordered) == 0 or row >= len(pred_kp_ordered):
                            keypoint_fn += visible_gt_count
                        else:
                            pred_points = pred_kp_ordered[row]

                            if len(pred_kp_conf_ordered) and row < len(pred_kp_conf_ordered):
                                pred_visible = pred_kp_conf_ordered[row] >= kp_conf_threshold
                                pred_scores = pred_kp_conf_ordered[row]
                            else:
                                pred_visible = np.ones((len(pred_points),), dtype=bool)
                                pred_scores = np.ones((len(pred_points),), dtype=np.float32)

                            count = min(len(gt_points), len(pred_points), len(gt_visible), len(pred_visible), len(pred_scores))

                            if count == 0:
                                keypoint_fn += visible_gt_count
                            else:
                                gt_points = gt_points[:count]
                                pred_points = pred_points[:count]
                                gt_visible = gt_visible[:count]
                                pred_visible = pred_visible[:count]
                                pred_scores = pred_scores[:count]

                                diagonal = float(np.sqrt(image_w ** 2 + image_h ** 2))
                                distance_threshold = max(diagonal * float(kp_distance_ratio), 1.0)
                                distances = np.linalg.norm(pred_points - gt_points, axis=1)

                                correct = gt_visible & pred_visible & (distances <= distance_threshold)
                                wrong = gt_visible & pred_visible & (distances > distance_threshold)
                                not_predicted = gt_visible & (~pred_visible)

                                keypoint_tp += int(np.sum(correct))
                                keypoint_fp += int(np.sum(wrong))
                                keypoint_fn += int(np.sum(wrong) + np.sum(not_predicted))

                                for kp_score, is_correct, is_visible in zip(pred_scores, correct, gt_visible):
                                    if bool(is_visible):
                                        keypoint_scores.append(float(kp_score))
                                        keypoint_matches.append(1 if bool(is_correct) else 0)

                                if int(np.sum(gt_visible)) < visible_gt_count:
                                    keypoint_fn += visible_gt_count - int(np.sum(gt_visible))
            else:
                field_fp += 1
                field_scores.append(box_score)
                field_matches.append(0)

                if len(pred_kp_conf_ordered) and row < len(pred_kp_conf_ordered):
                    fp_count = int(np.sum(pred_kp_conf_ordered[row] >= kp_conf_threshold))
                    keypoint_fp += fp_count
                    for score in pred_kp_conf_ordered[row][pred_kp_conf_ordered[row] >= kp_conf_threshold]:
                        keypoint_scores.append(float(score))
                        keypoint_matches.append(0)

        field_fn += int(len(gt_boxes_cls) - len(matched_gt))

        if len(gt_kp_cls):
            for gt_idx in range(len(gt_kp_cls)):
                if gt_idx not in matched_gt:
                    visible_count = int(np.sum(gt_vis_cls[gt_idx] > 0))
                    keypoint_fn += visible_count
                    keypoint_count += visible_count

        processed += 1

        if progress_bar is not None and len(pairs) > 0:
            progress_bar.progress((idx + 1) / len(pairs))

    field_precision, field_recall, field_f1 = _precision_recall_f1(field_tp, field_fp, field_fn)
    keypoint_precision, keypoint_recall, keypoint_f1 = _precision_recall_f1(keypoint_tp, keypoint_fp, keypoint_fn)

    field_ap = _average_precision(field_scores, field_matches, field_count)
    keypoint_ap = _average_precision(keypoint_scores, keypoint_matches, keypoint_count)

    macro_precision = float(np.mean([field_precision, keypoint_precision]))
    macro_recall = float(np.mean([field_recall, keypoint_recall]))
    macro_f1 = float(np.mean([field_f1, keypoint_f1]))
    macro_ap = float(np.mean([field_ap, keypoint_ap]))

    rows = [
        {
            "Компонент оцінювання": "Область футбольного поля",
            "Precision": round(float(field_precision), 3),
            "Recall": round(float(field_recall), 3),
            "F1-Score": round(float(field_f1), 3),
            "mAP": round(float(field_ap), 3),
            "Кількість елементів": int(field_count),
        },
        {
            "Компонент оцінювання": "Ключові точки футбольного поля",
            "Precision": round(float(keypoint_precision), 3),
            "Recall": round(float(keypoint_recall), 3),
            "F1-Score": round(float(keypoint_f1), 3),
            "mAP": round(float(keypoint_ap), 3),
            "Кількість елементів": int(keypoint_count),
        },
        {
            "Компонент оцінювання": "Макро-середнє значення",
            "Precision": round(macro_precision, 3),
            "Recall": round(macro_recall, 3),
            "F1-Score": round(macro_f1, 3),
            "mAP": round(macro_ap, 3),
            "Кількість елементів": int(field_count + keypoint_count),
        },
    ]

    return rows, processed, skipped, missing


def _prepare_yolo_dataframe(rows):
    df = pd.DataFrame(rows)
    return df.rename(
        columns={
            "Категорія": "Компонент оцінювання",
            "F1-Score": "F1-score",
            "Кількість зразків": "Кількість елементів",
        }
    )


def _prepare_pose_dataframe(rows):
    df = pd.DataFrame(rows)
    return df.rename(columns={"F1-Score": "F1-score"})


def _build_download_csv(df, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Завантажити CSV", data=csv, file_name=filename, mime="text/csv", width="stretch")


def show_testing_page(device: str):
    st.title("Тестування моделей")

    if not st.session_state.get("models_loaded", False):
        st.warning("Спочатку завантажте моделі на головній сторінці")
        return

    player_path = st.session_state["player_path"]
    pitch_path = st.session_state["pitch_path"]

    player_model, pitch_model = load_models(player_path, pitch_path, device)

    conf_threshold = st.slider("Поріг впевненості", min_value=0.001, max_value=0.9, value=0.25, step=0.001, format="%.3f")
    iou_threshold = st.slider("Поріг IoU для області поля та обмежувальних рамок", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    kp_conf_threshold = st.slider("Поріг впевненості ключових точок", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    kp_distance_ratio = st.slider(
        "Допустима похибка ключової точки відносно діагоналі кадру",
        min_value=0.001,
        max_value=0.05,
        value=0.015,
        step=0.001,
        format="%.3f",
    )

    st.subheader("YOLO26")

    yolo_images = st.file_uploader(
        "Завантажте тестові зображення YOLO26 або ZIP-архів",
        type=["jpg", "jpeg", "png", "bmp", "webp", "zip"],
        accept_multiple_files=True,
        key="yolo26_test_images",
    )

    yolo_labels = st.file_uploader(
        "Завантажте тестові анотації YOLO26 або ZIP-архів",
        type=["txt", "zip"],
        accept_multiple_files=True,
        key="yolo26_test_labels",
    )

    st.subheader("YOLO26-pose")

    pose_images = st.file_uploader(
        "Завантажте тестові зображення YOLO26-pose або ZIP-архів",
        type=["jpg", "jpeg", "png", "bmp", "webp", "zip"],
        accept_multiple_files=True,
        key="yolo26_pose_test_images",
    )

    pose_labels = st.file_uploader(
        "Завантажте тестові анотації YOLO26-pose або ZIP-архів",
        type=["txt", "zip"],
        accept_multiple_files=True,
        key="yolo26_pose_test_labels",
    )

    if st.button("Почати тестування", width="stretch", type="primary"):
        if not yolo_images or not yolo_labels or not pose_images or not pose_labels:
            st.error("Завантажте зображення та анотації для обох моделей")
            return

        with st.spinner("Виконується тестування моделей..."):
            yolo_images_root = _save_dataset_uploads(yolo_images, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            yolo_labels_root = _save_dataset_uploads(yolo_labels, (".txt",))

            pose_images_root = _save_dataset_uploads(pose_images, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            pose_labels_root = _save_dataset_uploads(pose_labels, (".txt",))

            yolo_image_files = _collect_files(yolo_images_root, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            yolo_label_files = _collect_files(yolo_labels_root, (".txt",))

            pose_image_files = _collect_files(pose_images_root, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            pose_label_files = _collect_files(pose_labels_root, (".txt",))

            if len(yolo_image_files) == 0 or len(yolo_label_files) == 0:
                st.error("Зображення або анотації YOLO26 не знайдено")
                return

            if len(pose_image_files) == 0 or len(pose_label_files) == 0:
                st.error("Зображення або анотації YOLO26-pose не знайдено")
                return

            yolo_rows, yolo_processed, yolo_skipped, yolo_missing = _evaluate_image_dataset(
                player_model,
                yolo_image_files,
                yolo_label_files,
                {2: "Гравець", 1: "Воротар", 0: "М’яч"},
                PLAYER_IMGSZ,
                conf_threshold,
                iou_threshold,
                None,
            )

            pose_rows, pose_processed, pose_skipped, pose_missing = _evaluate_pose_dataset(
                pitch_model,
                pose_image_files,
                pose_label_files,
                PITCH_IMGSZ,
                conf_threshold,
                iou_threshold,
                kp_conf_threshold,
                kp_distance_ratio,
                None,
            )

            st.subheader("Зведені результати оцінювання якості детекції об’єктів моделлю YOLO26")

            yolo_df = _prepare_yolo_dataframe(yolo_rows)
            st.dataframe(yolo_df, width="stretch", hide_index=True)

            _build_download_csv(yolo_df, "yolo26_test_metrics.csv")
            st.caption(f"Оброблено зображень: {yolo_processed}. Пропущено зображень: {yolo_skipped}.")

            if len(yolo_missing) > 0:
                st.info(f"Додатково знайдено {len(yolo_missing)} зображень без відповідних анотацій. В оцінювання включено лише знайдені пари зображення–анотація.")

            st.subheader("Зведені результати оцінювання локалізації футбольного поля моделлю YOLO26-pose")

            pose_df = _prepare_pose_dataframe(pose_rows)
            st.dataframe(pose_df, width="stretch", hide_index=True)

            _build_download_csv(pose_df, "yolo26_pose_test_metrics.csv")
            st.caption(f"Оброблено зображень: {pose_processed}. Пропущено зображень: {pose_skipped}.")

            if len(pose_missing) > 0:
                st.info(f"Додатково знайдено {len(pose_missing)} зображень без відповідних анотацій. В оцінювання включено лише знайдені пари зображення–анотація.")


def show_home_page():
    st.title("Головна сторінка")

    if st.session_state.get("models_loaded", False):
        st.success("Моделі вже завантажено")
        st.info("Оберіть режим роботи в бічній панелі")
        return

    player_model_file = st.file_uploader("Модель детекції гравців", type=["pt"], key="player_model")
    pitch_model_file = st.file_uploader("Модель локалізації поля", type=["pt"], key="pitch_model")

    if player_model_file and pitch_model_file:
        if st.button("Завантажити моделі", width="stretch", type="primary"):
            with st.spinner("Виконується завантаження моделей..."):
                try:
                    player_path = save_uploaded_file(player_model_file, ".pt")
                    pitch_path = save_uploaded_file(pitch_model_file, ".pt")
                    device = "cuda" if _torch_cuda_available() else "cpu"

                    load_models(player_path, pitch_path, device)

                    st.session_state["models_loaded"] = True
                    st.session_state["player_path"] = player_path
                    st.session_state["pitch_path"] = pitch_path

                    st.success("Моделі успішно завантажено")
                    st.rerun()
                except Exception as e:
                    st.error(f"Помилка завантаження: {str(e)}")
    else:
        st.info("Завантажте обидві моделі для початку роботи")


def show_players_positioning_page(device: str):
    st.title("Визначення просторового положення гравців")

    if not st.session_state.get("models_loaded", False):
        st.warning("Спочатку завантажте моделі на головній сторінці")
        return

    uploaded_file = st.file_uploader("Завантажте відео", type=["mp4", "avi", "mov"], key="players_positioning_uploader")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        st.subheader("Початкове відео")
        st.video(input_path)

        if st.button("Почати обробку", key="players_positioning_process"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
                output_path = tmp_output.name

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Виконується обробка відео...")

            try:
                player_path = st.session_state["player_path"]
                pitch_path = st.session_state["pitch_path"]

                player_model, pitch_model = load_models(player_path, pitch_path, device)

                spatial_metrics_df = process_players_positioning_mode(
                    input_path,
                    output_path,
                    player_model,
                    pitch_model,
                    progress_bar,
                )

                status_text.text("Обробку завершено")
                st.subheader("Результат аналізу")
                st.video(output_path)

                st.subheader("Показники просторової трансформації")
                st.dataframe(spatial_metrics_df, width="stretch", hide_index=True)
                _build_download_csv(spatial_metrics_df, "spatial_transformation_metrics.csv")

                output_filename = generate_output_filename(uploaded_file.name, "просторове_положення_гравців")

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Завантажити результат",
                        data=f,
                        file_name=output_filename,
                        mime="video/mp4",
                        key="players_positioning_download",
                    )

            except Exception as e:
                st.error(f"Помилка: {str(e)}")


def navigate_to(page: str):
    st.query_params["page"] = page
    st.rerun()


def main():
    st.set_page_config(page_title="Аналіз футбольного відео", page_icon="images/football-ball.png", layout="wide")

    current_page = st.query_params.get("page", "home")

    if st.sidebar.button("Головна сторінка", width="stretch"):
        navigate_to("home")

    if st.sidebar.button("Просторове положення гравців", width="stretch"):
        navigate_to("players_positioning")

    if st.sidebar.button("Тестування моделей", width="stretch"):
        navigate_to("model_testing")

    device = "cuda" if _torch_cuda_available() else "cpu"

    if current_page == "home":
        show_home_page()
    elif current_page == "players_positioning":
        show_players_positioning_page(device)
    elif current_page == "model_testing":
        show_testing_page(device)
    else:
        show_home_page()


if __name__ == "__main__":
    main()