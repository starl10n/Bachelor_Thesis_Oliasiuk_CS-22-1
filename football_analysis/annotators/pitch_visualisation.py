from typing import Optional, List
import cv2
import supervision as sv
import numpy as np
from football_analysis.configs.pitch_config import FootballPitchConfiguration

def draw_pitch(
    config: FootballPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    scaled_width = int(round(config.width * scale))
    scaled_length = int(round(config.length * scale))
    scaled_circle_radius = int(round(config.centre_circle_radius * scale))
    scaled_penalty_spot_distance = int(round(config.penalty_spot_distance * scale))

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (
            int(round(config.vertices[start - 1][0] * scale)) + padding,
            int(round(config.vertices[start - 1][1] * scale)) + padding
        )
        point2 = (
            int(round(config.vertices[end - 1][0] * scale)) + padding,
            int(round(config.vertices[end - 1][1] * scale)) + padding
        )
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image

def draw_points_on_pitch(
    config: FootballPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 14,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        if not np.isfinite(point).all():
            continue
        px = int(round(point[0] * scale)) + padding
        py = int(round(point[1] * scale)) + padding
        if px < 0 or py < 0 or px >= pitch.shape[1] or py >= pitch.shape[0]:
            continue
        scaled_point = (px, py)
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch

def draw_paths_on_pitch(
    config: FootballPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = []
        for point in path:
            if point.size == 0 or not np.isfinite(point).all():
                continue
            scaled_path.append(
                (
                    int(round(point[0] * scale)) + padding,
                    int(round(point[1] * scale)) + padding
                )
            )

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

    return pitch

