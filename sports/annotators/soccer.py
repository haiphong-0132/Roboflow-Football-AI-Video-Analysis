from typing import Optional, List
import cv2
import supervision as sv
import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration

def draw_pitch(
        config: SoccerPitchConfiguration,
        background_color: sv.Color = sv.Color(34, 139, 34),
        line_color: sv.Color = sv.Color(255, 255, 255),
        padding: int = 50,
        line_thickness: int = 4,
        point_radius: int = 8,
        scale: float = 0.1
) -> np.ndarray:
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3), dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        
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
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
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
        config: SoccerPitchConfiguration,
        xy: np.ndarray,
        face_color: sv.Color = sv.Color(255, 0, 0),
        edge_color: sv.Color = sv.Color(0, 0, 0),
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    
    for point in xy:
        scale_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        
        cv2.circle(
            img=pitch,
            center=scale_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )

        cv2.circle(
            img=pitch,
            center=scale_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch

def draw_paths_on_pitch(
        config: SoccerPitchConfiguration,
        paths: List[np.ndarray],
        color: sv.Color = sv.Color(255, 255, 255),
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            ) for point in path if point.size > 0
        ]

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

def draw_pitch_voronoi_diagram(
        config: SoccerPitchConfiguration,
        team_1_xy: np.ndarray,
        team_2_xy: np.ndarray,
        team_1_color: sv.Color = sv.Color(255, 0, 0),
        team_2_color: sv.Color = sv.Color(255, 255, 255),
        opacity: float = 0.5,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices(
        (scaled_width + 2 * padding, scaled_length + 2 * padding)
    )
    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt(
            (xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
            (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2
        )
    
    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay