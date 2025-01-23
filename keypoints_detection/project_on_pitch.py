import supervision as sv
from tqdm import tqdm
from inference import get_model
import numpy as np
from sports.common.team import TeamClassifier
from sports.annotators.soccer import (draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

API_KEY = '3Def0JUFoipsYZetEUac'
video = './test(35).mp4'
player_detection_model = get_model(model_id='football-players-detection-3zvbc/12', api_key=API_KEY)

# player_detection_model = get_model(model_id='football1-udxft/1', api_key=API_KEY)

field_detection_model = get_model(model_id='football-field-detection-f07vi/14', api_key=API_KEY)

CONFIG = SoccerPitchConfiguration()


PLAYER_ID = 2
BALL_ID = 0
GOALKEEPER_ID = 1
REFEREE_ID = 3

# Team assignment model
STRIDE = 30

frame_generator = sv.get_video_frames_generator(source_path=video, stride=STRIDE)
crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    result = player_detection_model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier()
team_classifier.fit(crops)

# Projection
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20, height=17
)

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(video)
frame = next(frame_generator)

# Detect players, ball and referee

result = player_detection_model.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections = tracker.update_with_detections(detections=all_detections)

goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
players_detections = all_detections[all_detections.class_id == PLAYER_ID]
referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

# Team assignment

player_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
players_detections.class_id = team_classifier.predict(player_crops)

goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

referees_detections.class_id -= 1

all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

# frame visualization

labels = [
    f'#{tracker_id}' for tracker_id in all_detections.tracker_id
]

all_detections.class_id = all_detections.class_id.astype(int)

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections,
    labels=labels
)
annotated_frame = triangle_annotator.annotate(
    scene=annotated_frame,
    detections= ball_detections
)

sv.plot_image(annotated_frame)

players_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

# Detect pitch key points

result = field_detection_model.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

# Project ball, players and referees on pitch

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
pitch_reference_points = np.array(CONFIG.vertices)[filter]

transformer = ViewTransformer(
    source=frame_reference_points,
    target=pitch_reference_points
)

frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pitch_player_xy = transformer.transform_points(points=players_xy)

referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pitch_referees_xy = transformer.transform_points(points=referees_xy)

# Visualize radar view

annotated_frame = draw_pitch(config=CONFIG)
annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_ball_xy,
    face_color=sv.Color.from_hex('#FFFFFF'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=10,
    pitch=annotated_frame
)
annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_player_xy[players_detections.class_id == 0],
    face_color=sv.Color.from_hex('#00BFFF'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=16,
    pitch=annotated_frame
)
annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_player_xy[players_detections.class_id == 1],
    face_color=sv.Color.from_hex('#FF1493'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=16,
    pitch=annotated_frame
)
annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_referees_xy,
    face_color=sv.Color.from_hex('#FFD700'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=16,
    pitch=annotated_frame
)

sv.plot_image(annotated_frame)

# Visualize Voronoi diagram

annotated_frame = draw_pitch(CONFIG)
annotated_frame = draw_pitch_voronoi_diagram(
    config=CONFIG,
    team_1_xy=pitch_player_xy[players_detections.class_id == 0],
    team_2_xy=pitch_player_xy[players_detections.class_id == 1],
    team_1_color=sv.Color.from_hex('#00BFFF'),
    team_2_color=sv.Color.from_hex('#FF1493'),
    pitch=annotated_frame
)

sv.plot_image(annotated_frame)

# Visualize voronoi diagram with blend

annotated_frame  = draw_pitch(
    config=CONFIG,
    background_color=sv.Color.from_hex('#FFFFFF'),
    line_color=sv.Color.from_hex('#000000')
)

annotated_frame = draw_pitch_voronoi_diagram(
    config=CONFIG,
    team_1_xy=pitch_player_xy[players_detections.class_id == 0],
    team_2_xy=pitch_player_xy[players_detections.class_id == 1],
    team_1_color=sv.Color.from_hex('#00BFFF'),
    team_2_color=sv.Color.from_hex('#FF1493'),
    pitch=annotated_frame
)

annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_ball_xy,
    face_color=sv.Color.from_hex('#FFFFFF'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=10,
    pitch=annotated_frame
)

annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_referees_xy,
    face_color=sv.Color.from_hex('#FFD700'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=16,
    pitch=annotated_frame
)

annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_player_xy[players_detections.class_id == 0],
    face_color=sv.Color.from_hex('#00BFFF'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=16,
    pitch=annotated_frame
)

annotated_frame = draw_points_on_pitch(
    config=CONFIG,
    xy=pitch_player_xy[players_detections.class_id == 1],
    face_color=sv.Color.from_hex('#FF1493'),
    edge_color=sv.Color.from_hex('#000000'),
    radius=16,
    pitch=annotated_frame
)

sv.plot_image(annotated_frame)
