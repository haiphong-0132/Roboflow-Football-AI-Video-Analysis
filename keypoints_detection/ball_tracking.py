from collections import deque
import supervision as sv
from tqdm import tqdm
import numpy as np
from inference import get_model
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from sports.common.view import ViewTransformer

video = './121364_0.mp4'
BALL_ID = 0
MAXLEN = 5
CONFIG = SoccerPitchConfiguration()
API_KEY = '3Def0JUFoipsYZetEUac'
player_detection_model = get_model(model_id='football-players-detection-3zvbc/12', api_key=API_KEY)
field_detection_model = get_model(model_id='football-field-detection-f07vi/14', api_key=API_KEY)

video_info = sv.VideoInfo.from_video_path(video)
# print(video_info)
# VideoInfo(width=1920, height=1080, fps=25, total_frames=750)

frame_generator = sv.get_video_frames_generator(source_path=video, stride=30)

path_raw = []
M = deque(maxlen=MAXLEN)

for frame in tqdm(frame_generator, total=video_info.total_frames//30):
    result = player_detection_model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    result = field_detection_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    M.append(transformer.m)
    transformer.m = np.mean(np.array(M), axis=0)

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    path_raw.append(pitch_ball_xy)

path = [
    np.empty((0, 2), dtype=np.float32) if coordinates.shape[0] >= 2 else coordinates for coordinates in path_raw
]

path = [coordinates.flatten() for coordinates in path]

annotated_frame = draw_pitch(CONFIG)
annotated_frame = draw_paths_on_pitch(
    config=CONFIG,
    paths=[path],
    color=sv.Color.from_hex('#FFFFFF'),
    pitch=annotated_frame
)

sv.plot_image(annotated_frame)