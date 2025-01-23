import numpy as np
import supervision as sv
from inference import get_model
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

video = './121364_0.mp4'
model = get_model(model_id="football-field-detection-f07vi/14", api_key='3Def0JUFoipsYZetEUac')

CONFIG = SoccerPitchConfiguration()



edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#00BFFF'),
    thickness=2,
    edges= CONFIG.edges
)

vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    radius=8
)

vertex_annotator_2 = sv.VertexAnnotator(
    color=sv.Color.from_hex('#00BFFF')
)

frame_generator = sv.get_video_frames_generator(video, start=200)
frame = next(frame_generator)

result = model.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

filter = key_points.confidence[0] > 0.5

frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])

pitch_reference_points = np.array(CONFIG.vertices)[filter]

transformer = ViewTransformer(
    source=pitch_reference_points,
    target=frame_reference_points
)

pitch_all_points = np.array(CONFIG.vertices)
frame_all_points = transformer.transform_points(points=pitch_all_points)

frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=frame_all_key_points)
annotated_frame = vertex_annotator_2.annotate(scene=annotated_frame, key_points=frame_all_key_points)
annotated_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=frame_reference_key_points)

sv.plot_image(annotated_frame)