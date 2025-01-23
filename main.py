import os
import cv2
import time
import numpy as np
from collections import defaultdict, deque
import supervision as sv
from tqdm import tqdm
from inference import get_model
from deep_sort_realtime.deepsort_tracker import DeepSort
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram)
from functions.detection_annotators import (annotate_human, annotate_ball, annotate_label)

def main(input_video: str, output_folder: str):
    # CONSTANT CONFIGURATION
    API_KEY = '3Def0JUFoipsYZetEUac'

    CONFIG = SoccerPitchConfiguration()

    PLAYER_DETECTION_MODEL = get_model(model_id='football1-udxft/1', api_key=API_KEY)
    BALL_DETECTION_MODEL = get_model(model_id='football-ball-detection-rejhg/2', api_key=API_KEY)
    FIELD_DETECTION_MODEL = get_model(model_id='football-field-detection-f07vi/14', api_key=API_KEY)

    GOALKEEPER_ID = 0
    REFEREE_ID = 1
    PLAYER_ID = 2
    BALL_ID = 3

    STRIDE = 1
    FRAME_RATE = 25
    FPS = 25

    # USER CONFIGURATION

    VIDEO_PATH = input_video

    OUTPUT_FOLDER = output_folder

    OUTPUT_TRACKING_VIDEO = os.path.join(OUTPUT_FOLDER, 'tracking_video.avi')
    OUTPUT_PLAYER_PROJECTION_VIDEO = os.path.join(OUTPUT_FOLDER, 'player_projection_video.avi')
    OUTPUT_VORONOI_DIAGRAM_VIDEO = os.path.join(OUTPUT_FOLDER, 'voronoi_diagram_video.avi')

    # MAIN CODE

    # Create team assignment model

    frame_generator = sv.get_video_frames_generator(
        source_path=VIDEO_PATH, stride=30
    )

    crops = []
    for frame in tqdm(frame_generator, desc='Extracting crops:'):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy)
                        for xyxy in players_detections.xyxy]
        
        crops.extend(players_crops)

    team_classifier = TeamClassifier()
    team_classifier.fit(crops)

    # Create video writers

    frame_generator = sv.get_video_frames_generator(source_path=VIDEO_PATH, stride=STRIDE)
    frame = next(frame_generator)
    height, width, _ = frame.shape

    pitch_frame = draw_pitch(config=CONFIG)
    pitch_height, pitch_width, _ = pitch_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = FPS

    video_writer1 = cv2.VideoWriter(
        filename=OUTPUT_TRACKING_VIDEO,
        fourcc=fourcc,
        fps=fps,
        frameSize=(width, height)
    )

    video_writer2 = cv2.VideoWriter(
        filename=OUTPUT_PLAYER_PROJECTION_VIDEO,
        fourcc=fourcc,
        fps=fps,
        frameSize=(pitch_width, pitch_height)
    )

    video_writer3 = cv2.VideoWriter(
        filename=OUTPUT_VORONOI_DIAGRAM_VIDEO,
        fourcc=fourcc,
        fps=fps,
        frameSize=(pitch_width, pitch_height)
    )

    # Get ball detections

    def get_ball_detections(
        frame, players_detections: sv.Detections,
        previous_detections: sv.Detections=sv.Detections(xyxy=np.empty((0, 4)))
        ) -> sv.Detections:

        result = BALL_DETECTION_MODEL.infer(frame, confidence=0.5)[0]
        ball_detections_1 = sv.Detections.from_inference(result)

        ball_detections_2 = players_detections[players_detections.class_id == BALL_ID]

        ball_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
        previous_center = previous_detections.get_anchors_coordinates(sv.Position.CENTER)

        min_dist = float('inf')

        def find_closest(detections):
            centers = detections.get_anchors_coordinates(sv.Position.CENTER)
            if len(previous_center) == 0:
                return detections[0], 0
            dists = np.linalg.norm(centers - previous_center)
            min_idx = np.argmin(dists)
            return detections[[min_idx]], np.min(dists)

        if len(ball_detections_1) > 0:
            closest_1, dist_1 = find_closest(ball_detections_1)
        else:
            dist_1 = float('inf')

        if len(ball_detections_2) > 0:
            closest_2, dist_2 = find_closest(ball_detections_2)
        else:
            dist_2 = float('inf')

        if dist_1 < dist_2:
            ball_detections = closest_1
        elif dist_2 < dist_1:
            ball_detections = closest_2


        if len(ball_detections) != 0:
            ball_detections.class_id[:] = BALL_ID
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=5)
        return ball_detections

    # Resolve team id of goalkeepers
    def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0) if len(players_xy[players.class_id == 0]) > 0 else np.array([0, 0])
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0) if len(players_xy[players.class_id == 1]) > 0 else np.array([height, width])

        distances_to_team_0 = np.linalg.norm(goalkeepers_xy - team_0_centroid, axis=1)
        distances_to_team_1 = np.linalg.norm(goalkeepers_xy - team_1_centroid, axis=1)

        return np.argmin(np.column_stack((distances_to_team_0, distances_to_team_1)), axis=1)

    def from_supervision_detections(detections: sv.Detections) -> list:
        detect = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]
            detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])

        return detect

    def from_deepsort_tracks(tracks) -> sv.Detections:
        xyxy = []
        confidence = []
        class_id = []
        tracker_id = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            xyxy.append([x1, y1, x2, y2])
            confidence.append(track.get_det_conf())
            class_id.append(track.get_det_class())
            tracker_id.append(track.track_id)
        if not xyxy:
            return sv.Detections(xyxy=np.empty((0,4)))
        
        return sv.Detections(
            xyxy=np.array(np.array(xyxy)),
            confidence=np.array(confidence),
            class_id=np.array(class_id),
            tracker_id=np.array(tracker_id)
        )

    tracker = DeepSort(
        max_age=15,
        embedder_gpu=False
    )

    start = time.time()
    previous_player_positions = defaultdict(lambda: deque(maxlen=5))
    player_speeds = {}
    speed_history = defaultdict(lambda: deque(maxlen=10))
    previous_frame_time = None
    previous_detections = sv.Detections(xyxy=np.empty((0, 4)))


    for frame in frame_generator:
        try:
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)
            
            ball_detections = get_ball_detections(frame, detections, previous_detections)
            previous_detections = ball_detections

            human_detections = detections[detections.class_id != BALL_ID]
            human_detections = human_detections.with_nms(threshold=0.5, class_agnostic=True)

            deepsort_detections = from_supervision_detections(human_detections)
            tracks = tracker.update_tracks(deepsort_detections, frame=frame)
            human_detections = from_deepsort_tracks(tracks)

            if len(human_detections) == 0:
                continue

            goalkeepers_detections = human_detections[human_detections.class_id == GOALKEEPER_ID]
            players_detections = human_detections[human_detections.class_id == PLAYER_ID]
            referees_detections = human_detections[human_detections.class_id == REFEREE_ID]

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)

            if len(goalkeepers_detections) > 0:
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

            if len(referees_detections) > 0:
                referees_detections.class_id += 1
            
            human_detections = sv.Detections.merge([
                players_detections,
                goalkeepers_detections,
                referees_detections
            ])

            players_detections = sv.Detections.merge([
                players_detections,
                goalkeepers_detections
            ])

            field_detection_result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            key_points = sv.KeyPoints.from_inference(field_detection_result)

            filtered_key_points = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filtered_key_points]

            pitch_reference_points = np.array(CONFIG.vertices)[filtered_key_points]

            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = transformer.transform_points(points=players_xy)

            current_frame_time = time.time()

            if previous_frame_time is not None:
                for track_id in human_detections.tracker_id:
                    if track_id in players_detections.tracker_id:
                        try:
                            player_index = players_detections.tracker_id.tolist().index(track_id)
                        except ValueError:
                            continue

                        current_position_pitch = pitch_players_xy[player_index]
                        previous_player_positions[track_id].append(current_position_pitch)

                        if len(previous_player_positions[track_id]) == 5:
                            first_position = previous_player_positions[track_id][0]
                            last_position = previous_player_positions[track_id][-1]

                            distance_pitch = np.linalg.norm(last_position - first_position)
                            time_diff = current_frame_time - previous_frame_time - 5
                            pixels_per_meter = 12
                            distance_meters = distance_pitch / pixels_per_meter * 3
                            speed_meters_per_second = distance_meters / time_diff
                            speed_kmh = speed_meters_per_second * 3.6

                            speed_history[track_id].append(speed_kmh)
                            player_speeds[track_id] = np.mean(speed_history[track_id])

            previous_frame_time = current_frame_time      

            labels = []
            for tracker_id in human_detections.tracker_id:
                if tracker_id in players_detections.tracker_id:
                    label = f'#{tracker_id}\n{player_speeds.get(tracker_id, 0):.2f} km/h'
                else:
                    label = f'#{tracker_id}'

                labels.append(label)
        
            human_detections.class_id = human_detections.class_id.astype(int)

            annotated_frame = frame.copy()
            annotated_frame = annotate_human(annotated_frame, human_detections)
            annotated_frame = annotate_ball(annotated_frame, ball_detections)
            annotated_frame = annotate_label(annotated_frame, human_detections, labels)
            
            video_writer1.write(annotated_frame)
            
            # Draw Player Projection

            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_referees_xy = transformer.transform_points(points=referees_xy)

            annotated_frame = draw_pitch(config=CONFIG)
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy = pitch_ball_xy,
                face_color=sv.Color.from_hex('#FFFFFF'),
                edge_color=sv.Color.from_hex('#000000'),
                radius=10,
                pitch=annotated_frame
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 0],
                face_color=sv.Color.from_hex('#00BFFF'),
                edge_color=sv.Color.from_hex('#000000'),
                radius=16,
                pitch=annotated_frame    
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 1],
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

            video_writer2.write(annotated_frame)

            # Draw Voronoi Diagram

            annotated_frame = draw_pitch(
                config=CONFIG,
                background_color=sv.Color.from_hex('#FFFFFF'),
                line_color=sv.Color.from_hex('#000000')
            )
            annotated_frame = draw_pitch_voronoi_diagram(
                config=CONFIG,
                team_1_xy=pitch_players_xy[players_detections.class_id == 0],
                team_2_xy=pitch_players_xy[players_detections.class_id == 1],
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
                xy=pitch_players_xy[players_detections.class_id == 0],
                face_color=sv.Color.from_hex('#00BFFF'),
                edge_color=sv.Color.from_hex('#000000'),
                radius=16,
                pitch=annotated_frame
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 1],
                face_color=sv.Color.from_hex('#FF1493'),
                edge_color=sv.Color.from_hex('#000000'),
                radius=16,
                pitch=annotated_frame
            )
            
            video_writer3.write(annotated_frame)

        except:
            continue

    video_writer1.release()
    video_writer2.release()
    video_writer3.release()