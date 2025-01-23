import os
import cv2
from main import main

def create_folder(base_path, folder_prefix):
    folder_number = 1
    while True:
        newfolder_name = f"{folder_prefix}_{folder_number}"
        newfolder_path = os.path.join(base_path, newfolder_name)
        if not os.path.exists(newfolder_path):
            os.makedirs(newfolder_path)
            return newfolder_path
        folder_number += 1

def process_video(input_path, output_folder):
    output_path = create_folder(output_folder, 'video')

    main(input_path, output_path)
    
    OUTPUT_TRACKING_VIDEO = os.path.join(output_path, 'tracking_video.avi')
    OUTPUT_PLAYER_PROJECTION_VIDEO = os.path.join(output_path, 'player_projection_video.avi')
    OUTPUT_VORONOI_DIAGRAM_VIDEO = os.path.join(output_path, 'voronoi_diagram_video.avi')

    return [OUTPUT_TRACKING_VIDEO, OUTPUT_PLAYER_PROJECTION_VIDEO, OUTPUT_VORONOI_DIAGRAM_VIDEO]