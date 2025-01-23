from moviepy.editor import VideoFileClip
import sys
import os 

sys.stdout.reconfigure(encoding='utf-8')


def convert_avi_to_mp4(input_file, output_file):
    try:
        with VideoFileClip(input_file) as clip:
            clip.write_videofile(output_file, codec="libx264")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi chuyển đổi: {e}")


