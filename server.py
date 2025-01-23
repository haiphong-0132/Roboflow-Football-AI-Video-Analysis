import os
import avi_mp4
from flask import Flask, render_template, request, jsonify, send_from_directory
from process.video_processing import process_video 

app = Flask(__name__)

OUTPUT_FOLDER = 'static'
results_path = os.path.join(OUTPUT_FOLDER, 'results')  
uploads_path = os.path.join(OUTPUT_FOLDER, 'uploads') 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    # Delete old upload files

    for filename in os.listdir(uploads_path):
        if filename.endswith('mp4') or filename.endswith('avi'):
            file_path = os.path.join(uploads_path, filename)
            os.remove(file_path)

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and (file.filename.endswith('.mp4') or file.filename.endswith('.avi')):  # Kiểm tra file có phải là video không
        file_path = os.path.join(uploads_path, file.filename)  # Lưu file vào thư mục static/uploads
        file.save(file_path)
        return jsonify({'file_path': file_path}), 200
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/process-video', methods=['POST'])
def process_video_route():
    data = request.get_json()
    file_path = data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Invalid file path'}), 400

    try:
        output_videos = process_video(file_path, results_path)
        return jsonify({'message': 'Video processed successfully', 'output_videos': output_videos}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video-list', methods=['GET', 'POST'])
def video_list():
    # Get folders from results directory
    folders = [folder for folder in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, folder))]

    if not folders:
        return jsonify({'error': 'No folders found'}), 404

    if request.method == 'POST':
        # USER SELECTED FOLDER
        selected_folder = request.form.get('folder')
        if not selected_folder or selected_folder not in folders:
            return jsonify({'error': 'Invalid folder selected'}), 400
    else:
        selected_folder = folders[0]  # Default selected folder

    folder_path = os.path.join(results_path, selected_folder)
    videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]

    # CONVERT AVI TO MP4
    for video in videos:
        if video.endswith('.avi'):
            avi_path = os.path.join(folder_path, video)
            mp4_path = os.path.join(folder_path, video.replace('.avi', '.mp4'))
            avi_mp4.convert_avi_to_mp4(avi_path, mp4_path)
            os.remove(avi_path)  # Delete AVI

    # Update videos list
    videos = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    return render_template('video_list.html', videos=videos, folders=folders, folder_name=selected_folder)


@app.route('/video/<folder>/<filename>')
def video(folder, filename):
    return send_from_directory(os.path.join(results_path, folder), filename)

if __name__ == '__main__':
    app.run(debug=True)