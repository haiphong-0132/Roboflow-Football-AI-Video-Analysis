# ⚽ Football AI Video Analysis 

## Overview

Football match video analysis tool using AI and computer vision. Tracks players, analyzes team tactics, and generates performance insights. Built with Python, YOLOv8, and Flask.

## 📖 Acknowledgements
This source code is referenced and developed based on the [Roboflow Sports Repository](https://github.com/roboflow/sports). Sincere thanks to the Roboflow development team for providing the initial reference and foundation for this project.

## 🚀 Key Features

### Intelligent Object Detection
- **Player & Ball & Pitch Tracking**: Precise detection using Roboflow models
- **DeepSORT Tracking**: Consistent object identification across video frames
- **Team Classification**: Intelligent player team segregation using SigLIP - Sigmoid Loss for Language Image Pretraining

### Advanced Analytics
- **Player Speed Calculation**: Real-time movement velocity estimation
- **2D Pitch Projection**: Intuitive field representation
- **Voronoi Diagram Generation**: Visualize player control zones

### Technical Stack
- Deep Learning: Hugging Face, Ultralytics YOLOv8
- Web Framework: Flask
- Object Detection: Roboflow
- Classification: SigLIP

## 🛠 Quick Setup

### Prerequisites
- Python 3.11.0
- GPU recommended for optimal performance

### Installation Steps
```bash
# Clone Repository
git clone https://github.com/haiphong-0132/Roboflow-Football-AI-Video-Analysis.git
cd Roboflow-Football-AI-Video-Analysis

# Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install Dependencies
pip install -r requirements.txt
```

## 🎥 Usage

1. Start the Flask server:
   ```bash
   python -u server.py
   ```

2. Access web interface: `http://127.0.0.1:5000/`

3. Upload football match video (`.mp4` or `.avi`)

4. Process and analyze video

## 📂 Project Structure

| Directory/File | Purpose |
|---------------|---------|
| `main.py` | Core video processing logic |
| `server.py` | Web application backend |
| `functions/` | Annotation and processing utilities |
| `sports/` | Sports-specific modules |
| `static/` | Web assets and processed videos |

## 💡 Performance Tips
- Use high-quality input videos
- Prefer GPU-enabled machines
- Expect longer processing times for extended videos

## 🤝 Contributions

Contributions are welcome! 
- Open issues for bugs/suggestions
- Submit pull requests
- Follow existing code style

## 🔗 References
- [Roboflow Sports Repository](https://github.com/roboflow/sports)
