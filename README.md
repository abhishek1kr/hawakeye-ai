# Hawkeye AI - Road Intelligence & Infrastructure Suite

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-v18-61DAFB.svg)](https://reactjs.org/)
[![YOLO11](https://img.shields.io/badge/AI-YOLO11-red.svg)](https://ultralytics.com/)

## 📂 Project Structure

```text
Abhishek_Project/
├── backend/            # AI Pipeline, FastAPI, Models, Configs
│   ├── src/            # Python Source Code
│   ├── models/         # Trained AI Weights
│   ├── config/         # System Configurations
│   └── data/           # Uploaded Videos & Results
├── frontend/           # React Dashboard (Vite)
└── README.md           # This guide
```

## 🚀 Key Features

- **YOLO11 Power**: High-precision detection and segmentation of potholes and cracks.
- **Geospatial Mapping**: Interactive Leaflet maps with color-coded road health.
- **Maintenance Budgeting**: Automated repair cost estimation in INR.
- **Enterprise Security**: JWT-based user authentication.

## 🚦 Setup & Execution

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
# Start the API
$env:PYTHONPATH = "."; python src/api/server.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
# Start the Dashboard
npm run dev
```

3. Open **http://localhost:5173** and Create an Account!
