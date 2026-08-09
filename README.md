# Disaster Classification App

##  Project Overview
The Disaster Classification App is an AI-powered full-stack web application designed to classify natural disasters from images. It provides a sleek, modern interface where users can upload images and receive instant predictions, complete with confidence scores and a Top-3 prediction chart. 

##  Dataset Details
- **Source**: AIDERv2 (Aerial Image Database for Emergency Response)
- **Size**: Approximately 8,000 images were used for training, validation, and testing (sampled from the 16,723 total).
- **Classes**:
  1. `Earthquake`
  2. `Fire`
  3. `Flood`
  4. `Normal`

##  Model Architecture & Performance
The selected model is **`model_b_deeper`**, a custom Convolutional Neural Network (CNN).
- **Architecture**: 3 Convolutional Blocks (Filters: 32, 64, 128) with max pooling and a dense classification head (128 units).
- **Parameters**: 110,276
- **Input Size**: 160x160 RGB images (`160, 160, 3`).
- **Evaluation Metrics**:
  - **Training Accuracy**: 87.66%
  - **Validation Accuracy**: 83.67%
  - **Test Accuracy**: 86.31%
  - **Test Macro F1-Score**: 0.855

##  Backend Specifications
- **Framework**: FastAPI running on Python 3.13.
- **Package Manager**: `uv` for fast dependency resolution.
- **CORS**: Configured with `CORSMiddleware` to accept cross-origin requests from the frontend.
- **Model Loading**: Implements **lazy loading** (`get_model()`). The `best_model.keras` file is only loaded into memory upon the first prediction request, ensuring fast startup times.
- **API Endpoint (`POST /predict`)**:
  - Accepts `multipart/form-data` with an `image` file.
  - Validates format (JPG/PNG) and size (Max 10MB).
  - Preprocesses the image (resize to 160x160, normalize by `/ 255.0`).
  - Returns a JSON response containing `predicted_class`, `confidence`, and `top_predictions` array.

##  Frontend Specifications
- **Framework**: React with Vite.
- **UI & Styling**: Custom responsive dark-mode CSS with glassmorphism effects and a purple accent (`#9d4edd`).
- **Libraries**: 
  - `axios` for handling image uploads via `FormData`.
  - `lucide-react` for SVG icons.
  - `chart.js` and `react-chartjs-2` for rendering the Top-3 confidence bar chart.

##  Docker & Containerization
The project is fully containerized using Docker and Docker Compose.
- **Backend (`Dockerfile.backend`)**: Built on `python:3.13-slim`, leveraging `uv` for dependency installation. Runs on port 8000 via Uvicorn.
- **Frontend (`Dockerfile.frontend`)**: Built on `node:22-alpine`, runs the Vite development server on port 5173.
- **Docker Compose**: Orchestrates both services. The `models/` directory (along with `data/config.py` and `data/pipeline.py`) is mapped as a **volume** into the backend container, allowing the trained `best_model.keras` to be accessed dynamically without rebuilding the image.

##  Deployment
- **Backend**: Hosted as a Web Service on **Render** (URL: `https://disaster-classification-app-1.onrender.com/`).
- **Frontend**: Hosted as a Static Site on Render. The React application makes direct API calls to the deployed FastAPI backend.

##  Local Setup

### Using Docker (Recommended)
```bash
# Start both frontend and backend
docker-compose up --build
```
- Frontend: `http://localhost:5173`
- Backend API Docs: `http://localhost:8000/docs`

### Manual Setup
**1. Backend**
```bash
# Install dependencies using uv
uv sync
# Start server
uv run uvicorn backend.main:app --reload --port 8000
```

**2. Frontend**
```bash
cd frontend
# Install dependencies
npm install
# Start Vite server
npm run dev
```
