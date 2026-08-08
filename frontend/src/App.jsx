import React, { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, Image as ImageIcon, Loader2, RefreshCw } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      validateAndSetFile(selectedFile);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    setError(null);
    setPrediction(null);

    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError('Unsupported image format. Please upload JPG, JPEG, or PNG.');
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('File too large. Maximum size is 10 MB.');
      return;
    }

    setFile(selectedFile);
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      validateAndSetFile(droppedFile);
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please upload an image first.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      // Direct API call to FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (err) {
      if (err.response) {
        setError(err.response.data?.detail || 'Server error occurred.');
      } else if (err.request) {
        setError('Backend is unavailable. Please ensure the server is running.');
      } else {
        setError('An unexpected error occurred.');
      }
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const chartData = prediction ? {
    labels: prediction.top_predictions.map(p => p.class),
    datasets: [
      {
        label: 'Confidence Score (%)',
        data: prediction.top_predictions.map(p => p.confidence),
        backgroundColor: 'rgba(157, 78, 221, 0.7)',
        borderColor: '#9d4edd',
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: '#1a1625',
        titleColor: '#f8f9fa',
        bodyColor: '#c77dff',
        borderColor: 'rgba(157, 78, 221, 0.4)',
        borderWidth: 1,
        padding: 12,
        displayColors: false,
        callbacks: {
          label: (context) => `${context.parsed.y}%`
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: '#adb5bd',
        }
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          color: '#adb5bd',
          font: {
            size: 14
          }
        }
      }
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Disaster Classification AI</h1>
        <p>Upload an image to classify the type of disaster using our CNN model.</p>
      </header>

      {error && <div className="error-message">{error}</div>}

      {!prediction && (
        <div className="upload-card">
          {!previewUrl ? (
            <div
              className="dropzone"
              onClick={() => fileInputRef.current.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <UploadCloud size={48} className="icon-purple" />
              <p>Click or drag & drop an image here</p>
              <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', color: '#6c757d' }}>JPG, JPEG, PNG up to 10MB</p>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/jpeg, image/jpg, image/png"
                style={{ display: 'none' }}
              />
            </div>
          ) : (
            <div className="preview-container">
              <img src={previewUrl} alt="Preview" className="preview-image" />
              <div className="actions">
                <button className="remove-btn" onClick={resetForm} disabled={loading}>
                  Remove
                </button>
                <button className="btn" onClick={handlePredict} disabled={loading}>
                  {loading ? (
                    <><Loader2 className="loading-spinner" size={20} /> Analyzing...</>
                  ) : (
                    <><ImageIcon size={20} /> Classify Image</>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {prediction && (
        <div className="prediction-card">
          <div className="prediction-result">
            <h2>Predicted Disaster Class</h2>
            <h3>{prediction.predicted_class}</h3>
            <p style={{ color: '#adb5bd', marginTop: '0.5rem' }}>
              Confidence: {prediction.confidence.toFixed(1)}%
            </p>
          </div>

          <div className="chart-container">
            <Bar data={chartData} options={chartOptions} />
          </div>

          <div className="actions" style={{ marginTop: '2.5rem' }}>
            <button className="btn" onClick={resetForm}>
              <RefreshCw size={18} /> Classify Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
