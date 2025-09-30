#!/bin/bash
# ExoBengal API Startup Script

echo "🚀 Starting ExoBengal API Server..."


# Install/upgrade requirements
echo "📦 Installing/upgrading requirements..."
pip install -r requirements.txt

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "❌ Error: models directory not found!"
    echo "   Please ensure the models directory exists with the required model files."
    exit 1
fi

# Check if all required model files exist
required_files=("random_forest_classifier.pkl" "decision_tree_classifier.pkl" "cnn_model.h5" "knn_model.pkl" "scaler.pkl" "imputer.pkl")

for file in "${required_files[@]}"; do
    if [ ! -f "models/$file" ]; then
        echo "❌ Error: models/$file not found!"
        exit 1
    fi
done

echo "✅ All model files found!"

# Start the API server
echo "🌟 Starting FastAPI server on http://localhost:8000"
echo "📖 API Documentation available at: http://localhost:8000/docs"
echo "🔍 Interactive API explorer at: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
