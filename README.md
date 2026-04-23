# 🌱 AgriScan AI - Crop Disease Detection

A modern, professional Streamlit application for detecting crop diseases using deep learning and the PlantVillage dataset.

## ✨ Features

### 🔍 AI Diagnoser
- Upload crop leaf images for instant disease detection
- Real-time AI analysis with confidence scores
- Detailed disease information including:
  - Pathogen cause (scientific names)
  - Transmission methods
  - Treatment recommendations
- **Session History Table**: Track all diagnoses in the current session
- **Interactive Table Management**: Delete specific rows or clear entire history

### 📊 Analytics Dashboard
- **5 Key Metric Cards**:
  1. Total PlantVillage Dataset (54,306 images)
  2. Dataset Categories (38 disease classes)
  3. Current Session Uploads (dynamic counter)
  4. Model Accuracy (95.28%)
  5. Active Pathogen Alerts (diseased plants detected)
- **Interactive Charts**:
  - Crop distribution bar chart
  - Health status pie chart
  - Session-specific plant distribution
- **Session Statistics**: Real-time analysis of current session data

## 🎨 Design Features

- Clean, modern SaaS interface
- Soft green agricultural color palette
- Professional typography (Inter font family)
- Smooth animations and hover effects
- Responsive layout with proper spacing
- Accessible color contrasts

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application
├── disease_database.py     # Disease information database (38 classes)
├── model.h5               # Trained TensorFlow model
├── requirements.txt       # Python dependencies
└── test_images/          # Sample images for testing
```

## 🧠 Model Information

- **Architecture**: TensorFlow/Keras CNN
- **Dataset**: PlantVillage (54,306 images)
- **Classes**: 38 disease categories across 14 crop types
- **Accuracy**: 95.28%

## 📋 Supported Crops & Diseases

The model can detect diseases in:
- Apple, Blueberry, Cherry, Corn, Grape, Orange
- Peach, Pepper (Bell), Potato, Raspberry, Soybean
- Squash, Strawberry, Tomato

## 🔄 Session State Features

The app maintains session state for:
- **Upload Counter**: Tracks total uploads in current session
- **Diagnosis History**: Stores all predictions with plant names, diseases, and recommendations
- **Persistent Data**: History persists across page navigation within the same session

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **TensorFlow**: Deep learning model
- **Plotly**: Interactive data visualizations
- **Pandas**: Data manipulation and display
- **PIL**: Image processing

## 📝 License

This project uses the PlantVillage dataset for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for farmers and agricultural researchers**
