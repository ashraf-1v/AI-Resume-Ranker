# AI Resume Ranker

A comprehensive AI-powered resume analysis and candidate ranking system that automatically processes resumes, extracts key information, and ranks candidates based on job requirements using advanced machine learning techniques.

## ✨ Features

### Core Functionality
- **Multi-format Resume Processing**: Supports PDF and DOCX files
- **Intelligent Data Extraction**: Automatically extracts names, skills, experience, education, and contact information
- **Advanced NLP Processing**: Uses spaCy, NLTK, and TF-IDF for text analysis
- **Smart Candidate Ranking**: Ranks candidates based on similarity scores using cosine similarity
- **Real-time Processing**: Handles large batches of resumes efficiently
- **Automatic Hiring Status Tracking**: All candidates auto-recorded as "not hired" by default

### Advanced Analytics
- **Hiring Pattern Analysis**: Analyzes past hiring decisions to identify success factors
- **Machine Learning Models**: Builds predictive models using Logistic Regression, Random Forest, and Gradient Boosting
- **Skill Weight Calculation**: Dynamically calculates skill importance based on hiring success rates
- **Clustering Analysis**: Groups candidates using K-Means and DBSCAN algorithms
- **Performance Visualization**: Generates clean, simplified analytics dashboards with 3 key plots
- **Model Persistence**: Automatically saves and loads trained models across sessions

### Security & Privacy
- **Session Management**: Secure authentication with JWT tokens
- **Access Control**: Role-based authentication system
- **Data Privacy**: Minimal data extraction and secure storage

### User Interface
- **Modern Web Interface**: Built with Streamlit for intuitive user experience
- **Responsive Design**: Clean, professional UI with custom CSS styling
- **Real-time Updates**: Live progress tracking and status updates
- **Export Capabilities**: Download results in Excel and PDF formats
- **Clear Hiring Status**: Visual checkboxes showing hired/not hired status

## 🏗️ Architecture

The system follows a microservices architecture with three main components:

### 1. Flask Backend (`flask_backend.py`)
- RESTful API endpoints for all operations
- AI resume processing and ranking engine
- Advanced analytics and machine learning models
- Security and authentication management
- File processing and data extraction
- **Automatic model persistence** with pickle

### 2. Streamlit Frontend (`streamlit_frontend.py`)
- Interactive web interface
- File upload and job description input
- Real-time candidate ranking display
- Filtering and search capabilities
- Export functionality
- **Clear hiring status indicators**

### 3. Advanced Analytics (`advanced_analytics.py`)
- Machine learning model training and evaluation
- Hiring pattern analysis
- Skill weight calculation
- Clustering algorithms
- Performance reporting and visualization
- **Automatic model saving/loading**

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended for large datasets
- Internet connection for initial setup

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI-Resume-Ranker
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK Data (Automatic)
The system automatically downloads required NLTK data on first run.

## ⚡ Quick Start

### 1. Start the Backend Server
```bash
python flask_backend.py
```
The Flask server will start on `http://localhost:5000`

### 2. Launch the Frontend
```bash
streamlit run streamlit_frontend.py
```
The Streamlit app will open in your browser at `http://localhost:8501`

### 3. Login
- Username: `admin`
- Password: `admin123`

## 🐳 Docker Deployment

You can also run the application using Docker.

### 1. Build and Run
```bash
docker-compose up --build
```

### 2. Access the App
- Frontend: `http://localhost:8501`
- Backend: `http://localhost:5000`

## 📖 Usage Guide

### Step 1: Upload Resumes
1. Click "Choose resume files" in the sidebar
2. Select multiple PDF or DOCX files
3. The system will automatically process and extract information

### Step 2: Add Job Description
Choose one of two methods:
- **Paste Text**: Directly paste the job description
- **Upload File**: Upload a PDF or TXT file containing the job description

### Step 3: Process and Rank
1. Click "Process Files" to start analysis
2. The system will:
   - Extract text from all resumes
   - Calculate similarity scores using TF-IDF
   - Rank candidates by match quality
   - Display top 10 candidates
   - **Auto-record all candidates as "not hired"**

### Step 4: Mark Hiring Decisions
- **Checkbox unchecked**: ❌ Not Hired (default)
- **Checkbox checked**: ✅ Hired
- All candidates are automatically tracked for analytics

### Step 5: Review and Filter
- Use filters to narrow down candidates by:
  - Minimum match score
  - Minimum experience years
  - Keyword search
- View detailed candidate information

### Step 6: Generate Analytics
1. Click "Generate Analytics Report"
2. View 3 key insights:
   - **Top Skills by Hiring Success**: Which skills lead to hires
   - **Average Experience**: Hired vs not hired experience comparison
   - **CV Similarity Score**: How well candidates match job description
3. Analytics saved as `analytics_visualizations.png`

### Step 7: Export Results
- Download top 10 candidates as Excel or PDF
- Export includes all candidate details and rankings

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/login` - User authentication
- `POST /api/auth/logout` - User logout

### Resume Processing
- `POST /api/upload-resumes` - Upload and process resume files
- `POST /api/process-job-description` - Process job description and rank candidates
- `POST /api/process-job-description-file` - Process job description from file

### Candidate Management
- `GET /api/candidate/<id>` - Get detailed candidate information
- `POST /api/filter-candidates` - Filter candidates by criteria
- `POST /api/update-hiring-status` - Update hiring status

### Analytics
- `POST /api/analyze-patterns` - Analyze hiring patterns
- `POST /api/build-prediction-model` - Build ML prediction model  
- `POST /api/predict-hiring` - Predict hiring success for candidate
- `GET /api/performance-report` - Get comprehensive performance report
- `POST /api/generate-visualizations` - Generate analytics plots

### Export
- `GET /api/download-top-candidates/<format>` - Download top candidates (excel/pdf)

## 🤖 Machine Learning Features

### Text Processing
- **TF-IDF Vectorization**: Converts text to numerical features
- **N-gram Analysis**: Uses unigrams and bigrams for better context
- **Stop Word Removal**: Filters common words for better analysis
- **Text Preprocessing**: Cleans and normalizes text data

### Similarity Calculation
- **Cosine Similarity**: Measures similarity between job description and resumes
- **Feature Engineering**: Combines skills, experience, and text features
- **Dynamic Weighting**: Adjusts weights based on hiring success patterns

### Predictive Models
- **Logistic Regression**: Fast, interpretable baseline model
- **Random Forest**: Handles non-linear relationships and feature interactions
- **Gradient Boosting**: High-performance ensemble method
- **Model Selection**: Automatically selects best model based on F1-score
- **Model Persistence**: Saves/loads models automatically using pickle

### Clustering
- **K-Means**: Groups similar candidates for batch processing
- **DBSCAN**: Identifies candidate clusters with noise detection
- **Automatic Selection**: Chooses optimal algorithm based on data characteristics

## 📊 Analytics Dashboard

The system provides **3 simplified analytics plots**:

### 1. Top Skills by Hiring Success
- Shows which technical skills appear most in hired candidates
- Success rate from 0.0 (never hired) to 1.0 (always hired)
- Helps identify valuable skills for job descriptions

### 2. Average Experience
- Compares years of experience between hired and not hired
- Green bar = hired candidates average
- Red bar = not hired candidates average
- Shows if you prefer junior, mid, or senior levels

### 3. CV Similarity to Job Description
- Measures how well candidate CVs match the job posting
- Higher score = better alignment with job requirements
- Shows if job description matching matters in decisions

All visualizations are auto-saved to `analytics_visualizations.png`

## 💾 Model Persistence

All trained models are automatically saved and loaded:

### Saved Models
- `models/tfidf_vectorizer.pkl` - TF-IDF model
- `models/tfidf_matrix.pkl` - Document vectors
- `models/best_model.pkl` - Best ML classifier + scaler
- `models/skill_weights.pkl` - Skill importance weights
- `models/performance_metrics.pkl` - Model performance data
- `models/pattern_analysis.pkl` - Hiring patterns

### Benefits
- ✅ Skip retraining on every restart
- ✅ Faster application startup
- ✅ Reuse expensive computations
- ✅ Models improve incrementally over time

## ⚙️ Configuration

### Performance Settings
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 1GB)
- `UPLOAD_TIMEOUT`: Upload timeout (default: 10 minutes)
- `TFIDF_MAX_FEATURES`: Maximum TF-IDF features (default: 2000)

## 📁 Project Structure

```
AI-Resume-Ranker/
├── flask_backend.py          # Main Flask API server
├── streamlit_frontend.py     # Streamlit web interface
├── advanced_analytics.py     # ML analytics and models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Saved ML models (auto-created)
├── Datasets/
│   └── Resumes/             # Sample resume files
└── __pycache__/             # Python cache files
```

## 🔧 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

2. **NLTK Data Missing**
The system automatically downloads required NLTK data on first run.

3. **Memory Issues with Large Datasets**
- Process files in smaller batches
- Increase system RAM
- Close other applications

4. **File Upload Errors**
- Check file format (PDF/DOCX only)
- Verify file size (max 1GB)
- Ensure files are not corrupted

5. **Analytics Plot Empty/Not Showing**
- Mark at least 2 candidates as hired/not hired
- Restart Flask backend to load latest models
- Check `models/` directory exists

### Performance Tips
- Use SSD storage for faster file processing
- Allocate at least 4GB RAM for large datasets
- Use batch processing for 100+ resumes
- Models are cached after first training

## 🆕 Recent Updates

### Model Persistence (v2.0)
- ✅ Automatic save/load of all trained models
- ✅ Models persist across application restarts
- ✅ Faster startup with pre-trained models

### Analytics Improvements (v2.1)
- ✅ Simplified to 3 key plots (removed F1 score plot)
- ✅ Cleaner visualizations with value labels
- ✅ Horizontal layout for better readability
- ✅ Comprehensive plot explanations

### UX Enhancements (v2.2)
- ✅ Auto-record all candidates as "not hired" by default
- ✅ Clear checkbox labels (✅ Hired / ❌ Not Hired)
- ✅ Immediate visual feedback on status changes

### Code Optimization (v2.3)
- ✅ Removed 6 unused imports
- ✅ Cleaner, more maintainable codebase
- ✅ Improved error handling

## 🔮 Future Enhancements

- [ ] Multi-language support (Arabic, Spanish, etc.)
- [ ] Advanced NLP models (BERT, GPT)
- [ ] Mobile app interface
- [ ] Integration with job boards
- [ ] Advanced reporting dashboards
- [ ] Automated candidate outreach
- [ ] Email notifications for top candidates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for smarter hiring decisions**
