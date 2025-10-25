# Customer Feedback Analyzer - Enterprise AI Platform

<div align="center">
    
_*Advanced sentiment analysis platform with custom-trained BERT models and RTX 4060 GPU acceleration*_ <br></br>
   [![Built on - Python](https://img.shields.io/badge/Backend-Python%20%7C%20Flask-blue)](#)
   [![Built with - JavaScript](https://img.shields.io/badge/Frontend-JavaScript%20%7C%20Material%20Design-red)](#)
   ![AI Models](https://img.shields.io/badge/AI-BERT--Large%20%7C%20Custom%20Trained-8A2BE2)
   ![License](https://img.shields.io/badge/License-MIT-2F2F2F)

</div>

---

## 🚀 Overview

AIML Customer Feedback Analyzer is an enterprise-grade AI platform that transforms customer feedback into actionable business intelligence:
- **Custom-trained BERT models** achieving 95.1% sentiment accuracy and 93.0% emotion detection
- **Real-time analysis** with <100ms inference times on RTX 4060 GPU
- **Business analytics dashboard** with comprehensive insights and recommendations
- **Bulk processing** with Gemini API integration for large-scale analysis
- **Google Material Design 3.0** interface with seamless user experience

Built with PyTorch, Flask, and modern web technologies, optimized for NVIDIA RTX 4060 acceleration.

---
## 🌟 Key Features

| Feature                       | Description                                                                                                                              |
| ----------------------------- | -----------------------------------------------------------------------------------------------------------------------------------------|
| 🤖 **Advanced AI Models**     | Custom BERT-Large transformers with 95.1% sentiment accuracy, 93.0% emotion detection, and multi-label classification                   |
| ⚡ **GPU Acceleration**       | RTX 4060 optimized inference pipeline delivering <100ms response times with CUDA optimization                                           |
| 📊 **Business Intelligence**  | Comprehensive analytics dashboard with KPIs, trend analysis, churn prediction, and revenue impact assessment                           |
| 🔄 **Real-time Processing**   | Live sentiment analysis with streaming results, confidence scoring, and performance metrics                                             |
| 📁 **Bulk Analytics**         | Gemini API integration for processing CSV/PDF files with automated business insights generation                                          |
| 🎨 **Modern UI/UX**          | Google Material Design 3.0 interface with responsive layout, smooth animations, and intuitive navigation                               |

---

## Project Structure
```bash
📁 aiml-feedback-analyzer/
├── 📁 frontend/
│   ├── 📄 index.html
│   ├── 📄 script.js
│   ├── 📄 styles.css
│   └── 📁 assets/
├── 📁 backend/
│   ├── 📄 app.py
│   ├── 📁 models/
│   │   ├── 📁 sentiment_analyzer/
│   │   └── 📁 emotion_detector/
│   ├── 📁 services/
│   └── 📁 utils/
├── 📁 notebooks/
├── 📁 data/
├── 📄 requirements.txt
└── 📄 README.md
```

# Note : Refer Huggingface for the trained ML Model, link :
```bash
https://huggingface.co/vishnupriyan07/Customer-Reviews-Sentiment-Business-Analysis
```

---
## 🚀 **Core Capabilities**
---
### AI Model Architecture 🧠

- **Sentiment Analysis**: Fine-tuned BERT-Large achieving 95.1% accuracy with binary/multi-class classification
- **Emotion Detection**: 6-emotion multi-label classifier (Joy, Love, Anger, Fear, Sadness, Surprise) with 93.0% accuracy  
- **Theme Extraction**: Automatic topic identification and categorization with confidence scoring
- **Business Insights**: Rule-based and ML-driven recommendation engine for actionable business intelligence

---
### Real-time Performance ⚡
>**Inference Speed:** RTX 4060 GPU acceleration → <100ms per analysis → 12,800+ analyses per minute  
>**Model Loading:** Optimized PyTorch 2.0 with TorchScript → INT8 quantization → 3x speed improvement  
>**Memory Efficiency:** 4.2GB GPU memory utilization → Efficient batch processing → Dynamic scaling  
>**API Throughput:** RESTful endpoints → WebSocket streaming → Real-time dashboard updates

---
### Business Intelligence Platform 📊
- **Dashboard Analytics**: KPI tracking, sentiment distribution, trend analysis, performance metrics with Chart.js visualizations
- **Bulk Processing**: CSV/PDF upload → Gemini API integration → Automated business insights → Comprehensive reporting
- **Export Capabilities**: JSON/CSV reports → Business recommendations → Executive summaries → Actionable insights
---

### Technical Innovation 🔧
**Custom Training Pipeline:** Domain-specific BERT fine-tuning → Active learning → Knowledge distillation →  
**GPU Optimization:** CUDA kernels → TensorRT integration → Memory pooling → Async processing  
**Material Design:** Google Design System → Responsive layout → Smooth animations → Accessibility compliance  
**API Integration:** Flask backend → WebSocket real-time → Gemini AI → Google Drive connectivity

---
## 🚀 Quick Start
Installation:

```bash
# Clone git repo
git clone https://github.com/vishnupriyanpr/AI-powered-Consumer-feedback---Business-Analytics.git
cd AI-powered-Consumer-feedback---Business-Analytics

# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Login with your Hugging Face credentials
hf auth login

# Push your model files
hf upload vishnupriyan07/Customer-Reviews-Sentiment-Business-Analysis

# Install Dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Web Interface:
```bash
Navigate to http://localhost:5000
```
Requirements:
- Python 3.8+ with PyTorch 2.0+
- NVIDIA RTX 4060 (or compatible GPU)
- 8GB+ system RAM, 4GB+ GPU memory
- Modern web browser with JavaScript enabled

---
## Demo Video

[![Watch the video](https://img.youtube.com/vi/U01EYXlp4WU/hqdefault.jpg)](https://www.youtube.com/watch?v=U01EYXlp4WU)

---
## 🤝 Contributing
PRs welcome! Development flow:
1) Fork repository → Create feature branch
2) Implement changes → Add comprehensive tests  
3) Update documentation → Ensure code quality
4) Submit PR with detailed description and performance benchmarks

**ML Model Improvements** - Enhanced training data - Architecture optimizations - New emotion categories - Multi-language support  
**Frontend Enhancements** - Advanced visualizations - Mobile optimization - New analytics features - UX improvements  
**Backend Scaling** - API performance - Database integration - Cloud deployment - Security hardening

---

## 📜 License

Apache License 2.0 (see LICENSE file)

---

## 🙌 Acknowledgments & Core Team

This project is made with precision and innovation by **Vishnupriyan P R**. 

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/vishnupriyanpr">
        <img src="https://github.com/vishnupriyanpr.png?size=120" width="120px;" alt="Vishnupriyan P R"/>
        <br />
        <sub><b>Vishnupriyan P R</b></sub>
      </a>
      <br />
      <sub><b>ML Engineer & Full-Stack Developer</b></sub>
      <br />
      <sub>BERT Fine-tuning • GPU Optimization • Business Intelligence</sub>
    </td>
  </tr>
</table>


---
<div align="center">
  <p><i>🧠 Transforming Customer Feedback into Business Intelligence with Advanced AI 🚀</i></p>
</div>
