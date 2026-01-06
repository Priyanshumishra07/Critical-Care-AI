# Multi-Agent Medical Imaging Diagnostic System
## Project Report & Documentation

---

## 📋 **Project Overview**

A multi-agent orchestration framework for multi-modal medical imaging diagnosis, integrating specialized agents for pneumonia detection (chest X-ray) and skin lesion segmentation (dermatology).

---

## 🎯 **System Architecture**

### **Agents:**
1. **Pneumonia Agent** - DenseNet121-based classification (97.84% accuracy)
2. **Skin Lesion Agent** - U-Net-based segmentation (pre-trained)
3. **Routing System** - LLM-powered with rule-based fallback
4. **RAG Agent** - Medical knowledge retrieval (optional)
5. **Web Search Agent** - Latest medical research (optional)

### **Key Features:**
- Multi-agent orchestration via LangGraph
- Intelligent routing with fallback mechanisms
- High-performance specialized models
- Scalable architecture for new modalities

---

## 📊 **Results**

### **Pneumonia Detection:**
- **Accuracy:** 97.84%
- **Precision:** 97.86%
- **Recall:** 97.84%
- **F1-Score:** 97.84%
- **Test Samples:** 878 images
- **Inference Time:** 0.06 seconds/image

### **Skin Lesion Segmentation:**
- **Status:** Functional (pre-trained U-Net)
- **Inference Time:** 0.49 seconds/image
- **Model:** ISIC2018 pre-trained

### **Routing System:**
- **Accuracy:** 76.19% (with fallback)
- **Fallback:** Rule-based routing (works without LLM)

---

## 🚀 **Quick Start**

### **1. Installation:**
```bash
pip install -r requirements.txt
```

### **2. Configuration:**
- Create `.env` file with API keys (optional for core functionality)
- See `README.md` for details

### **3. Run System:**
```bash
python app.py
```
- Access at: http://localhost:8000

### **4. Test System:**
```bash
python training/test_system.py
```

### **5. Run Evaluation:**
```bash
python evaluation/evaluate_system.py
```

---

## 📁 **Project Structure**

```
Multi-Agent-Medical-Assistant/
├── agents/                    # Agent implementations
│   ├── agent_decision.py      # Routing & orchestration
│   ├── image_analysis_agent/  # Medical imaging agents
│   ├── rag_agent/            # Knowledge retrieval
│   └── web_search_processor_agent/
├── app.py                    # FastAPI server
├── config.py                 # Configuration
├── evaluation/               # Evaluation framework
├── training/                 # Training scripts
├── templates/                # Web UI
└── data/                     # Datasets & models
```

---

## 📝 **For Paper/Report**

### **Key Metrics:**
- Pneumonia: 97.84% accuracy (878 test images)
- System: Multi-agent architecture
- Routing: 76% accuracy (fallback mode)

### **Contributions:**
1. Novel multi-agent orchestration for medical imaging
2. Intelligent routing with robust fallback
3. High-performance specialized agents
4. Scalable framework design

---

## 🔧 **Technical Details**

- **Framework:** FastAPI, LangGraph, PyTorch
- **Models:** DenseNet121 (Pneumonia), U-Net (Skin Lesion)
- **Dataset:** Kaggle Chest X-Ray (5,863 images)
- **Evaluation:** 878 test images

---

## 📚 **References**

- DenseNet: Huang et al. (2017)
- U-Net: Ronneberger et al. (2015)
- LangGraph: LangChain (2024)

---

**Last Updated:** 2025-11-27

