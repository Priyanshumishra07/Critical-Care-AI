# Multi-Agent Medical Imaging Diagnostic System
## Comprehensive Technical Documentation

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Agent Implementations](#agent-implementations)
4. [Data Flow & Processing](#data-flow--processing)
5. [API Endpoints](#api-endpoints)
6. [Configuration System](#configuration-system)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Frontend Architecture](#frontend-architecture)
9. [Deployment](#deployment)
10. [Technical Specifications](#technical-specifications)

---

## System Architecture

### Overview

The Multi-Agent Medical Imaging Diagnostic System is a distributed, modular architecture built on FastAPI and LangGraph that orchestrates specialized AI agents for medical image analysis and knowledge retrieval.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application (app.py)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   /chat      │  │  /upload    │  │  /validate   │       │
│  │  Endpoint    │  │  Endpoint   │  │  Endpoint    │       │
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘       │
└─────────┼──────────────────┼─────────────────┼──────────────┘
          │                  │                 │
          └──────────────────┴─────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │   Agent Decision System              │
          │   (agent_decision.py)                │
          │   - LangGraph Orchestration          │
          │   - Intelligent Routing               │
          │   - Fallback Mechanisms              │
          └──────────────┬───────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌──────────┐
    │ Medical │   │   RAG    │   │   Web    │
    │ Imaging │   │  Agent   │   │  Search  │
    │ Agents  │   │          │   │  Agent   │
    └─────────┘   └──────────┘   └──────────┘
```

### Key Design Principles

1. **Modularity**: Each agent is independently implemented and can be added/removed without affecting others
2. **Fallback Mechanisms**: System continues functioning even when LLM services are unavailable
3. **State Management**: LangGraph maintains conversation state across agent interactions
4. **Scalability**: Architecture supports adding new medical imaging modalities easily

---

## Core Components

### 1. FastAPI Application (`app.py`)

**Purpose**: Main application entry point providing REST API endpoints

**Key Features**:
- RESTful API endpoints for chat, image upload, and validation
- Static file serving for uploaded images and results
- Session management via cookies
- Background tasks for audio cleanup
- Error handling and validation

**Key Functions**:

```python
@app.post("/chat")
def chat(request: QueryRequest, response: Response, session_id: Optional[str] = Cookie(None)):
    """
    Processes text queries through the multi-agent system.
    
    Flow:
    1. Receives query from frontend
    2. Calls process_query() from agent_decision.py
    3. Returns response with agent name and optional result images
    """
```

```python
@app.post("/upload")
async def upload_image(image: UploadFile, text: str = Form(""), ...):
    """
    Handles medical image uploads with optional text.
    
    Flow:
    1. Validates file type and size
    2. Saves file securely with UUID prefix
    3. Routes to agent_decision system
    4. Returns analysis results with visualization images
    5. Cleans up temporary files
    """
```

**Directory Structure**:
- `uploads/backend/`: Temporary storage for uploaded images
- `uploads/frontend/`: Frontend-accessible uploads
- `uploads/pneumonia_output/`: Pneumonia analysis visualizations
- `uploads/skin_lesion_output/`: Skin lesion segmentation results
- `uploads/speech/`: Temporary audio files (auto-cleaned every 5 minutes)

### 2. Configuration System (`config.py`)

**Purpose**: Centralized configuration management for all system components

**Architecture**:

```python
class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()    # LLM for routing
        self.conversation = ConversationConfig()        # LLM for chat
        self.rag = RAGConfig()                          # RAG system config
        self.medical_cv = MedicalCVConfig()            # CV models config
        self.web_search = WebSearchConfig()            # Web search config
        self.api = APIConfig()                          # API settings
        self.speech = SpeechConfig()                    # TTS/STT config
        self.validation = ValidationConfig()          # Human validation
```

**Key Configuration Classes**:

#### `AgentDecisoinConfig`
- **LLM**: GPT-4o-mini (temperature=0.1) for precise routing decisions
- **Purpose**: Determines which agent should handle each query

#### `MedicalCVConfig`
- **Model Paths**:
  - Pneumonia: `./agents/image_analysis_agent/pneumonia_agent/models/pneumonia_classification_model.pth`
  - Skin Lesion: `./agents/image_analysis_agent/skin_lesion_agent/models/skin_lesion_segmentation.pth`
  - Brain Tumor: `./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_model.pth`
  - COVID-19: `./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth`
- **LLM**: GPT-4o-mini (temperature=0.1) for image classification

#### `RAGConfig`
- **Vector DB**: Qdrant (local or cloud)
- **Embedding Model**: text-embedding-3-large (1536 dimensions)
- **Chunk Size**: 512 tokens
- **Chunk Overlap**: 50 tokens
- **Reranker**: cross-encoder/ms-marco-TinyBERT-L-6
- **Top K Retrieval**: 5 documents
- **Reranker Top K**: 3 documents
- **Min Confidence**: 0.40 (for auto-routing to web search)

#### `ValidationConfig`
- **Required Validation**:
  - `PNEUMONIA_AGENT`: True
  - `SKIN_LESION_AGENT`: True
  - `CONVERSATION_AGENT`: False
  - `RAG_AGENT`: False
- **Timeout**: 300 seconds
- **Default Action**: "reject"

**Environment Variables**:
```bash
OPENAI_API_KEY=sk-...              # Required for LLM features
ELEVEN_LABS_API_KEY=...            # Optional: TTS/STT
TAVILY_API_KEY=...                 # Optional: Web search
QDRANT_URL=...                     # Optional: Cloud Qdrant
QDRANT_API_KEY=...                 # Optional: Cloud Qdrant
HUGGINGFACE_TOKEN=...              # Optional: Reranker
```

### 3. Agent Decision System (`agents/agent_decision.py`)

**Purpose**: Core orchestration engine using LangGraph for multi-agent routing

**Architecture**:

```python
def create_agent_graph():
    """
    Creates LangGraph workflow:
    
    1. analyze_input() - Detects images, applies guardrails
    2. check_if_bypassing() - Checks guardrail bypass
    3. route_to_agent() - LLM or rule-based routing
    4. Agent execution (pneumonia, skin_lesion, conversation, etc.)
    5. END - Returns final state
    """
```

**State Management** (`AgentState`):

```python
class AgentState(MessagesState):
    messages: List[BaseMessage]           # Conversation history
    agent_name: Optional[str]             # Current agent
    current_input: Optional[Union[str, Dict]]  # Input (text or {text, image})
    has_image: bool                       # Image presence flag
    image_type: Optional[str]             # Detected image type
    output: Optional[str]                 # Final response
    needs_human_validation: bool          # Validation required
    retrieval_confidence: float           # RAG confidence score
    bypass_routing: bool                  # Guardrail bypass flag
    insufficient_info: bool                # RAG insufficient info flag
```

**Routing Logic**:

1. **LLM-Based Routing** (when API keys available):
   - Uses GPT-4o-mini with structured prompt
   - Returns JSON: `{agent, reasoning, confidence}`
   - Routes based on query content, image type, and context

2. **Fallback Routing** (when LLM unavailable):
   ```python
   def fallback_decision_chain(input_dict):
       input_text = input_dict.get("input", "")
       if "pneumonia" in input_text.lower() or "chest" in input_text.lower():
           return {"agent": "PNEUMONIA_AGENT", ...}
       elif "skin" in input_text.lower() or "lesion" in input_text.lower():
           return {"agent": "SKIN_LESION_AGENT", ...}
       else:
           return {"agent": "CONVERSATION_AGENT", ...}
   ```

**Graph Nodes**:

1. **analyze_input**: 
   - Detects images in input
   - Classifies image type using ImageClassifier
   - Applies guardrails if available
   - Sets `has_image` and `image_type` flags

2. **check_if_bypassing**:
   - Conditional routing based on guardrail results
   - Routes to `apply_guardrails` or `route_to_agent`

3. **route_to_agent**:
   - Makes routing decision (LLM or fallback)
   - Routes to appropriate agent node

4. **Agent Execution Nodes**:
   - `run_pneumonia_agent`
   - `run_skin_lesion_agent`
   - `run_brain_tumor_agent`
   - `run_chest_xray_agent`
   - `run_conversation_agent`
   - `run_rag_agent`
   - `run_web_search_agent`

5. **END**: Returns final state with response

---

## Agent Implementations

### 1. Image Analysis Agent (`agents/image_analysis_agent/`)

#### 1.1 Image Classifier (`image_classifier.py`)

**Purpose**: Classifies uploaded images as medical or non-medical and determines type

**Implementation**:

```python
class ImageClassifier:
    def __init__(self, vision_model=None):
        self.vision_model = vision_model  # GPT-4o-mini with vision
        
    def classify_image(self, image_path: str) -> Dict:
        """
        Classifies image type:
        - CHEST_XRAY (pneumonia detection)
        - SKIN_LESION (dermatology)
        - BRAIN_MRI_SCAN (brain tumor)
        - CHEST_XRAY_COVID (COVID-19)
        - NON_MEDICAL
        """
```

**Classification Methods**:

1. **LLM-Based** (when available):
   - Uses GPT-4o-mini vision capabilities
   - Analyzes image content and structure
   - Returns structured classification

2. **Fallback** (when LLM unavailable):
   ```python
   def _fallback_classify_image(self, image_path: str):
       """
       Uses filename keywords and image dimensions:
       - Keywords: 'chest', 'xray', 'x-ray' → CHEST_XRAY
       - Keywords: 'skin', 'lesion', 'dermatology' → SKIN_LESION
       - Dimensions: 224x224 typical for medical → Medical
       - Default: NON_MEDICAL
       """
   ```

#### 1.2 Pneumonia Agent (`pneumonia_agent/pneumonia_inference.py`)

**Purpose**: Classifies chest X-rays for pneumonia detection

**Model Architecture**:

```python
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # DenseNet121 backbone (ImageNet pre-trained)
        self.backbone = models.densenet121(weights=None)
        num_ftrs = self.backbone.classifier.in_features
        # Custom classification head
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
```

**Technical Details**:
- **Base Model**: DenseNet121
- **Input Size**: 224x224 RGB
- **Classes**: ['normal', 'pneumonia']
- **Preprocessing**:
  - Resize to 224x224
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Convert to tensor

**Inference Pipeline**:

```python
def predict(self, img_path):
    """
    1. Load and preprocess image
    2. Convert to tensor
    3. Forward pass through model
    4. Apply softmax for probabilities
    5. Return class with confidence
    """
```

**Performance**:
- **Accuracy**: 97.84%
- **Precision**: 97.86%
- **Recall**: 97.84%
- **F1-Score**: 97.84%
- **Inference Time**: ~0.06 seconds/image

#### 1.3 Skin Lesion Agent (`skin_lesion_agent/skin_lesion_inference.py`)

**Purpose**: Segments skin lesions in dermatology images

**Model Architecture**:
- **Base Model**: U-Net (pre-trained on ISIC2018)
- **Input**: Variable size RGB images
- **Output**: Binary segmentation mask

**Segmentation Pipeline**:

```python
def predict(self, image_path: str, output_path: str) -> bool:
    """
    1. Load image
    2. Preprocess (resize, normalize)
    3. Forward pass through U-Net
    4. Generate binary mask
    5. Create overlay visualization
    6. Save result image
    """
```

**Visualization**:
- Original image + segmentation mask overlay
- Color-coded lesion boundaries
- Confidence heatmap (if available)

**Performance**:
- **Inference Time**: ~0.49 seconds/image
- **Model**: Pre-trained U-Net (ISIC2018)

#### 1.4 Brain Tumor Agent (`brain_tumor_agent/brain_tumor_inference.py`)

**Purpose**: Detects brain tumors in MRI scans

**Implementation**: Similar architecture to pneumonia agent
- MRI-specific preprocessing
- Tumor detection classification
- Confidence scoring

#### 1.5 COVID-19 Chest X-ray Agent (`chest_xray_agent/covid_chest_xray_inference.py`)

**Purpose**: Classifies chest X-rays for COVID-19 detection

**Implementation**: Similar to pneumonia agent
- COVID-19 specific classification
- Multi-class: Normal, Pneumonia, COVID-19

### 2. RAG Agent (`agents/rag_agent/`)

**Purpose**: Retrieval-Augmented Generation for medical knowledge queries

**Architecture**:

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Query Expander   │  (Expands query with synonyms)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Vector Search    │  (Qdrant similarity search)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Reranker         │  (Cross-encoder reranking)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Response Gen     │  (LLM generates answer)
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

**Components**:

#### 2.1 Document Parser (`doc_parser.py`)
- **Technology**: Docling (IBM)
- **Capabilities**:
  - Extracts text from PDFs
  - Extracts tables
  - Extracts images
  - Preserves document structure

#### 2.2 Content Processor (`content_processor.py`)
- **Image Summarization**: Uses LLM to describe extracted images
- **Document Formatting**: Combines text and image summaries
- **Semantic Chunking**: Splits documents into semantic sections using LLM

#### 2.3 Vector Store (`vectorstore_qdrant.py`)
- **Database**: Qdrant (local or cloud)
- **Embeddings**: OpenAI text-embedding-3-large (1536 dims)
- **Distance Metric**: Cosine similarity
- **Operations**:
  - Create collection
  - Add documents
  - Similarity search
  - Metadata filtering

#### 2.4 Query Expander (`query_expander.py`)
- **Purpose**: Expands queries with medical synonyms
- **Method**: LLM-based expansion
- **Example**: "pneumonia" → "pneumonia, lung infection, respiratory infection"

#### 2.5 Reranker (`reranker.py`)
- **Model**: cross-encoder/ms-marco-TinyBERT-L-6 (HuggingFace)
- **Purpose**: Reranks retrieved documents for relevance
- **Top K**: 3 documents after reranking

#### 2.6 Response Generator (`response_generator.py`)
- **LLM**: GPT-4o-mini (temperature=0.3)
- **Context Window**: 8192 tokens
- **Features**:
  - Includes source citations
  - Handles insufficient information
  - Maintains conversation context

**Ingestion Pipeline**:

```python
def ingest_file(self, document_path: str):
    """
    1. Parse document (extract text, tables, images)
    2. Summarize images using LLM
    3. Format document with image summaries
    4. Chunk document semantically
    5. Generate embeddings
    6. Store in Qdrant vector database
    """
```

### 3. Web Search Agent (`agents/web_search_processor_agent/`)

**Purpose**: Retrieves recent medical information from web sources

**Components**:

#### 3.1 Tavily Search (`tavily_search.py`)
- **API**: Tavily Search API
- **Purpose**: General web search for medical topics
- **Returns**: URLs, snippets, titles

#### 3.2 PubMed Search (`pubmed_search.py`)
- **API**: PubMed E-utilities
- **Purpose**: Academic medical literature search
- **Returns**: Research papers, abstracts

#### 3.3 Web Search Processor (`web_search_processor.py`)
- **Orchestration**: Combines Tavily and PubMed results
- **LLM Processing**: Summarizes and formats results
- **Context Management**: Maintains conversation history

### 4. Conversation Agent

**Purpose**: Handles general conversation and medical Q&A

**Implementation**:
- **LLM**: GPT-4o-mini (temperature=0.7)
- **Context**: Maintains last 20 messages (10 Q&A pairs)
- **Capabilities**:
  - General conversation
  - Medical question answering
  - Follow-up questions
  - Image analysis interpretation

**Fallback**: When LLM unavailable, returns informative message about API key requirements

---

## Data Flow & Processing

### Text Query Flow

```
User Input (Text)
    │
    ▼
FastAPI /chat endpoint
    │
    ▼
process_query() in agent_decision.py
    │
    ▼
analyze_input() node
    │
    ├─→ Guardrails check (if available)
    │
    ▼
route_to_agent() node
    │
    ├─→ LLM routing (if available)
    │   └─→ JSON: {agent, reasoning, confidence}
    │
    └─→ Fallback routing (if LLM unavailable)
        └─→ Rule-based keyword matching
    │
    ▼
Agent Execution Node
    │
    ├─→ CONVERSATION_AGENT
    ├─→ RAG_AGENT
    ├─→ WEB_SEARCH_AGENT
    │
    ▼
Response Generation
    │
    ▼
Return to FastAPI
    │
    ▼
JSON Response to Frontend
```

### Image Upload Flow

```
User Uploads Image
    │
    ▼
FastAPI /upload endpoint
    │
    ├─→ File validation (type, size)
    ├─→ Secure filename generation (UUID prefix)
    ├─→ Save to uploads/backend/
    │
    ▼
process_query({text, image}) in agent_decision.py
    │
    ▼
analyze_input() node
    │
    ├─→ ImageClassifier.classify_image()
    │   ├─→ LLM vision (if available)
    │   └─→ Fallback (filename, dimensions)
    │
    ├─→ Sets has_image=True
    ├─→ Sets image_type (CHEST_XRAY, SKIN_LESION, etc.)
    │
    ▼
route_to_agent() node
    │
    ├─→ Routes based on image_type
    │   ├─→ CHEST_XRAY → PNEUMONIA_AGENT
    │   ├─→ SKIN_LESION → SKIN_LESION_AGENT
    │   ├─→ BRAIN_MRI → BRAIN_TUMOR_AGENT
    │   └─→ CHEST_XRAY_COVID → CHEST_XRAY_AGENT
    │
    ▼
Agent Execution
    │
    ├─→ Pneumonia Agent
    │   ├─→ Load DenseNet121 model
    │   ├─→ Preprocess image
    │   ├─→ Inference
    │   ├─→ Generate visualization
    │   └─→ Save to uploads/pneumonia_output/
    │
    ├─→ Skin Lesion Agent
    │   ├─→ Load U-Net model
    │   ├─→ Preprocess image
    │   ├─→ Segmentation
    │   ├─→ Create overlay
    │   └─→ Save to uploads/skin_lesion_output/
    │
    ▼
Response with result_image URL
    │
    ▼
Cleanup temporary upload file
    │
    ▼
Return JSON to Frontend
```

### RAG Query Flow

```
User Query
    │
    ▼
RAG Agent
    │
    ├─→ Query Expander
    │   └─→ Expand with synonyms
    │
    ▼
Vector Search (Qdrant)
    │
    ├─→ Embed query
    ├─→ Similarity search (top 5)
    │
    ▼
Reranker
    │
    ├─→ Cross-encoder reranking
    ├─→ Select top 3
    │
    ▼
Response Generator
    │
    ├─→ Format context
    ├─→ LLM generation
    ├─→ Include sources
    │
    ▼
Return Response
```

---

## API Endpoints

### 1. `GET /`
- **Purpose**: Serve main HTML page
- **Response**: HTML template
- **Authentication**: None

### 2. `GET /health`
- **Purpose**: Health check endpoint
- **Response**: `{"status": "healthy"}`
- **Use Case**: Docker health checks, monitoring

### 3. `POST /chat`
- **Purpose**: Process text queries
- **Request Body**:
  ```json
  {
    "query": "What are the symptoms of pneumonia?",
    "conversation_history": []
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "response": "Pneumonia symptoms include...",
    "agent": "RAG_AGENT"
  }
  ```
- **Session**: Cookie-based session management

### 4. `POST /upload`
- **Purpose**: Process image uploads
- **Request**: Multipart form data
  - `image`: File (PNG, JPG, JPEG)
  - `text`: Optional text query
- **Response**:
  ```json
  {
    "status": "success",
    "response": "Analysis complete...",
    "agent": "PNEUMONIA_AGENT",
    "result_image": "/uploads/pneumonia_output/pneumonia_analysis.png"
  }
  ```
- **File Size Limit**: 5MB (configurable)
- **Validation**: File type and size checks

### 5. `POST /validate`
- **Purpose**: Human validation of medical outputs
- **Request**: Form data
  - `validation_result`: "yes" or "no"
  - `comments`: Optional comments
- **Response**: Validation confirmation
- **Use Case**: Human-in-the-loop validation for medical diagnoses

### 6. `POST /transcribe`
- **Purpose**: Speech-to-text transcription
- **Request**: Audio file (WebM, MP3)
- **Response**: `{"transcript": "transcribed text"}`
- **Technology**: ElevenLabs API
- **Features**: Diarization, audio event tagging

### 7. `POST /generate-speech`
- **Purpose**: Text-to-speech generation
- **Request Body**:
  ```json
  {
    "text": "Response text",
    "voice_id": "21m00Tcm4TlvDq8ikWAM"
  }
  ```
- **Response**: MP3 audio file
- **Technology**: ElevenLabs API

### Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `413`: Payload too large
- `500`: Internal server error

---

## Model Training & Evaluation

### Pneumonia Model Training

**Script**: `training/train_pneumonia_model.py`

**Dataset**:
- **Source**: Kaggle Chest X-Ray Dataset
- **Total Images**: 5,863
- **Split**:
  - Train: 4,087 images
  - Validation: 875 images
  - Test: 878 images
- **Classes**: Normal, Pneumonia

**Training Configuration**:

```python
# Model
model = DenseNet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Data Augmentation
transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**Training Process**:

1. **Data Loading**: Custom DataLoader with augmentation
2. **Training Loop**:
   - Forward pass
   - Loss calculation
   - Backward pass
   - Optimizer step
   - Validation after each epoch
3. **Model Saving**: Best model based on validation accuracy
4. **Visualization**: Training history plots (loss, accuracy)

**Results**:
- **Final Accuracy**: 97.84%
- **Precision**: 97.86%
- **Recall**: 97.84%
- **F1-Score**: 97.84%
- **Model Saved**: `pneumonia_classification_model.pth`

### Evaluation Framework

**Script**: `evaluation/evaluate_system.py`

**Evaluation Metrics**:

1. **Pneumonia Agent**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Per-class metrics
   - Inference time

2. **Skin Lesion Agent**:
   - Qualitative evaluation (no ground truth)
   - Inference time
   - Segmentation quality assessment

3. **Routing System**:
   - Routing accuracy
   - Fallback effectiveness
   - Agent selection correctness

4. **System-Wide**:
   - End-to-end latency
   - Error rates
   - Resource usage

**Evaluation Report**:
- JSON format: `evaluation/results/evaluation_report.json`
- Visualizations: Confusion matrices, performance charts

---

## Frontend Architecture

### Technology Stack

- **HTML5**: Structure
- **CSS3**: Styling (Bootstrap 5)
- **JavaScript**: Interactivity
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icons
- **Marked.js**: Markdown rendering
- **jsPDF**: PDF export
- **html2canvas**: Screenshot capture

### Key Features

1. **Chat Interface**:
   - Real-time message display
   - Markdown rendering
   - Image preview
   - Agent tags

2. **Image Upload**:
   - Drag-and-drop support
   - File validation
   - Preview before upload
   - Progress indicators

3. **Dark Mode**:
   - Toggle button
   - Persistent preference (localStorage)
   - Smooth transitions

4. **Chat History**:
   - Automatic persistence (localStorage)
   - Restore on page load
   - Clear history option

5. **PDF Export**:
   - Export full chat history
   - Formatted PDF with timestamps
   - Includes images and responses

6. **Voice Features**:
   - Speech-to-text (ElevenLabs)
   - Text-to-speech playback
   - Voice recording button

### Frontend-Backend Communication

```javascript
// Chat submission
fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: message})
})

// Image upload
const formData = new FormData();
formData.append('image', file);
formData.append('text', textQuery);
fetch('/upload', {
    method: 'POST',
    body: formData
})
```

### State Management

- **localStorage**: Chat history, theme preference
- **Session cookies**: Backend session management
- **DOM state**: UI state (open/closed panels, etc.)

---

## Deployment

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

**Build & Run**:
```bash
docker build -t medical-assistant .
docker run -p 8000:8000 medical-assistant
```

### Environment Setup

1. **Create `.env` file**:
```bash
OPENAI_API_KEY=sk-...
ELEVEN_LABS_API_KEY=...
TAVILY_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
HUGGINGFACE_TOKEN=...
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run Application**:
```bash
python app.py
```

4. **Access**: http://localhost:8000

### Production Considerations

1. **Security**:
   - API key management (environment variables)
   - File upload validation
   - Rate limiting
   - CORS configuration

2. **Performance**:
   - Model caching
   - Async processing for long operations
   - Background tasks for cleanup
   - CDN for static assets

3. **Monitoring**:
   - Health check endpoint
   - Logging
   - Error tracking
   - Performance metrics

---

## Technical Specifications

### System Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **CUDA**: Optional (for GPU acceleration)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB+ for models and data

### Dependencies

**Core**:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `langchain`: LLM framework
- `langgraph`: Agent orchestration
- `torch`: Deep learning
- `torchvision`: Computer vision models
- `pillow`: Image processing
- `numpy`: Numerical operations

**RAG**:
- `qdrant-client`: Vector database
- `docling`: Document parsing
- `sentence-transformers`: Embeddings (optional)

**Web Search**:
- `tavily-python`: Web search API
- `requests`: HTTP client

**Speech**:
- `elevenlabs`: TTS/STT API
- `pydub`: Audio processing

### Model Specifications

#### Pneumonia Model
- **Architecture**: DenseNet121
- **Parameters**: ~8M
- **Input**: 224x224 RGB
- **Output**: 2 classes (normal, pneumonia)
- **Size**: ~30MB
- **Inference**: CPU: ~60ms, GPU: ~10ms

#### Skin Lesion Model
- **Architecture**: U-Net
- **Input**: Variable size RGB
- **Output**: Binary segmentation mask
- **Size**: ~50MB
- **Inference**: ~490ms

### Performance Metrics

- **API Latency**: <100ms (text queries), <1s (image analysis)
- **Throughput**: 10 requests/second (configurable)
- **Model Loading**: ~2-3 seconds (first request)
- **Memory Usage**: ~2GB (with models loaded)

### Limitations

1. **API Dependencies**:
   - LLM features require OpenAI API keys
   - Web search requires Tavily API key
   - Speech requires ElevenLabs API key

2. **Model Limitations**:
   - Pneumonia model trained on specific dataset
   - Skin lesion model pre-trained (not fine-tuned)
   - No real-time training capability

3. **Scalability**:
   - Single-threaded model inference
   - No distributed processing
   - Limited concurrent request handling

---

## Conclusion

This technical documentation provides a comprehensive overview of the Multi-Agent Medical Imaging Diagnostic System. The architecture is designed for modularity, scalability, and robustness, with fallback mechanisms ensuring functionality even when external services are unavailable.

For questions or contributions, please refer to the main README.md or project repository.

**Last Updated**: 2025-01-27

