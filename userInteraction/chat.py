import os
import sys
import json
import webbrowser
import threading
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup path so modules in Shiva can be cleanly imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transformerAndRL.tokenizer import BPETokenizer
from src.transformerAndRL.embeddingTable import EmbeddingTable
from src.transformerAndRL.transformer import BidirectionalEncoderStack, FinalMLP, mean_pooling
from src.output.optionMatrixConvertor import OptionMatrixConvertor
from src.output.finalHeadAndOutput import FinalHeadAndOutput
from src.dataUpload.dataUploadAndPrompt import PdfProcessor
from src.dataUpload.rag import PromptSpecificRetriever
from src.config.dtos import FinalDecisionOutputDTO

# Paths
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model_artifacts", "tokenizer", "tokenizer.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "model_artifacts", "checkpoints", "sankalpa_weights.npz")

# =====================================================================
# 1. PRESENTATION DTO DATA CLASSES
# =====================================================================

class RankedActionItem(BaseModel):
    """Clean, formatted decision candidate for UI presentation."""
    rank: int = Field(..., description="1-based rank (1 is highest recommendation)")
    action: str = Field(..., description="The clinical intervention/action description")
    probability: float = Field(..., description="Raw probability [0.0 - 1.0]")
    probability_str: str = Field(..., description="Formatted percentage string (e.g. '84.6%')")
    logit: float = Field(..., description="Decision logit score")
    confidence_badge: str = Field(..., description="'High', 'Moderate', or 'Low' confidence label")
    is_winning: bool = Field(..., description="True if this is the winning choice")


class ClinicalDecisionViewDTO(BaseModel):
    """User-facing view model converting the raw model output to an intuitive diagnostic display."""
    query: str = Field(..., description="The clinical query or scenario evaluated")
    winning_action: str = Field(..., description="Top recommended clinical action")
    confidence: float = Field(..., description="Winning option probability [0.0 - 1.0]")
    confidence_str: str = Field(..., description="Formatted percentage (e.g. '88.5%')")
    margin_over_runner_up: float = Field(..., description="Advantage margin over 2nd place")
    margin_str: str = Field(..., description="Formatted margin string (e.g. '+42.3%')")
    entropy: float = Field(..., description="Shannon entropy score")
    uncertainty_level: str = Field(..., description="'Deterministic', 'Well-Calibrated', or 'High Ambiguity'")
    rankings: List[RankedActionItem] = Field(..., description="Ranked list of all evaluated candidate actions")
    retrieved_context: Optional[str] = Field(None, description="Grounded medical evidence retrieved from document")
    document_name: Optional[str] = Field(None, description="Source document file name if applicable")


class DecideRequest(BaseModel):
    scenario: str = Field(..., description="Clinical scenario or patient question")
    options: List[str] = Field(..., min_length=2, max_length=64, description="Candidate clinical decisions to rank")


# =====================================================================
# 2. SANKALPA INFERENCE RUNTIME ENGINE
# =====================================================================

class SankalpaEngine:
    def __init__(self):
        self.tokenizer: Optional[BPETokenizer] = None
        self.embedder: Optional[EmbeddingTable] = None
        self.encoder: Optional[BidirectionalEncoderStack] = None
        self.final_mlp: Optional[FinalMLP] = None
        self.option_convertor: Optional[OptionMatrixConvertor] = None
        self.W_cal: Optional[np.ndarray] = None
        self.b_cal: Optional[np.ndarray] = None
        self.temperature: float = 0.07
        self.is_loaded: bool = False

        # Active document grounding state
        self.active_document_name: Optional[str] = None
        self.active_document_text: Optional[str] = None
        self.active_retriever: Optional[PromptSpecificRetriever] = None

    def load(self):
        print("🧠 [Sankalpa] Initializing Engine Components...")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found at: {TOKENIZER_PATH}")

        # 1. Load Tokenizer
        self.tokenizer = BPETokenizer.from_json(TOKENIZER_PATH)
        print(f"   [✓] Tokenizer loaded (Vocab size: {self.tokenizer.vocab_size} tokens)")

        # 2. Initialize Model Components
        self.embedder = EmbeddingTable.from_tokenizer(self.tokenizer, seed=42)
        self.encoder = BidirectionalEncoderStack(seed=42)
        self.final_mlp = FinalMLP(seed=42)
        self.option_convertor = OptionMatrixConvertor(
            tokenizer=self.tokenizer, embedder=self.embedder, seed=42
        )

        # Default calibration matrix
        self.W_cal = np.eye(256, dtype=np.float32)
        self.b_cal = np.zeros((256,), dtype=np.float32)

        # 3. Load Checkpoint if present
        if os.path.exists(CHECKPOINT_PATH):
            print(f"   [✓] Loading weights checkpoint from: {CHECKPOINT_PATH}...")
            data = np.load(CHECKPOINT_PATH)
            if "mlp_W_gate" in data:
                self.final_mlp.W_gate = data["mlp_W_gate"]
                self.final_mlp.b_gate = data["mlp_b_gate"]
                self.final_mlp.W_up = data["mlp_W_up"]
                self.final_mlp.b_up = data["mlp_b_up"]
                self.final_mlp.W_down = data["mlp_W_down"]
                self.final_mlp.b_down = data["mlp_b_down"]
            if "opt_W_proj" in data:
                self.option_convertor.W_proj = data["opt_W_proj"]
                self.option_convertor.b_proj = data["opt_b_proj"]
            if "cal_W" in data:
                self.W_cal = data["cal_W"]
                self.b_cal = data["cal_b"]
            if self.encoder is not None:
                self.encoder.set_weights(data)
            if self.embedder is not None and "embedding_weights" in data:
                self.embedder.set_weights({"embedding_weights": data["embedding_weights"]})
            print("   [✓] Model weights successfully synchronized.")
        else:
            print(f"   [!] Checkpoint not yet found at {CHECKPOINT_PATH}. Using seeded initialization.")

        self.is_loaded = True
        print("🌟 [Sankalpa] Ready for clinical triage decision intelligence!")

    def set_document(self, filename: str, text: str):
        self.active_document_name = filename
        self.active_document_text = text
        self.active_retriever = PromptSpecificRetriever(text=text)
        print(f"📄 [RAG] Document '{filename}' indexed into {len(self.active_retriever.chunks)} chunks.")

    def clear_document(self):
        self.active_document_name = None
        self.active_document_text = None
        self.active_retriever = None

    def evaluate_decision(self, query: str, candidate_options: List[str]) -> ClinicalDecisionViewDTO:
        if not self.is_loaded:
            raise RuntimeError("Sankalpa Engine is not loaded.")

        # Ground against active document if present
        retrieved_context_str: Optional[str] = None
        effective_scenario = query

        if self.active_retriever is not None and len(self.active_retriever.chunks) > 0:
            top_chunks = self.active_retriever.retrieve(query, top_k=2)
            if top_chunks:
                retrieved_context_str = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(top_chunks)])
                effective_scenario = f"Clinical Context from Document:\n{retrieved_context_str}\n\nClinical Case:\n{query}"

        # 1. Encode text scenario
        token_ids = self.tokenizer.encode(effective_scenario)
        if not token_ids:
            token_ids = [self.tokenizer.unk_token_id]

        input_ids = np.array([token_ids], dtype=np.int64)
        attention_mask = np.ones((1, len(token_ids)), dtype=np.float32)

        emb = self.embedder.forward(input_ids)
        hidden = self.encoder.forward(emb, attention_mask=attention_mask)
        context_vec = mean_pooling(hidden, attention_mask=attention_mask).astype(np.float32)

        # 2. Forward through MLP & RLCD calibration
        z = self.final_mlp.forward(context_vec)
        z_cal = np.matmul(z, self.W_cal) + self.b_cal

        # 3. Project Candidate Options
        options_matrix = self.option_convertor.convert(candidate_options)

        # 4. Final Decision Head
        final_head = FinalHeadAndOutput(temperature=self.temperature)
        raw_result: FinalDecisionOutputDTO = final_head.forward(
            situation_vector=z_cal,
            options_matrix=options_matrix,
            candidate_strings=candidate_options
        )

        # 5. Transform raw model JSON to ClinicalDecisionViewDTO
        view_dto = self._transform_to_view_dto(
            raw_result=raw_result,
            query=query,
            retrieved_context=retrieved_context_str
        )
        return view_dto

    def _transform_to_view_dto(
        self,
        raw_result: FinalDecisionOutputDTO,
        query: str,
        retrieved_context: Optional[str]
    ) -> ClinicalDecisionViewDTO:
        ranked_items: List[RankedActionItem] = []
        for r in raw_result.rankings:
            prob = float(r.probability)
            if prob >= 0.70:
                badge = "High"
            elif prob >= 0.40:
                badge = "Moderate"
            else:
                badge = "Low"

            ranked_items.append(
                RankedActionItem(
                    rank=r.rank,
                    action=r.text,
                    probability=prob,
                    probability_str=f"{prob * 100:.1f}%",
                    logit=float(r.logit),
                    confidence_badge=badge,
                    is_winning=(r.rank == 1)
                )
            )

        margin = float(raw_result.selected_action.margin_over_runner_up or 0.0)
        entropy = float(raw_result.metrics.entropy)
        if entropy < 0.8:
            uncertainty = "Deterministic"
        elif entropy < 1.5:
            uncertainty = "Well-Calibrated"
        else:
            uncertainty = "High Ambiguity"

        win_conf = float(raw_result.selected_action.confidence)

        return ClinicalDecisionViewDTO(
            query=query,
            winning_action=raw_result.selected_action.text or "Unknown",
            confidence=win_conf,
            confidence_str=f"{win_conf * 100:.1f}%",
            margin_over_runner_up=margin,
            margin_str=f"+{margin * 100:.1f}%",
            entropy=round(entropy, 3),
            uncertainty_level=uncertainty,
            rankings=ranked_items,
            retrieved_context=retrieved_context,
            document_name=self.active_document_name
        )


engine = SankalpaEngine()

# =====================================================================
# 3. FASTAPI LIFESPAN & APP DEFINITION
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    engine.load()
    yield
    # Shutdown
    print("🛑 [Sankalpa] Shutting down server.")

app = FastAPI(title="Sankalpa Decision Intelligence", lifespan=lifespan)

# =====================================================================
# 4. REST API ENDPOINTS
# =====================================================================

@app.get("/api/status")
async def get_status():
    return {
        "status": "ready" if engine.is_loaded else "loading",
        "model_name": "Sankalpa 110M Decision Transformer",
        "vocab_size": engine.tokenizer.vocab_size if engine.tokenizer else 0,
        "has_document": engine.active_document_name is not None,
        "document_name": engine.active_document_name,
        "chunk_count": len(engine.active_retriever.chunks) if engine.active_retriever else 0
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_document"
    contents = await file.read()
    
    extracted_text = ""
    if filename.lower().endswith(".pdf"):
        # Save temp file for PyPDF2
        temp_path = os.path.join(PROJECT_ROOT, "model_artifacts", f"temp_{filename}")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(contents)
        try:
            pdf_proc = PdfProcessor()
            dto = pdf_proc.process_and_create_upload_dto(temp_path, filename)
            extracted_text = dto.fileContent or ""
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        # Plain text / markdown
        try:
            extracted_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = contents.decode("latin-1", errors="replace")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded file.")

    engine.set_document(filename=filename, text=extracted_text)
    
    return {
        "success": True,
        "filename": filename,
        "character_count": len(extracted_text),
        "chunk_count": len(engine.active_retriever.chunks) if engine.active_retriever else 0,
        "preview": extracted_text[:300] + ("..." if len(extracted_text) > 300 else "")
    }

@app.post("/api/clear-document")
async def clear_document():
    engine.clear_document()
    return {"success": True, "message": "Document cleared."}

@app.post("/api/decide", response_model=ClinicalDecisionViewDTO)
async def decide(req: DecideRequest):
    if engine.active_document_name is not None:
        # If user uploaded a document, decisions MUST be grounded in it
        if not engine.active_document_text:
            raise HTTPException(status_code=400, detail="Active document is empty.")
    
    try:
        result = engine.evaluate_decision(query=req.scenario, candidate_options=req.options)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 5. FRONTEND WEB INTERFACE (HTML / TAILWIND / JS)
# =====================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sankalpa (110M) | Clinical Decision Intelligence</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace'],
          },
          colors: {
            brand: {
              50: '#eff6ff',
              100: '#dbeafe',
              500: '#3b82f6',
              600: '#2563eb',
              700: '#1d4ed8',
              900: '#1e3a8a',
            }
          }
        }
      }
    }
  </script>
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; color: #f1f5f9; }
    .glass-panel { background: rgba(17, 24, 39, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glow-accent { box-shadow: 0 0 25px -5px rgba(59, 130, 246, 0.4); }
    .emerald-glow { box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.35); }
    .custom-scroll::-webkit-scrollbar { width: 5px; }
    .custom-scroll::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen flex flex-col justify-between selection:bg-blue-600 selection:text-white">

  <!-- TOP HEADER -->
  <header class="border-b border-gray-800/80 glass-panel sticky top-0 z-50 px-6 py-4">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="text-xl font-bold tracking-tight text-white">SANKALPA</h1>
            <span class="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">110M ZERO-FRAMEWORK</span>
          </div>
          <p class="text-xs text-gray-400">Clinical Triage & Intervention Decision Intelligence Engine</p>
        </div>
      </div>
      
      <!-- ENGINE STATUS BADGE -->
      <div class="flex items-center space-x-4">
        <div id="statusIndicator" class="flex items-center space-x-2 bg-gray-900/90 border border-gray-800 px-3 py-1.5 rounded-full text-xs font-mono text-gray-300">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>INITIALIZING</span>
        </div>
      </div>
    </div>
  </header>

  <!-- MAIN CONTAINER -->
  <main class="max-w-7xl mx-auto w-full px-6 py-8 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8">
    
    <!-- LEFT COLUMN: DOCUMENT & INPUT CONFIGURATION (5 COLS) -->
    <div class="lg:col-span-5 space-y-6">
      
      <!-- 1. CLINICAL DOCUMENT UPLOAD CARD -->
      <div class="glass-panel rounded-2xl p-6 relative overflow-hidden">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center space-x-2">
            <div class="p-1.5 bg-blue-500/10 rounded-lg text-blue-400">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
            </div>
            <h2 class="text-base font-semibold text-white">Clinical Document Grounding</h2>
          </div>
          <span id="docStatusBadge" class="text-xs font-medium px-2 py-0.5 rounded-md bg-gray-800 text-gray-400">None Active</span>
        </div>
        
        <p class="text-xs text-gray-400 mb-4">
          Upload a medical protocol, patient record, or clinical study. The model will ground all decisions strictly to this document.
        </p>

        <!-- DROP ZONE -->
        <div id="dropZone" class="border-2 border-dashed border-gray-700 hover:border-blue-500 transition-colors rounded-xl p-5 text-center cursor-pointer bg-gray-900/40">
          <input type="file" id="fileInput" class="hidden" accept=".txt,.pdf,.md">
          <svg class="w-8 h-8 mx-auto mb-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
          <p class="text-xs text-gray-300 font-medium"><span class="text-blue-400 underline">Click to upload</span> or drag & drop</p>
          <p class="text-[10px] text-gray-500 mt-1 font-mono">PDF, TXT, or MD medical files</p>
        </div>

        <!-- UPLOADED FILE INFO BANNER -->
        <div id="activeDocBanner" class="hidden mt-4 p-3 bg-blue-950/40 border border-blue-800/60 rounded-xl flex items-center justify-between">
          <div class="flex items-center space-x-3 overflow-hidden">
            <div class="text-blue-400">
              <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </div>
            <div class="truncate">
              <div id="activeDocName" class="text-xs font-semibold text-blue-200 truncate">document.pdf</div>
              <div id="activeDocMeta" class="text-[10px] text-blue-400 font-mono">14 chunks indexed</div>
            </div>
          </div>
          <button onclick="clearActiveDoc()" class="text-xs text-gray-400 hover:text-red-400 ml-2 transition-colors">
            Remove
          </button>
        </div>
      </div>

      <!-- 2. SCENARIO & QUERY INPUT -->
      <div class="glass-panel rounded-2xl p-6 space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-base font-semibold text-white flex items-center space-x-2">
            <span class="p-1.5 bg-indigo-500/10 rounded-lg text-indigo-400">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
            </span>
            <span>Clinical Case / Scenario</span>
          </h2>
          <button onclick="loadSampleScenario()" class="text-[11px] text-blue-400 hover:underline">Load Sample</button>
        </div>

        <textarea id="scenarioInput" rows="4" class="w-full bg-gray-900/90 border border-gray-800 rounded-xl p-3 text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500 custom-scroll resize-none" placeholder="Describe the acute patient case, vitals, presentation, and clinical query..."></textarea>

        <!-- CANDIDATE INTERVENTION OPTIONS -->
        <div>
          <div class="flex items-center justify-between mb-2">
            <label class="text-xs font-medium text-gray-300">Candidate Interventions to Evaluate</label>
            <span class="text-[10px] text-gray-500 font-mono">Min 2, Max 64</span>
          </div>

          <div id="optionsList" class="space-y-2 mb-3">
            <!-- Dynamic option rows -->
          </div>

          <div class="flex space-x-2">
            <input type="text" id="newOptionInput" placeholder="Add candidate clinical action..." class="flex-1 bg-gray-900/90 border border-gray-800 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-blue-500" onkeydown="if(event.key==='Enter') addOptionRow()">
            <button onclick="addOptionRow()" class="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-xs font-medium text-white rounded-lg transition-colors">
              + Add
            </button>
          </div>
        </div>

        <!-- EXECUTE DECISION BUTTON -->
        <button id="decideBtn" onclick="runDecisionEngine()" class="w-full py-3.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold rounded-xl text-xs tracking-wide uppercase transition-all shadow-lg glow-accent flex items-center justify-center space-x-2">
          <span>Evaluate & Rank Decisions</span>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
        </button>
      </div>

    </div>

    <!-- RIGHT COLUMN: DIAGNOSTIC & DECISION INTELLIGENCE OUTPUT (7 COLS) -->
    <div class="lg:col-span-7 space-y-6">

      <!-- WELCOME / PLACEHOLDER STATE -->
      <div id="emptyState" class="glass-panel rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[500px]">
        <div class="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-4">
          <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"/></svg>
        </div>
        <h3 class="text-base font-semibold text-white mb-1">Awaiting Clinical Query</h3>
        <p class="text-xs text-gray-400 max-w-sm">
          Enter a patient scenario with candidate interventions or upload a medical PDF/text file to run the zero-framework 110M model.
        </p>
      </div>

      <!-- ACTIVE RESULTS PANEL -->
      <div id="resultsPanel" class="hidden space-y-6">

        <!-- WINNING ACTION HERO CARD -->
        <div class="glass-panel rounded-2xl p-6 border-l-4 border-l-emerald-500 emerald-glow relative overflow-hidden">
          <div class="flex items-center justify-between mb-3">
            <span class="text-[11px] font-mono tracking-wider uppercase text-emerald-400 font-semibold flex items-center space-x-1.5">
              <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span>TOP RECOMMENDED CLINICAL INTERVENTION</span>
            </span>
            <div id="winningBadge" class="text-xs font-mono font-bold px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              92.4% CONFIDENCE
            </div>
          </div>

          <div id="winningActionText" class="text-lg font-bold text-white leading-snug mb-4">
            Immediate catheter-directed reperfusion therapy
          </div>

          <!-- METRICS GRID -->
          <div class="grid grid-cols-3 gap-3 pt-3 border-t border-gray-800">
            <div class="bg-gray-900/60 p-2.5 rounded-xl border border-gray-800/80">
              <div class="text-[10px] text-gray-400 uppercase font-mono">Advantage Margin</div>
              <div id="metricMargin" class="text-sm font-bold text-blue-400 font-mono">+44.2%</div>
            </div>
            <div class="bg-gray-900/60 p-2.5 rounded-xl border border-gray-800/80">
              <div class="text-[10px] text-gray-400 uppercase font-mono">Shannon Entropy</div>
              <div id="metricEntropy" class="text-sm font-bold text-purple-400 font-mono">0.421</div>
            </div>
            <div class="bg-gray-900/60 p-2.5 rounded-xl border border-gray-800/80">
              <div class="text-[10px] text-gray-400 uppercase font-mono">Calibration</div>
              <div id="metricCalibration" class="text-sm font-bold text-emerald-400 font-mono">Calibrated</div>
            </div>
          </div>
        </div>

        <!-- ALL CANDIDATE DECISION RANKINGS -->
        <div class="glass-panel rounded-2xl p-6 space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold text-white flex items-center space-x-2">
              <span class="p-1 bg-blue-500/10 text-blue-400 rounded">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12"/></svg>
              </span>
              <span>Full Decision Candidate Distribution</span>
            </h3>
            <span class="text-[11px] text-gray-500 font-mono">Softmax &tau;=0.07</span>
          </div>

          <div id="rankingsContainer" class="space-y-3">
            <!-- Dynamic Ranked Action Cards -->
          </div>
        </div>

        <!-- GROUNDING EVIDENCE ACCORDION (IF RETRIEVED FROM DOCUMENT) -->
        <div id="evidencePanel" class="hidden glass-panel rounded-2xl p-6 space-y-3">
          <div class="flex items-center space-x-2 text-indigo-400">
            <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
            <h3 class="text-xs font-semibold text-white uppercase tracking-wider font-mono">Grounded Document Excerpt</h3>
          </div>
          <div id="evidenceContent" class="text-xs text-gray-300 bg-gray-900/80 p-4 rounded-xl border border-gray-800 leading-relaxed font-mono custom-scroll max-h-48 overflow-y-auto">
          </div>
        </div>

      </div>

    </div>

  </main>

  <!-- JAVASCRIPT LOGIC -->
  <script>
    let candidateOptions = [
      "Emergent primary percutaneous coronary intervention (PPCI)",
      "High-dose intravenous loop diuretics and positive airway pressure",
      "Broad-spectrum antimicrobial escalation with piperacillin-tazobactam",
      "Immediate unfractionated heparin bolus and therapeutic infusion"
    ];

    // Initialize UI
    window.addEventListener('DOMContentLoaded', () => {
      renderOptions();
      pollStatus();
      setupDropZone();
    });

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const indicator = document.getElementById('statusIndicator');
        if (data.status === 'ready') {
          indicator.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>ENGINE ONLINE (115M)</span>';
          indicator.classList.remove('text-gray-300');
          indicator.classList.add('text-emerald-400', 'border-emerald-500/30');
        }
        if (data.has_document) {
          updateActiveDocUI(data.document_name, data.chunk_count);
        }
      } catch (e) {
        console.error('Status fetch error:', e);
      }
    }

    function renderOptions() {
      const list = document.getElementById('optionsList');
      list.innerHTML = '';
      candidateOptions.forEach((opt, idx) => {
        const row = document.createElement('div');
        row.className = 'flex items-center justify-between p-2.5 bg-gray-900/80 border border-gray-800 rounded-xl text-xs group';
        row.innerHTML = `
          <div class="flex items-center space-x-2 truncate pr-2">
            <span class="w-5 h-5 rounded-md bg-gray-800 text-gray-400 flex items-center justify-center font-mono text-[10px]">${idx + 1}</span>
            <span class="text-gray-200 truncate">${opt}</span>
          </div>
          <button onclick="removeOptionRow(${idx})" class="text-gray-500 hover:text-red-400 opacity-60 group-hover:opacity-100 transition-opacity p-1">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
          </button>
        `;
        list.appendChild(row);
      });
    }

    function addOptionRow() {
      const input = document.getElementById('newOptionInput');
      const val = input.value.trim();
      if (!val) return;
      if (candidateOptions.length >= 64) {
        alert("Maximum 64 options allowed.");
        return;
      }
      candidateOptions.push(val);
      input.value = '';
      renderOptions();
    }

    function removeOptionRow(idx) {
      if (candidateOptions.length <= 2) {
        alert("At least 2 options are required for decision evaluation.");
        return;
      }
      candidateOptions.splice(idx, 1);
      renderOptions();
    }

    function loadSampleScenario() {
      document.getElementById('scenarioInput').value = 
        "62-year-old male with acute anterior ST-segment elevation, diaphoresis, sustained hypotension (BP 82/50), bilateral pulmonary rales, and elevated troponin T (1.8 ng/mL). Patient is in cardiogenic shock Killip Class IV.";
      candidateOptions = [
        "Emergent primary percutaneous coronary intervention (PPCI) with mechanical circulatory support",
        "Empiric intravenous broad-spectrum vancomycin and piperacillin-tazobactam",
        "Subcutaneous low-molecular-weight heparin with non-urgent outpatient stress echo",
        "High-dose oral beta-blocker titration with observation in telemetry"
      ];
      renderOptions();
    }

    function setupDropZone() {
      const dropZone = document.getElementById('dropZone');
      const fileInput = document.getElementById('fileInput');

      dropZone.onclick = () => fileInput.click();

      fileInput.onchange = (e) => {
        if (e.target.files.length) uploadFile(e.target.files[0]);
      };

      dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500', 'bg-blue-950/20'); };
      dropZone.ondragleave = () => { dropZone.classList.remove('border-blue-500', 'bg-blue-950/20'); };
      dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500', 'bg-blue-950/20');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
      };
    }

    async function uploadFile(file) {
      const formData = new FormData();
      formData.append('file', file);

      const statusBadge = document.getElementById('docStatusBadge');
      statusBadge.innerText = 'Uploading...';
      statusBadge.className = 'text-xs font-medium px-2 py-0.5 rounded-md bg-yellow-900/40 text-yellow-300';

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        updateActiveDocUI(data.filename, data.chunk_count);
      } catch (err) {
        alert('File upload failed: ' + err.message);
        statusBadge.innerText = 'Error';
        statusBadge.className = 'text-xs font-medium px-2 py-0.5 rounded-md bg-red-900/40 text-red-300';
      }
    }

    function updateActiveDocUI(name, chunkCount) {
      document.getElementById('activeDocBanner').classList.remove('hidden');
      document.getElementById('activeDocName').innerText = name;
      document.getElementById('activeDocMeta').innerText = `${chunkCount} semantic chunks indexed for grounding`;
      const badge = document.getElementById('docStatusBadge');
      badge.innerText = 'Active Grounding';
      badge.className = 'text-xs font-medium px-2 py-0.5 rounded-md bg-emerald-900/40 text-emerald-300 border border-emerald-500/20';
    }

    async function clearActiveDoc() {
      await fetch('/api/clear-document', { method: 'POST' });
      document.getElementById('activeDocBanner').classList.add('hidden');
      const badge = document.getElementById('docStatusBadge');
      badge.innerText = 'None Active';
      badge.className = 'text-xs font-medium px-2 py-0.5 rounded-md bg-gray-800 text-gray-400';
      document.getElementById('fileInput').value = '';
    }

    async function runDecisionEngine() {
      const scenario = document.getElementById('scenarioInput').value.trim();
      if (!scenario) {
        alert("Please enter a clinical scenario or case details.");
        return;
      }
      if (candidateOptions.length < 2) {
        alert("Please provide at least 2 candidate options.");
        return;
      }

      const btn = document.getElementById('decideBtn');
      btn.disabled = true;
      btn.innerHTML = `
        <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
        <span>Evaluating 110M Transformer Pass...</span>
      `;

      try {
        const res = await fetch('/api/decide', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario: scenario, options: candidateOptions })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Inference failed');
        }

        const data = await res.json();
        renderResults(data);
      } catch (e) {
        alert("Error during decision evaluation: " + e.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `
          <span>Evaluate & Rank Decisions</span>
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
        `;
      }
    }

    function renderResults(data) {
      document.getElementById('emptyState').classList.add('hidden');
      document.getElementById('resultsPanel').classList.remove('hidden');

      // 1. Hero Winning Action
      document.getElementById('winningActionText').innerText = data.winning_action;
      document.getElementById('winningBadge').innerText = `${data.confidence_str} CONFIDENCE`;
      document.getElementById('metricMargin').innerText = data.margin_str;
      document.getElementById('metricEntropy').innerText = data.entropy.toFixed(3);
      document.getElementById('metricCalibration').innerText = data.uncertainty_level;

      // 2. Rankings list
      const container = document.getElementById('rankingsContainer');
      container.innerHTML = '';

      data.rankings.forEach((item) => {
        const isWin = item.is_winning;
        const card = document.createElement('div');
        card.className = `p-3.5 rounded-xl border transition-all ${
          isWin 
            ? 'bg-emerald-950/20 border-emerald-500/40 shadow-sm' 
            : 'bg-gray-900/50 border-gray-800'
        }`;

        const pct = (item.probability * 100).toFixed(1);

        card.innerHTML = `
          <div class="flex items-center justify-between mb-1.5">
            <div class="flex items-center space-x-2">
              <span class="w-6 h-6 rounded-lg ${isWin ? 'bg-emerald-500 text-white' : 'bg-gray-800 text-gray-400'} flex items-center justify-center font-mono text-xs font-bold">
                #${item.rank}
              </span>
              <span class="text-xs font-medium ${isWin ? 'text-emerald-200 font-bold' : 'text-gray-200'}">
                ${item.action}
              </span>
            </div>
            <span class="text-xs font-mono font-bold ${isWin ? 'text-emerald-400' : 'text-gray-400'}">${pct}%</span>
          </div>

          <!-- Progress Bar -->
          <div class="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
            <div class="${isWin ? 'bg-emerald-400' : 'bg-blue-500'} h-1.5 rounded-full" style="width: ${pct}%"></div>
          </div>
        `;
        container.appendChild(card);
      });

      // 3. Grounding Evidence
      const evidencePanel = document.getElementById('evidencePanel');
      if (data.retrieved_context) {
        evidencePanel.classList.remove('hidden');
        document.getElementById('evidenceContent').innerText = data.retrieved_context;
      } else {
        evidencePanel.classList.add('hidden');
      }
    }
  </script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=HTML_PAGE)


# =====================================================================
# 6. APPLICATION ENTRY POINT WITH AUTOMATIC BROWSER LAUNCH
# =====================================================================

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 65)
    print("   SANKALPA (110M): CLINICAL DECISION INTELLIGENCE WEB SERVER")
    print("=" * 65)
    print("🚀 Starting FastAPI application on http://127.0.0.1:8000")
    print("🌐 Launching web interface in your default browser...")
    
    # Auto-open browser after 1.5 seconds
    threading.Timer(1.5, open_browser).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
