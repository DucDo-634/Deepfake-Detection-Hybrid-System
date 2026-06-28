import torch
import gc
import logging
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Đã sửa lại đường dẫn Import chuẩn xác
from backend.image_processor import LocalImageProcessor
from backend.analysis_engine import LocalAnalysisEngine
from backend.verification_engine import LocalVerificationEngine

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("FastAPIServer")

app = FastAPI(title="Digital Forensics - Deepfake Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"[Boot Sequence] Server attached to compute device: {device}")

processor = LocalImageProcessor()
try:
    analyzer = LocalAnalysisEngine(device=device)
except Exception as e:
    logger.error(f"❌ [Critical Error] Failed to load model weights: {str(e)}")
    analyzer = None

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    if analyzer is None:
        return JSONResponse(status_code=500, content={"error": "System offline. Model weights were not loaded properly."})
        
    logger.info(f"🔔 [Incoming Request] Analyzing file: {file.filename}")
    try:
        raw_bytes = await file.read()
        streams = processor.process_raw_image(raw_bytes)
        
        if streams is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format or corrupted file."})
            
        logger.info("▶ INITIATING STREAM 1: GLOBAL (Detecting Diffusion/GAN anomalies)")
        g_score = analyzer.run_analysis(streams["global"])
        
        f_score = None
        if streams["face"] is not None:
            logger.info("▶ INITIATING STREAM 2: FACE CROP (Detecting FaceSwap/Reenactment artifacts)")
            f_score = analyzer.run_analysis(streams["face"])
        else:
            logger.info("▶ Skipping Stream 2 (No biological human face detected)")
            
        final_report = LocalVerificationEngine.generate_dual_report(g_score, f_score)
        
        # Aggressive memory management for stability under heavy loads
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return final_report
        
    except Exception as e:
        logger.error(f"[Internal Server Error] {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main_run:app", host="0.0.0.0", port=8000, reload=False)