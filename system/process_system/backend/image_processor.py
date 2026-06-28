import cv2
import numpy as np
from PIL import Image
import logging
import io

logger = logging.getLogger("ProductionInput.Processor")

class LocalImageProcessor:
    def __init__(self):
        # Using Haar Cascade for ultra-fast CPU face detection, reserving GPU for the main B3 Model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("[Initialization] ImageProcessor with OpenCV Face Tracker successfully loaded.")

    def process_raw_image(self, raw_bytes):
        try:
            # 1. GLOBAL STREAM (To capture Text-to-Image AI generation or whole-frame GANs)
            raw_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            global_img = raw_img
            
            # 2. FACE STREAM (To capture FaceSwap or micro-expressions)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_img is None:
                raise ValueError("Failed to decode image buffer.")

            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            face_img = None
            if len(faces) > 0:
                # Prioritize the largest face detected (main subject)
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                center_x, center_y = x + w // 2, y + h // 2
                
                # Tight crop (1.1x) to force AI attention onto skin/hair boundary artifacts
                box_size = int(max(w, h) * 1.1)
                x1 = max(0, center_x - box_size // 2)
                y1 = max(0, center_y - box_size // 2)
                x2 = min(cv_img.shape[1], center_x + box_size // 2)
                y2 = min(cv_img.shape[0], center_y + box_size // 2)
                
                cv_crop = cv_img[y1:y2, x1:x2]
                if cv_crop.size != 0:
                    cv_crop = np.ascontiguousarray(cv_crop)
                    face_img = Image.fromarray(cv2.cvtColor(cv_crop, cv2.COLOR_BGR2RGB))
                
            return {"global": global_img, "face": face_img}
            
        except Exception as e:
            logger.error(f"[Error] Failed to extract image streams: {str(e)}")
            return None