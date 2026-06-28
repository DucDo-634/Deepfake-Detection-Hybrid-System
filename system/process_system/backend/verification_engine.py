import logging

logger = logging.getLogger("OutputLayer.Engine")

class LocalVerificationEngine:
    @staticmethod
    def generate_dual_report(g_score, f_score):
        # Determine the lowest score (highest anomaly probability) as the final verdict
        final_score = g_score
        trigger = "GLOBAL STREAM (Whole Image)"
        
        if f_score is not None and f_score < g_score:
            final_score = f_score
            trigger = "FACE STREAM (Cropped Region)"
            
        # =========================================================================
        # 3-TIER EVALUATION PROTOCOL: >= 0.65 (Real), 0.35-0.64 (Suspicious), < 0.35 (Fake)
        # =========================================================================
        
        # TIER 1: AUTHENTIC / NATURAL CAPTURE
        if final_score >= 0.65:
            label = "AUTHENTIC IMAGE (REAL)"
            conf = final_score * 100.0
            comment = "Uniform spatial and spectral noise distribution. No traces of AI generation or manipulation detected."
            
        # TIER 2: SUSPICIOUS / HEAVILY EDITED
        elif 0.35 <= final_score < 0.65:
            label = "SUSPICIOUS (HEAVILY EDITED)"
            # Map score to a logical confidence percentage
            conf = max(final_score, 1.0 - final_score) * 100.0
            comment = "⚠️ Low confidence region. Potential indicators of aggressive digital filtering, heavy compression, or subtle retouching. Manual verification advised."
            
        # TIER 3: AI GENERATED / DEEPFAKE
        else:
            label = "AI GENERATED (DEEPFAKE)"
            # Confidence in FAKE classification
            conf = (1.0 - final_score) * 100.0
            
            if trigger == "GLOBAL STREAM (Whole Image)":
                comment = "🚨 RED ALERT: FFT module detected decomposed frequency spectrums. Image exhibits strong traits of Text-to-Image synthesis (e.g., Diffusion/GAN)."
            else:
                comment = "🚨 RED ALERT: SRM module detected structural artifact disruptions in the facial region. High probability of FaceSwap or Reenactment manipulation."

        # Cap confidence for display logic
        conf = max(50.0, min(99.99, conf))
        
        print(f"\n==================================================")
        print(f"      DIGITAL FORENSICS ANALYSIS REPORT   ")
        print(f"==================================================")
        print(f"  FINAL VERDICT : {label} ({conf:.2f}%)")
        print(f"  TRIGGERED BY  : {trigger}")
        print(f"  OBSERVATIONS  : {comment}")
        print(f"==================================================\n")
        
        return {
            "status": label, 
            "confidence": f"{conf:.2f}%", 
            "metrics": {
                "final_ai_score": round(final_score, 4),
                "trigger_stream": trigger,
                "evaluation": comment
            }
        }   