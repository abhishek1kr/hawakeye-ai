import json
from loguru import logger

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class LLMReporter:
    """Generates natural language summaries of road conditions using a lightweight LLM."""
    
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = "auto"):
        self.model_name = model_name
        self.generator = None
        if pipeline is None:
            logger.warning("transformers not installed. LLM Reporter disabled.")
            return

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            device = "cpu"
        else:
            try:
                device = int(device)
            except:
                device = "cpu"

        logger.info(f"Loading LLM Reporter: {model_name} on {device}...")
        try:
            # Use text2text-generation for FLAN-T5 or text-generation for Qwen/Llama
            task = "text2text-generation" if "t5" in model_name.lower() else "text-generation"
            self.generator = pipeline(task, model=model_name, device=device)
            logger.info("LLM Reporter loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM Reporter: {e}")

    def summarize(self, report_dict: dict) -> str:
        if not self.generator:
            return "AI summarization is unavailable."
            
        try:
            o = report_dict.get("overall", {})
            c = report_dict.get("cracks", {})
            b = report_dict.get("maintenance_budget", {})
            
            prompt = (
                f"Write a short, professional 2-sentence summary of this road condition. "
                f"The road safety score is {o.get('safety_score')}/100, which is considered {o.get('risk_level')}. "
                f"The surface is {o.get('surface_type')}. "
                f"Total crack coverage is {c.get('total_avg_coverage_pct', 0)}%. "
                f"Estimated repair cost is {b.get('total_estimated_cost_inr', 0)} rupees."
            )
            
            if "text2text" in self.generator.task:
                out = self.generator(prompt, max_new_tokens=50)
                summary = out[0]['generated_text'].strip()
            else:
                out = self.generator(prompt, max_new_tokens=50, return_full_text=False)
                summary = out[0]['generated_text'].strip()
                
            return summary
        except Exception as e:
            logger.error(f"LLM Summarization failed: {e}")
            return "AI summarization failed."
