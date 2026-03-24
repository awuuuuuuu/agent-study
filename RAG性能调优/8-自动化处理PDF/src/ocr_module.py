import yaml
import os
import json
from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["ocr"]

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=self.config["use_doc_orientation_classify"],
            use_doc_unwarping=self.config["use_doc_unwarping"],
            use_textline_orientation=self.config["use_textline_orientation"],
            lang=self.config["language"]
        )

    def extract_text(self, file_path):
        # 执行OCR推理
        result = self.ocr.predict(file_path)
        for res in result:
            res.save_to_json("output")
        
        with open("output/example_res.json", "r", encoding="utf-8") as f:
            jsondata = f.read()
        return json.loads(jsondata)
    
if __name__ == "__main__":
    ocr_engine = OCREngine()

    base_url = os.path.dirname(__file__)

    image_path = os.path.join(base_url, "example.png")

    extracted_text = ocr_engine.extract_text(image_path)

    print(extracted_text.get("rec_texts", []))