import os
from ocr_module import OCREngine
from processor_module import TextProcessor
from dify_module import DifyUploader
from metadata_module import MetadataManager

def main(image_path):
    ocr_engine = OCREngine()
    text = ocr_engine.extract_text(image_path)
    print(f"OCR提取的文本: {text}")

    processor = TextProcessor()
    processed_text = processor.process(text)
    print(f"处理后的文本: {processed_text}")

    uploader = DifyUploader()
    document_id = uploader.upload_document_by_text(processed_text)
    print(f"文档上传成功，ID: {document_id}")

    metadata_manager = MetadataManager()
    metadata = [
        {'name': 'author', 'value': 'default_author'},
        {'name': 'doc_source', 'value': image_path}
    ]
    metadata_manager.bind_metadata(document_id, metadata)
    print(f"元数据绑定成功")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "example.png")
    main(image_path)