import asyncio
from core.tools.workspace import Workspace
from core.vision.image_loader import ImageLoader
from core.vision.ollama_ocr import OllamaOCRProvider

async def test_image():
    ws = Workspace("workspace")
    loader = ImageLoader(ws)
    img_input = loader.load("Question.png")
    print(f"Loaded image: {img_input.image_id}, size: {len(img_input.bytes)} bytes")

    ocr = OllamaOCRProvider(base_url="http://127.0.0.1:11434", default_model="qwen2.5vl:3b", request_timeout_seconds=300)
    print("Running OCR extraction via recognize()...")
    res = await asyncio.to_thread(ocr.recognize, img_input)
    print("\n=== OCR TEXT OUTPUT ===")
    print(res.full_text)
    print("=======================")

if __name__ == "__main__":
    asyncio.run(test_image())
