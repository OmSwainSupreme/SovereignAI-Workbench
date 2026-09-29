import asyncio
import time
from pathlib import Path
from core.tools.workspace import Workspace
from core.vision.image_loader import ImageLoader
from core.vision.ollama_vision import OllamaVisionProvider
from core.agent.types import ToolCall
from core.vision.tools import _analyze_image_callable


async def test_format(ext: str, filename: str):
    print(f"\n==========================================")
    print(f"Testing {ext.upper()} format: {filename}")
    print(f"==========================================")
    workspace = Workspace(r"C:\sovereign-ai\workspace")
    loader = ImageLoader(workspace)
    
    # 1. Load image and check metadata
    img = loader.load(filename)
    print(f"1. ImageLoader.load: Content-Type={img.metadata.content_type}, Size={img.metadata.size_bytes} bytes")
    assert img.metadata.content_type in ("image/png", "image/jpeg", "image/jpg")
    
    # 2. Vision provider check
    provider = OllamaVisionProvider(
        base_url="http://127.0.0.1:11434",
        default_model="qwen2.5vl:3b",
        request_timeout_seconds=900,
    )
    
    # 3. Tool callable execution
    callable_tool = _analyze_image_callable(provider, loader)
    tool_call = ToolCall(
        call_id=f"test_{ext}_call",
        tool_name="analyze_image",
        arguments={"path": filename, "prompt": "Read the questions in this image and summarize what they are about in 1 brief sentence."},
    )
    
    start_time = time.time()
    print(f"2. Executing analyze_image with qwen2.5vl:3b...")
    result = await callable_tool(tool_call.arguments)
    elapsed = time.time() - start_time
    
    print(f"3. Result keys: {list(result.keys()) if isinstance(result, dict) else result}")
    if isinstance(result, dict) and "error" in result:
        print(f"   Error: {result['error']}")
        assert False, f"analyze_image failed for {ext}: {result['error']}"
    
    description = result.get("description", "") if isinstance(result, dict) else str(result)
    print(f"4. Result output ({elapsed:.1f}s): {description[:300]}")
    assert len(description) > 10, f"Description too short for {ext}"
    print(f"SUCCESS: {ext.upper()} format analyzed successfully by qwen2.5vl:3b!")


async def main():
    # Test JPG
    await test_format("jpg", "test.jpg")
    # Test JPEG
    await test_format("jpeg", "test.jpeg")
    # Test PNG
    await test_format("png", "test_image.png")
    print("\nALL 3 IMAGE FORMATS (PNG, JPG, JPEG) PASSED REAL OLLAMA VISION ANALYSIS!")


if __name__ == "__main__":
    asyncio.run(main())
