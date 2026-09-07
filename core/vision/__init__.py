"""SovereignAI Workbench — OCR + Vision Abstraction (Phase 5C).

Public surface for the OCR/Vision subsystem. All imports are from this module;
sub-modules are implementation details.

The subsystem provides two independent capabilities:

* **OCR** — extract text from images (scanned documents, photos, etc.)
* **Vision** — reason about image content (captioning, analysis, understanding)

Both are framework-agnostic, locally run, and use replaceable providers.

Example::

    from core.vision import (
        FakeOCRProvider,
        FakeVisionProvider,
        Workspace,
        ImageLoader,
        register_vision_tools,
    )
    from core.agent import DefaultToolRegistry

    workspace = Workspace(root_path="/data/workspace")
    loader = ImageLoader(workspace)

    # OCR
    ocr = FakeOCRProvider()
    img = loader.load("scans/report.png")
    result = ocr.recognize(img)
    print(result.full_text)

    # Vision
    vision = FakeVisionProvider()
    analysis = vision.analyze(img, "Describe what you see")
    print(analysis.description)

    # Agent tools
    registry = DefaultToolRegistry()
    register_vision_tools(registry, ocr, vision, loader)
"""
from core.vision.types import (
    # Image input types
    ImageInput,
    ImageMetadata,
    ImageDocument,
    SUPPORTED_IMAGE_CONTENT_TYPES,
    # OCR types
    OCRResult,
    OCRBlock,
    OCRLine,
    OCRWord,
    OCRBoundingBox,
    # Vision types
    VisionResult,
    VisionRegion,
    # Routing helpers
    build_vision_routing_request,
)
from core.vision.ocr import (
    OCRProvider,
    FakeOCRProvider,
)
from core.vision.vision import (
    VisionProvider,
    FakeVisionProvider,
)
from core.vision.ollama_ocr import OllamaOCRProvider
from core.vision.ollama_vision import OllamaVisionProvider
from core.vision.image_loader import ImageLoader
from core.vision.rag_bridge import ocr_to_document
from core.vision.tools import (
    OCR_IMAGE_TOOL,
    ANALYZE_IMAGE_TOOL,
    register_vision_tools,
)

__all__ = [
    # Image types
    "ImageInput",
    "ImageMetadata",
    "ImageDocument",
    "SUPPORTED_IMAGE_CONTENT_TYPES",
    # OCR types
    "OCRResult",
    "OCRBlock",
    "OCRLine",
    "OCRWord",
    "OCRBoundingBox",
    # Vision types
    "VisionResult",
    "VisionRegion",
    # Routing helpers
    "build_vision_routing_request",
    # Providers
    "OCRProvider",
    "FakeOCRProvider",
    "OllamaOCRProvider",
    "VisionProvider",
    "FakeVisionProvider",
    "OllamaVisionProvider",
    # Image loader
    "ImageLoader",
    # RAG bridge
    "ocr_to_document",
    # Tools
    "OCR_IMAGE_TOOL",
    "ANALYZE_IMAGE_TOOL",
    "register_vision_tools",
]
