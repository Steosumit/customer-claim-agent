import base64
import os
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class ImageLoaderInput(BaseModel):
    image_paths: str = Field(
        description="Semicolon-separated image file paths to load"
    )


class ImageLoaderTool(BaseTool):
    name: str = "image_loader"
    description: str = (
        "Load one or more images from disk and return their paths and base64-encoded "
        "content for visual analysis. Input: semicolon-separated image paths."
    )
    args_schema: Type[BaseModel] = ImageLoaderInput

    def _run(self, image_paths: str) -> str:
        paths = [p.strip() for p in image_paths.split(";") if p.strip()]
        results = []
        for path in paths:
            if not os.path.exists(path):
                results.append(f"MISSING: {path}")
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("utf-8")
                size_kb = len(data) / 1024
                results.append(
                    f"LOADED: {os.path.basename(path)} | "
                    f"size={size_kb:.1f}KB | "
                    f"base64_length={len(b64)} | "
                    f"full_path={path}"
                )
            except Exception as e:
                results.append(f"ERROR: {path} -> {e}")
        return "\n".join(results)
