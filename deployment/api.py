"""FastAPI application exposing hybrid model inference endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch # type: ignore
import typer # type: ignore
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore

from deployment.inference import HybridInferenceService # type: ignore
from utils.logging_utils import create_logger # type: ignore

app = FastAPI(title="Quantum-AI Hybrid Inference API", version="1.0.0")
logger = create_logger(None)

inference_service: HybridInferenceService | None = None


class PredictRequest(BaseModel):
    """Schema for prediction requests."""
    samples: List[List[float]]


class PredictResponse(BaseModel):
    """Schema for prediction responses."""
    predictions: List[int]
    probabilities: List[List[float]]
    backend: str


@app.on_event("startup")
async def startup_event() -> None:
    """Log startup event."""
    logger.info("Inference API started.")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    backend = inference_service.backend_name if inference_service else "unloaded"
    return {"status": "ok", "backend": backend}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict endpoint returning class predictions and probabilities."""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    samples = torch.tensor(request.samples, dtype=torch.float32)
    predictions, probabilities = inference_service.predict(samples)
    return PredictResponse(
        predictions=predictions.tolist(),
        probabilities=probabilities.tolist(),
        backend=inference_service.backend_name,
    )


def main(
    model_path: Path = typer.Option(..., "--model-path", help="Checkpoint .pt file."),
    device: str = typer.Option("cpu", "--device", help="Device identifier (cpu/cuda/mps)."),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """CLI entry-point launching the FastAPI server with Uvicorn."""
    global inference_service
    inference_service = HybridInferenceService(model_path=model_path, device=device)
    logger.info("Loaded model from %s with backend %s", model_path, inference_service.backend_name)

    import uvicorn # type: ignore

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)