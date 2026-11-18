"""Optional local ASGI adapter. Deploy behind authentication and TLS if exposed."""

from typing import Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .data import sha256
from .inference import Policy


class PredictRequest(BaseModel):
    fen: Annotated[str, Field(min_length=15, max_length=128)]
    top_k: Annotated[int, Field(ge=0, le=256)] = 5
    temperature: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)] = 0
    seed: Annotated[int | None, Field(ge=0)] = None


def create_app(checkpoint, device="cpu", threads=4):
    policy = Policy(checkpoint, device, threads)
    model_id = sha256(checkpoint)
    app = FastAPI(title="Geometric Chess Policy", version="0.5.0")

    @app.get("/health")
    def health():
        return {"status": "ready", "model_sha256": model_id, "search": False}

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            return policy.predict(
                request.fen, top_k=request.top_k, temperature=request.temperature, seed=request.seed
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
