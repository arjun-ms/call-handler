"""Tests for the API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_health():
    """Health endpoint should return ok."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_infer_valid_audio(sample_wav):
    """Inference endpoint should return structured response for valid audio."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_wav, "rb") as f:
            resp = await client.post(
                "/v1/infer",
                files={"file": ("test.wav", f, "audio/wav")},
            )

    assert resp.status_code == 200
    data = resp.json()

    # Validate response schema
    assert "contact_id" in data
    assert "gender" in data
    assert "age_bracket" in data
    assert "processing_ms" in data
    assert "audio_quality" in data

    # Validate nested prediction structure
    assert "prediction" in data["gender"]
    assert "confidence" in data["gender"]
    assert "prediction" in data["age_bracket"]
    assert "confidence" in data["age_bracket"]

    # Confidence should be 0-1
    assert 0 <= data["gender"]["confidence"] <= 1
    assert 0 <= data["age_bracket"]["confidence"] <= 1


@pytest.mark.asyncio
async def test_infer_invalid_format():
    """Should reject unsupported file formats."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/infer",
            files={"file": ("test.txt", b"not audio data", "text/plain")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_infer_empty_file():
    """Should reject empty files."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/infer",
            files={"file": ("test.wav", b"", "audio/wav")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_request_id_header(sample_wav):
    """Response should include X-Request-ID header."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_wav, "rb") as f:
            resp = await client.post(
                "/v1/infer",
                files={"file": ("test.wav", f, "audio/wav")},
            )

    assert "x-request-id" in resp.headers
