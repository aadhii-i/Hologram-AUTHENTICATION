"""Integration tests for the FastAPI service (api/main.py)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_health():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_result_404_for_unknown_video_id():
    resp = client.get("/result/does-not-exist")
    assert resp.status_code == 404


def test_detect_404_without_prior_upload():
    resp = client.post("/detect", json={"video_id": "does-not-exist"})
    assert resp.status_code == 404


def test_upload_rejects_unsupported_extension(tmp_path):
    bogus = tmp_path / "not_a_video.txt"
    bogus.write_text("hello")
    with open(bogus, "rb") as f:
        resp = client.post("/upload", files={"file": ("not_a_video.txt", f, "text/plain")})
    assert resp.status_code == 400


def test_full_upload_detect_result_download_flow(genuine_sequence_video):
    with open(genuine_sequence_video, "rb") as f:
        upload_resp = client.post(
            "/upload", files={"file": ("genuine.mp4", f, "video/mp4")}
        )
    assert upload_resp.status_code == 200
    video_id = upload_resp.json()["video_id"]

    detect_resp = client.post("/detect", json={"video_id": video_id, "roi_mode": "auto"})
    assert detect_resp.status_code == 200
    body = detect_resp.json()
    assert body["label"] in ("GENUINE", "COUNTERFEIT", "UNKNOWN")

    result_resp = client.get(f"/result/{video_id}")
    assert result_resp.status_code == 200
    assert result_resp.json()["video_id"] == video_id

    download_resp = client.get(f"/download/{video_id}/result_json")
    assert download_resp.status_code == 200

    missing_artifact_resp = client.get(f"/download/{video_id}/not_a_real_artifact")
    assert missing_artifact_resp.status_code == 404
