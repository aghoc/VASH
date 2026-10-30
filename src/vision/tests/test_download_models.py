import hashlib
import io
import runpy

import pytest

from vagt_orca.config import PROJECT_ROOT


@pytest.fixture
def downloader():
    return runpy.run_path(str(PROJECT_ROOT / "scripts/download_models.py"))


def test_corrupt_download_is_not_published_as_a_checkpoint(tmp_path, monkeypatch, downloader):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
    output = tmp_path / "yolo26m.pt"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        downloader["restore"]("yolo26m", output)
    assert not output.exists()
    assert not output.with_suffix(".pt.part").exists()


def test_verified_download_uses_the_requested_model(tmp_path, monkeypatch, downloader):
    payload = b"verified checkpoint fixture"
    monkeypatch.setitem(downloader["MODEL_CHECKSUMS"], "yolo26s", hashlib.sha256(payload).hexdigest())
    requests = []

    def open_url(url, **kwargs):
        requests.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    output = tmp_path / "yolo26s.pt"
    downloader["restore"]("yolo26s", output)
    assert output.read_bytes() == payload
    assert requests == [downloader["MODEL_RELEASE"] + "/yolo26s.pt"]


def test_verify_only_never_downloads_or_overwrites_existing_file(tmp_path, monkeypatch, downloader):
    def unexpected_request(*args, **kwargs):
        pytest.fail("verification must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_request)
    output = tmp_path / "existing.pt"
    output.write_bytes(b"keep original")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        downloader["restore"]("yolo26m", output, verify_only=True)
    assert output.read_bytes() == b"keep original"
