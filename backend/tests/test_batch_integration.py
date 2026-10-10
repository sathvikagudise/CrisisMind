"""Integration tests for the batch processing API.

Run with::

    python -m pytest backend/tests/test_batch_integration.py -v

Or directly::

    python backend/tests/test_batch_integration.py
"""

import http.client
import json
import io
import uuid
import time

from PIL import Image

PORT = 9876
HOST = "127.0.0.1"


def _req(method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request(method, path, body, headers or {})
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    return resp.status, json.loads(data) if data else {}


def _make_img():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color="red").save(buf, format="PNG")
    return buf.getvalue()


def _upload_body(files, csv_content=None):
    b = uuid.uuid4().hex
    body = b""
    for name, data in files:
        body += f"--{b}\r\n".encode()
        body += f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'.encode()
        body += b"Content-Type: image/png\r\n\r\n" + data + b"\r\n"
    if csv_content:
        body += f"--{b}\r\n".encode()
        body += b'Content-Disposition: form-data; name="csv_file"; filename="report.csv"\r\n'
        body += b"Content-Type: text/csv\r\n\r\n" + csv_content + b"\r\n"
    body += f"--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def test_1_health():
    status, data = _req("GET", "/health")
    assert status == 200
    assert data["status"] == "healthy"
    print("  PASS: health")


def test_2_upload_and_process():
    img1, img2 = _make_img(), _make_img()
    body, ct = _upload_body([("test1.png", img1), ("test2.png", img2)])
    status, data = _req("POST", "/batch/upload", body, {"Content-Type": ct})
    assert status == 200, f"Upload failed: {data}"
    assert data["total_files"] == 2
    assert not data["has_csv"]
    bid = data["batch_id"]
    print(f"  PASS: upload — bid={bid}")

    status, data = _req("POST", f"/batch/start/{bid}", "{}", {"Content-Type": "application/json"})
    assert status == 200
    assert data["status"] == "processing"
    print("  PASS: start")

    for _ in range(20):
        status, data = _req("GET", f"/batch/status/{bid}")
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(2)
    assert data["status"] == "completed", f"Batch failed: {data}"
    assert data["completed"] == 2
    print(f"  PASS: poll — done ({data['progress_pct']}%)")

    status, data = _req("GET", f"/batch/results/{bid}")
    assert status == 200
    assert len(data["results"]) == 2
    assert data["summary"]["total_files"] == 2
    assert data["summary"]["successful"] == 2
    for r in data["results"]:
        assert r["success"]
        assert r["severity"] is not None
        assert r["confidence"] is not None
    print("  PASS: results")

    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("GET", f"/batch/download/csv/{bid}")
    resp = conn.getresponse()
    csv_text = resp.read().decode()
    conn.close()
    assert resp.status == 200
    assert "test1.png" in csv_text
    assert csv_text.count("\n") == 3  # header + 2 rows
    print("  PASS: csv download")

    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("GET", f"/batch/download/pdf/{bid}")
    resp = conn.getresponse()
    report = resp.read().decode()
    conn.close()
    assert resp.status == 200
    assert "CRISIS MIND" in report
    print("  PASS: report download")


def test_3_upload_with_csv():
    img1, img2 = _make_img(), _make_img()
    csv = b"filename,text\ntest1.png,Flood damage reported\ntest2.png,Earthquake felt\n"
    body, ct = _upload_body([("test1.png", img1), ("test2.png", img2)], csv)
    status, data = _req("POST", "/batch/upload", body, {"Content-Type": ct})
    assert status == 200
    assert data["total_files"] == 2
    assert data["has_csv"]
    bid = data["batch_id"]
    print(f"  PASS: upload with CSV — bid={bid}")

    status, data = _req("POST", f"/batch/start/{bid}", "{}", {"Content-Type": "application/json"})
    assert status == 200

    for _ in range(20):
        status, data = _req("GET", f"/batch/status/{bid}")
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(2)
    assert data["status"] == "completed"

    status, data = _req("GET", f"/batch/results/{bid}")
    assert status == 200
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert r["success"]
    print("  PASS: CSV batch results")


def test_4_error_handling():
    status, data = _req("GET", "/batch/status/nonexistent")
    assert status == 404
    print("  PASS: 404 for nonexistent batch")

    status, data = _req("POST", "/batch/start/nonexistent", "{}", {"Content-Type": "application/json"})
    assert status == 404
    print("  PASS: 404 for start nonexistent")

    status, data = _req("GET", "/batch/results/nonexistent")
    assert status == 404
    print("  PASS: 404 for results nonexistent")


if __name__ == "__main__":
    for fn in [test_1_health, test_2_upload_and_process, test_3_upload_with_csv, test_4_error_handling]:
        try:
            fn()
        except Exception as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            exit(1)
    print("\nALL INTEGRATION TESTS PASSED")
