"""Phase 6: warm-latency measurement of the deployed multimodal + single-modal paths.

NOTE: reported SEPARATELY from the final-experiment pipeline latency (133.38 ± 17.67 ms
per final report); these are different measurement environments (this runs the full
deployed HTTP/API stack including multipart handling, parsing, preprocessing, encoder
forward, fusion head, response builder on this CPU machine).
"""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import urllib.request

IMG = r"dataset_analysis\CrisisMMD_v2.0\data_image\california_wildfires\10_10_2017\917792930315821057_0.jpg"
TEXT = "Mass Evacuations in California as Wildfires Kill at Least 10 https://t.co/gyoKFWZuMB #CaliforniaWildfires"

boundary = "----LatB"
data = open(IMG, "rb").read()


def build(include_text=True):
    parts = []
    if True:
        parts.append(
            b'--' + boundary.encode() +
            b'\r\nContent-Disposition: form-data; name="image"; filename="a.jpg"\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
    if include_text:
        parts.append(
            b'--' + boundary.encode() +
            b'\r\nContent-Disposition: form-data; name="tweet_text"\r\n\r\n' +
            TEXT.encode() + b'\r\n')
    parts.append(b'--' + boundary.encode() + b'--\r\n')
    return b"".join(parts)


def call(body):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120).read().decode()
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, resp


def measure(label, include_text, n=12, warmup=3):
    times = []
    for i in range(warmup + n):
        body = build(include_text)
        dt, resp = call(body)
        if i >= warmup:
            times.append(dt)
    out = {
        "mode": label,
        "n": len(times),
        "mean_ms": round(statistics.mean(times), 2),
        "std_ms": round(statistics.stdev(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "p50_ms": round(sorted(times)[len(times) // 2], 2),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95) - 1], 2),
        "times_ms": [round(t, 2) for t in times],
    }
    print(label, out)
    return out


results = {
    "MULTIMODAL": measure("MULTIMODAL", include_text=True),
    "IMAGE_ONLY": measure("IMAGE_ONLY", include_text=False),
    "TEXT_ONLY": measure("TEXT_ONLY", include_text=False),
    "note": "Deployed full HTTP stack latency (CPU). Reported separately from final-exp 133.38ms.",
}

out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "research", "final_cleanup", "deployed_latency.json")
json.dump(results, open(out, "w", encoding="utf-8"), indent=2)
print("saved:", out)