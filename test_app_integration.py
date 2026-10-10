import json
from app import app

client = app.test_client()

print("\n==========================================")
print("  TruthLens End-to-End Integration Suite")
print("==========================================\n")

# 1. Test Homepage
res = client.get("/")
assert res.status_code == 200, f"Expected 200 on /, got {res.status_code}"
assert b"TruthLens" in res.data, "TruthLens not in home response"
print("  [PASS] GET / (Home)")

# 2. Test Analyze Page
res = client.get("/analyze")
assert res.status_code == 200, f"Expected 200 on /analyze, got {res.status_code}"
assert b"newsInput" in res.data, "newsInput textarea missing"
print("  [PASS] GET /analyze (Workbench)")

# 3. Test How It Works
res = client.get("/how-it-works")
assert res.status_code == 200, f"Expected 200 on /how-it-works, got {res.status_code}"
assert b"Architecture &amp; Methodology" in res.data or b"Architecture & Methodology" in res.data, "Architecture title missing"
print("  [PASS] GET /how-it-works")

# 4. Test Metrics Page & JSON
res = client.get("/metrics")
assert res.status_code == 200, f"Expected 200 on /metrics, got {res.status_code}"
print("  [PASS] GET /metrics")

res = client.get("/metrics/json")
assert res.status_code == 200, f"Expected 200 on /metrics/json, got {res.status_code}"
data = json.loads(res.data)
assert isinstance(data, list) and len(data) > 0, "Invalid metrics json"
print("  [PASS] GET /metrics/json (API)")

# 5. Test About Page
res = client.get("/about")
assert res.status_code == 200, f"Expected 200 on /about, got {res.status_code}"
print("  [PASS] GET /about")

# 6. Test Health Endpoint
res = client.get("/health")
assert res.status_code == 200, f"Expected 200 on /health, got {res.status_code}"
health_data = json.loads(res.data)
assert health_data.get("status") == "ok", "Health status not ok"
print("  [PASS] GET /health")

# 7. Test API /predict
pred_payload = {"text": "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent following a scheduled monetary policy committee review."}
res = client.post("/predict", json=pred_payload)
assert res.status_code == 200, f"Expected 200 on /predict, got {res.status_code}"
pred_data = json.loads(res.data)
assert "label" in pred_data and "confidence" in pred_data, "Missing fields in prediction"
print(f"  [PASS] POST /predict -> Label: {pred_data['label']}, Confidence: {pred_data['confidence']}%")

# 8. Test Form POST /analyze with CSRF disabled for test client
app.config["WTF_CSRF_ENABLED"] = False
form_res = client.post("/analyze", data={"news_text": "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent following a scheduled monetary policy committee review."})
assert form_res.status_code == 200, f"Expected 200 on POST /analyze, got {form_res.status_code}"
assert b"Intelligence Report" in form_res.data, "Intelligence Report missing in result"
assert b"REAL" in form_res.data or b"FAKE" in form_res.data or b"UNCERTAIN" in form_res.data
print("  [PASS] POST /analyze (Report View Rendering)")

# 9. Test 404 handler
res_404 = client.get("/nonexistent-page-test-1234")
assert res_404.status_code == 404, f"Expected 404 on nonexistent, got {res_404.status_code}"
assert b"404" in res_404.data, "404 page content missing"
print("  [PASS] GET /nonexistent (404 Page)")

print("\n==========================================")
print("  ALL INTEGRATION TESTS PASSED (9/9)!")
print("==========================================\n")
