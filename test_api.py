import requests
import json

BASE_URL = "http://127.0.0.1:8000"

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS -- {name}")
        passed += 1
    else:
        print(f"  FAIL -- {name} {detail}")
        failed += 1

print("=" * 55)
print("  W3C Reconciliation Service API Test Framework")
print("  CM3070 Final Year Project")
print("=" * 55)

# -------------------------------------------------------
print("\n[1] Service Manifest (GET /reconcile)")
# -------------------------------------------------------
r = requests.get(f"{BASE_URL}/reconcile")
test("Returns HTTP 200", r.status_code == 200)

manifest = r.json()
test("Has 'name' field", "name" in manifest)
test("Has 'versions' field", "versions" in manifest)
test("Has 'identifierSpace'", "identifierSpace" in manifest)
test("Has 'schemaSpace'", "schemaSpace" in manifest)
test("Has 'defaultTypes'", "defaultTypes" in manifest)
test("Version includes 0.2", "0.2" in manifest.get("versions", []))
test("defaultTypes is a list", isinstance(manifest.get("defaultTypes"), list))
test("defaultTypes has id and name", 
    all("id" in t and "name" in t 
        for t in manifest.get("defaultTypes", [])))

# -------------------------------------------------------
print("\n[2] Reconciliation Query (POST /reconcile)")
# -------------------------------------------------------
query = {"q1": {"query": "Singapore", "limit": 3}}
r = requests.post(
    f"{BASE_URL}/reconcile",
    data={"queries": json.dumps(query)}
)
test("Returns HTTP 200", r.status_code == 200)

data = r.json()
test("Response has query key 'q1'", "q1" in data)
test("q1 has 'result' field", "result" in data.get("q1", {}))

results = data.get("q1", {}).get("result", [])
test("Returns at least 1 result for 'Singapore'", len(results) > 0)

if results:
    first = results[0]
    test("Result has 'id' field", "id" in first)
    test("Result has 'name' field", "name" in first)
    test("Result has 'score' field", "score" in first)
    test("Result has 'match' field", "match" in first)
    test("Result has 'type' field", "type" in first)
    test("Score is a number", isinstance(first.get("score"), (int, float)))
    test("Match is a boolean", isinstance(first.get("match"), bool))
    test("Type is a list", isinstance(first.get("type"), list))
    test("Singapore scores 100", first.get("score") >= 100)

# -------------------------------------------------------
print("\n[3] Fuzzy Matching Test (POST /reconcile)")
# -------------------------------------------------------
query2 = {"q1": {"query": "Singapoer", "limit": 3}}
r2 = requests.post(
    f"{BASE_URL}/reconcile",
    data={"queries": json.dumps(query2)}
)
data2 = r2.json()
results2 = data2.get("q1", {}).get("result", [])
test("Returns candidate for misspelled 'Singapoer'", len(results2) > 0)
if results2:
    test("Fuzzy match score is less than 100", 
         results2[0].get("score", 100) < 100)

# -------------------------------------------------------
print("\n[4] Alternate Names Test (POST /reconcile)")
# -------------------------------------------------------
query3 = {"q1": {"query": "Saigon", "limit": 3}}
r3 = requests.post(
    f"{BASE_URL}/reconcile",
    data={"queries": json.dumps(query3)}
)
data3 = r3.json()
results3 = data3.get("q1", {}).get("result", [])
test("Finds 'Ho Chi Minh City' when querying 'Saigon'", 
     any("Ho Chi Minh" in r.get("name", "") for r in results3))

# -------------------------------------------------------
print("\n[5] Suggest Type Endpoint (GET /reconcile/suggest/type)")
# -------------------------------------------------------
r = requests.get(f"{BASE_URL}/reconcile/suggest/type?prefix=pop")
test("Returns HTTP 200", r.status_code == 200)
suggest = r.json()
test("Has 'result' field", "result" in suggest)
test("Returns at least 1 type", len(suggest.get("result", [])) > 0)
if suggest.get("result"):
    t = suggest["result"][0]
    test("Type has 'id' field", "id" in t)
    test("Type has 'name' field", "name" in t)

# -------------------------------------------------------
print("\n[6] Suggest Property Endpoint")
# -------------------------------------------------------
r = requests.get(f"{BASE_URL}/reconcile/suggest/property?prefix=pop")
test("Returns HTTP 200", r.status_code == 200)
props = r.json()
test("Has 'result' field", "result" in props)
test("Returns population property", 
     any("population" in p.get("id","") 
         for p in props.get("result", [])))

# -------------------------------------------------------
print("\n[7] Empty Query Test")
# -------------------------------------------------------
query4 = {"q1": {"query": "", "limit": 3}}
r4 = requests.post(
    f"{BASE_URL}/reconcile",
    data={"queries": json.dumps(query4)}
)
test("Returns HTTP 200 for empty query", r4.status_code == 200)

# -------------------------------------------------------
print("\n[8] Multiple Queries Test")
# -------------------------------------------------------
multi = {
    "q1": {"query": "Bangkok", "limit": 3},
    "q2": {"query": "Kuala Lumpur", "limit": 3},
    "q3": {"query": "Jakarta", "limit": 3}
}
r5 = requests.post(
    f"{BASE_URL}/reconcile",
    data={"queries": json.dumps(multi)}
)
test("Returns HTTP 200 for multiple queries", r5.status_code == 200)
data5 = r5.json()
test("Returns all 3 query keys", 
     all(k in data5 for k in ["q1", "q2", "q3"]))
test("Bangkok matched", 
     len(data5.get("q1", {}).get("result", [])) > 0)
test("Kuala Lumpur matched", 
     len(data5.get("q2", {}).get("result", [])) > 0)
test("Jakarta matched", 
     len(data5.get("q3", {}).get("result", [])) > 0)

# -------------------------------------------------------
print("\n" + "=" * 55)
print(f"  RESULTS: {passed} passed / {passed + failed} total")
print(f"  SCORE  : {round(passed/(passed+failed)*100)}%")
print("=" * 55)