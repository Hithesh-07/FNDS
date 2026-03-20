import json
from app import analyze
res = analyze("The government has announced a new fiscal policy regarding interest rates.")
print("JSON dump test:")
print(json.dumps(res, indent=2))
