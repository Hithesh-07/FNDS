import json
from app import run_analysis
res = run_analysis("The government has announced a new fiscal policy regarding interest rates.")
print("JSON dump test:")
print(json.dumps(res, indent=2))
