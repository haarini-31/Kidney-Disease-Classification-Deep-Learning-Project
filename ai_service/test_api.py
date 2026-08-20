import os
import requests
from pathlib import Path

# Local FastAPI endpoint URL
url = "http://127.0.0.1:8000/predict"

# Find a test image in the workspace dataset
project_root = Path(__file__).resolve().parent.parent
dataset_dir = project_root / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

test_file = None
if dataset_dir.exists():
    # Look for first available jpg image in the subfolders
    for p in dataset_dir.rglob("*.jpg"):
        test_file = p
        break

if not test_file or not test_file.exists():
    print("Dataset images not found in default layout. Searching artifacts...")
    # Fallback search path
    artifacts_dir = project_root / "artifacts"
    for p in artifacts_dir.rglob("*.jpg"):
        test_file = p
        break

if not test_file:
    print("Error: No test CT scan image (.jpg) could be found in the workspace.")
    exit(1)

print(f"Testing API with image: {test_file.name}")
print(f"Path: {test_file}")

try:
    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'image/jpeg')}
        response = requests.post(url, files=files)
        
    print("\n--- Test Results ---")
    print(f"HTTP Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(response.json())
        
        data = response.json()
        assert "prediction" in data, "Missing 'prediction' key in response"
        assert "confidence" in data, "Missing 'confidence' key in response"
        assert data["prediction"] in {"Cyst", "Normal", "Stone", "Tumor"}, f"Invalid prediction: {data['prediction']}"
        print("\nSUCCESS: E2E API Verification Completed successfully!")
    else:
        print(f"Error: {response.text}")
except Exception as err:
    print(f"Connection/execution error: {err}")
