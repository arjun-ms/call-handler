import os
import requests
import time
import json

def test_api():
    """Sends a real audio file to the FastAPI inference endpoint."""
    test_file = r"test-audio-files\CREMA-D\1001_male_51.wav"
    api_url = "http://localhost:8000/v1/infer"
    
    if not os.path.exists(test_file):
        print(f"Error: Could not find {test_file}")
        return

    print(f"Using test file: {test_file}")
    print(f"Sending POST request to {api_url}...")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (os.path.basename(test_file), f, 'audio/wav')}
            
            start_time = time.time()
            response = requests.post(api_url, files=files)
            end_time = time.time()
            
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        print("\nJSON Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to {api_url}.")
        print("Please make sure your FastAPI server is running! (Run: uvicorn app.main:app --reload)")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    test_api()
