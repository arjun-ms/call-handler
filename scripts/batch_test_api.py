import os
import requests
import time
import json
from datetime import datetime

def run_batch_test(directory: str, output_dir: str):
    api_url = "http://localhost:8000/v1/infer"
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_run_{timestamp}.json")
    
    results = {
        "timestamp": timestamp,
        "directory": directory,
        "files_tested": 0,
        "successful": 0,
        "failed": 0,
        "total_api_latency_ms": 0,
        "details": []
    }
    
    print(f"Starting batch test for directory: {directory}")
    print(f"Results will be saved to: {output_file}\n")
    
    audio_files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
    
    if not audio_files:
        print("No .wav files found in directory.")
        return

    for filename in audio_files:
        filepath = os.path.join(directory, filename)
        print(f"Testing {filename}...", end=" ", flush=True)
        
        file_result = {
            "filename": filename,
            "status": "pending",
            "api_latency_ms": 0,
            "response": {}
        }
        
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'audio/wav')}
                
                start_time = time.time()
                response = requests.post(api_url, files=files)
                end_time = time.time()
                
            api_latency_ms = round((end_time - start_time) * 1000)
            file_result["api_latency_ms"] = api_latency_ms
            results["total_api_latency_ms"] += api_latency_ms
            
            if response.status_code == 200:
                print(f"SUCCESS ({api_latency_ms}ms)")
                file_result["status"] = "success"
                file_result["response"] = response.json()
                results["successful"] += 1
            else:
                print(f"FAILED (Status: {response.status_code})")
                file_result["status"] = "failed"
                file_result["response"] = {"error": response.text, "status_code": response.status_code}
                results["failed"] += 1
                
        except requests.exceptions.ConnectionError:
            print("FAILED (Connection Error)")
            file_result["status"] = "failed"
            file_result["response"] = {"error": "Could not connect to API"}
            results["failed"] += 1
        except Exception as e:
            print(f"FAILED (Error: {str(e)})")
            file_result["status"] = "failed"
            file_result["response"] = {"error": str(e)}
            results["failed"] += 1
            
        results["details"].append(file_result)
        results["files_tested"] += 1
        
    print(f"\nBatch test complete: {results['successful']} successful, {results['failed']} failed.")
    
    # Save the aggregated results to the JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Detailed benchmark saved to: {output_file}")

if __name__ == "__main__":
    test_dir = os.path.join("test-audio-files", "CREMA-D")
    out_dir = "test-results"
    run_batch_test(test_dir, out_dir)
