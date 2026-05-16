import requests
import time
import argparse
import concurrent.futures
from statistics import mean, median

def send_request(url, filepath):
    start_time = time.perf_counter()
    try:
        with open(filepath, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
            
        elapsed = time.perf_counter() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True, 
                "latency_sec": elapsed, 
                "processing_ms": data.get("processing_ms"),
                "status_code": 200
            }
        else:
            return {"success": False, "latency_sec": elapsed, "status_code": response.status_code, "error": response.text}
            
    except Exception as e:
        return {"success": False, "latency_sec": time.perf_counter() - start_time, "error": str(e), "status_code": 0}

def main():
    parser = argparse.ArgumentParser(description="Performance Load Tester for Audio Inference API")
    parser.add_argument("--url", default="http://localhost:8000/analyze", help="API Endpoint")
    parser.add_argument("--file", default="sample_audio/performance_test.wav", help="Audio file to test")
    parser.add_argument("--requests", type=int, default=10, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Load Test...")
    print(f"Endpoint: {args.url}")
    print(f"File: {args.file}")
    print(f"Total Requests: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print("-" * 40)
    
    results = []
    start_wall = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_request, args.url, args.file) for _ in range(args.requests)]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            results.append(res)
            status = "✅" if res['success'] else "❌"
            print(f"[{i+1}/{args.requests}] {status} Status: {res.get('status_code')} | Latency: {res['latency_sec']:.2f}s | Server-side Processing: {res.get('processing_ms', 0)}ms")
            
    total_wall = time.perf_counter() - start_wall
    
    # Calculate stats
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("=" * 40)
    print(f"📊 PERFORMANCE REPORT")
    print("=" * 40)
    print(f"Total Wall Time: {total_wall:.2f} seconds")
    print(f"Success Rate:    {len(successful)}/{args.requests} ({(len(successful)/args.requests)*100:.1f}%)")
    
    if successful:
        latencies = [r['latency_sec'] for r in successful]
        processing_times = [r['processing_ms'] for r in successful if r['processing_ms']]
        
        print(f"\nLatency (Network + Processing):")
        print(f"  Min:    {min(latencies):.2f}s")
        print(f"  Mean:   {mean(latencies):.2f}s")
        print(f"  Median: {median(latencies):.2f}s")
        print(f"  Max:    {max(latencies):.2f}s")
        
        if processing_times:
            print(f"\nServer-Side Processing Time (from headers):")
            print(f"  Min:    {min(processing_times)}ms")
            print(f"  Mean:   {mean(processing_times):.0f}ms")
            print(f"  Max:    {max(processing_times)}ms")
            
    if failed:
        print(f"\nErrors encountered:")
        for f in failed[:5]: # Show first 5
            print(f"  - Status {f.get('status_code')}: {f.get('error')}")

if __name__ == "__main__":
    main()
