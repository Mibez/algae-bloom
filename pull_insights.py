import json
import os
import time
import google.generativeai as genai
from google.api_core import exceptions

# --- CONFIGURATION ---
INPUT_FILENAME = "bloom_insights.json"
OUTPUT_FILENAME = "bloom_insights_with_reports.json"

# API KEY SETUP
# Option 1: Set environment variable 'GEMINI_API_KEY'
# Option 2: Replace the string below with your actual key
API_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_API_KEY_HERE"

# Model Configuration
# Using 'gemini-1.5-flash' as requested for speed/efficiency
MODEL_NAME = "gemini-2.5-flash"

def setup_gemini():
    """Configures the Gemini API client."""
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
        raise ValueError("Please set a valid Google Gemini API Key in the script or environment variables.")
    
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(MODEL_NAME)

def generate_content_with_retry(model, prompt, max_retries=5):
    """
    Generates content with exponential backoff for rate limiting.
    Delays: 1s, 2s, 4s, 8s, 16s
    """
    wait_time = 1
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted:
            # Handle Rate Limit (429) errors specifically
            print(f"   [Rate Limit] Quota exceeded. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            wait_time *= 2
        except Exception as e:
            # Handle other potential network errors
            print(f"   [Error] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(wait_time)
            wait_time *= 2
            
    return None

def process_grids():
    # 1. Check if input file exists
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: {INPUT_FILENAME} not found. Run the analysis script first.")
        return

    # 2. Load Data
    with open(INPUT_FILENAME, 'r') as f:
        grid_data = json.load(f)
    
    print(f"Loaded {len(grid_data)} grids from {INPUT_FILENAME}.")
    
    # 3. Setup Model
    try:
        model = setup_gemini()
    except ValueError as e:
        print(e)
        return

    # 4. Iterate and Generate
    processed_count = 0
    
    print(f"Starting generation with model: {MODEL_NAME}...")
    
    for i, item in enumerate(grid_data):
        grid_id = item.get('grid_id', 'Unknown')
        
        # Skip if report already exists (useful if re-running after partial failure)
        if 'report' in item and item['report']:
            print(f"[{i+1}/{len(grid_data)}] Grid {grid_id}: Report already exists. Skipping.")
            continue
            
        prompt = item.get('prompt')
        if not prompt:
            print(f"[{i+1}/{len(grid_data)}] Grid {grid_id}: No prompt found.")
            continue

        print(f"[{i+1}/{len(grid_data)}] Grid {grid_id}: Generating report...", end="", flush=True)
        
        report_text = generate_content_with_retry(model, prompt)
        
        if report_text:
            item['report'] = report_text
            print(" Done.")
            processed_count += 1
        else:
            item['report'] = "Error: Failed to generate report after retries."
            print(" Failed.")
            
        # Optional: Save periodically to avoid losing progress
        if processed_count % 5 == 0:
             with open(OUTPUT_FILENAME, 'w') as f:
                json.dump(grid_data, f, indent=4)

        # Polite delay between calls (optional, but good practice)
        time.sleep(1) 

    # 5. Final Save
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(grid_data, f, indent=4)
        
    print(f"\nProcessing complete. Updated data saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    process_grids()