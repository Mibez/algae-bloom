import sys
import requests
import os
import io
import math
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- CONFIGURATION ---

# Layers
ALGAE_LAYER = "eo:EO_ALGAE_OLCI_S2_LC"
SST_LAYER = "eo:EO_SST"

# Base URL
WMS_BASE_URL = "https://geoserver2.ymparisto.fi/geoserver/eo/wms"

# WMS parameters
WMS_PARAMS_TEMPLATE = {
    "service": "WMS",
    "version": "1.1.0",
    "request": "GetMap",
    "bbox": "13.5231764283148,54.3568091987064,28.8519583183615,63.2446999640676",
    "width": "768",
    "height": "445",
    "srs": "EPSG:4326",
    "format": "image/png",
    "headers": {'User-Agent': 'Python WMS Script/1.0'}
}

START_YEAR = 2016
END_YEAR = 2024
SEASON_START_MONTH = 6
SEASON_END_MONTH = 9
DATE_STEP_DAYS = 1

RAW_DATA_FOLDER = "raw_data_prediction/"
PROCESSED_DATA_ALGAE = "prediction_data/algae/"
PROCESSED_DATA_SST = "prediction_data/sst/"
MASK_CACHE_FILENAME = "land_mask_predict.png"
REQUEST_TIMEOUT = 30

# Analysis Constants
CLOUD_BRIGHTNESS_THRESHOLD = 230
CLOUD_THRESHOLD_PERCENTAGE = 99
LAND_COLOR = (0, 0, 0)
SEA_FALLBACK_COLOR = (77, 124, 166) # Blue for remaining cloud gaps
SST_WARM_THRESHOLD = 130  # Lowered slightly to ensure we see some growth in temperate waters
WHITE_THRESHOLD = 245 

# --- UTILITIES ---

def get_seasonal_dates(start_year, end_year, start_month, end_month, step_days):
    date_list = []
    for year in range(start_year, end_year + 1):
        current_date = datetime(year, start_month, 1)
        while current_date.month <= end_month and current_date.year == year:
            date_list.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=step_days)
    return date_list

# --- FETCHING ---

def fetch_raw_image(layer_name, date_str):
    safe_layer = layer_name.replace(":", "_")
    layer_folder = os.path.join(RAW_DATA_FOLDER, safe_layer)
    os.makedirs(layer_folder, exist_ok=True)
    filepath = os.path.join(layer_folder, f"{date_str}.png")

    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return Image.open(io.BytesIO(f.read())).convert('RGB')
        except Exception:
            pass 

    params = WMS_PARAMS_TEMPLATE.copy()
    params['layers'] = layer_name
    params['time'] = date_str
    
    try:
        response = requests.get(WMS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception:
        return None

def generate_global_land_mask():
    """Generates a mask where pixels are 255 (Land) if they are ALWAYS white."""
    if os.path.exists(MASK_CACHE_FILENAME):
        print(f"Loading cached land mask from {MASK_CACHE_FILENAME}")
        return Image.open(MASK_CACHE_FILENAME).convert('1')

    print("Generating Global Land Mask...")
    # Use the new seasonal date generator
    weekly_dates = get_seasonal_dates(START_YEAR, END_YEAR, SEASON_START_MONTH, SEASON_END_MONTH, DATE_STEP_DAYS)
    first_img = fetch_raw_image(ALGAE_LAYER, weekly_dates[0])
    if not first_img:
        raise Exception("Could not fetch initial image for dimensions.")

    width, height = first_img.size
    composite_mask = Image.new('L', (width, height), color=255)
    mask_pixels = composite_mask.load()
    
    for date_str in weekly_dates:
        img = fetch_raw_image(ALGAE_LAYER, date_str)
        if not img: continue
        print(f"  [Mask Gen] Processing {date_str}...", end='\r')

        current_pixels = img.load()
        for x in range(width):
            for y in range(height):
                if mask_pixels[x, y] == 0: continue 
                if sum(current_pixels[x, y]) / 3 < WHITE_THRESHOLD:
                    mask_pixels[x, y] = 0 # Mark as Sea
                    
    print("\nLand Mask Generation Complete.")
    final_mask = composite_mask.convert('1')
    final_mask.save(MASK_CACHE_FILENAME)
    return final_mask


# --- PROCESSING LOGIC ---

def get_cloud_mask(img):
    gray = img.convert('L')
    return gray.point(lambda p: 255 if p >= CLOUD_BRIGHTNESS_THRESHOLD else 0, mode='L')

def apply_land_mask(img, land_mask):
    img.paste(LAND_COLOR, mask=land_mask)
    return img

def fill_gaps_current(current_img, history_img, land_mask):
    """
    Fills clouds in 'current_img' using 'history_img' (Temporal).
    Then uses nearest neighbor (Spatial) for remaining gaps.
    """
    width, height = current_img.size
    
    # 1. Identify Clouds
    cloud_mask = get_cloud_mask(current_img)
    
    # 2. Temporal Fill (Use history)
    restored = current_img.copy()
    if history_img:
        restored.paste(history_img, mask=cloud_mask)
    else:
        # If no history, fill with Sea Color as baseline
        fallback = Image.new('RGB', (width, height), SEA_FALLBACK_COLOR)
        restored.paste(fallback, mask=cloud_mask)
        
    # 3. Re-apply Land Mask (to ensure land doesn't get overwritten by sea color)
    restored.paste(LAND_COLOR, mask=land_mask)
    
    return restored

# --- MAIN LOOP ---

def main():
    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_DATA_ALGAE, exist_ok=True)
    os.makedirs(PROCESSED_DATA_SST, exist_ok=True)

    land_mask = generate_global_land_mask()
    dates = get_seasonal_dates(START_YEAR, END_YEAR, SEASON_START_MONTH, SEASON_END_MONTH, DATE_STEP_DAYS)
        
    # STATE: The last known valid look at the world
    state_algae = None
    state_sst = None 
        
    for date_str in dates:
        print(f"Processing {date_str}...", end='\r')
        
        # 1. Fetch Raw Data
        raw_algae = fetch_raw_image(ALGAE_LAYER, date_str)
        raw_sst = fetch_raw_image(SST_LAYER, date_str)
        
        if not raw_algae or not raw_sst:
            continue
            
        # 2. Masking
        raw_algae = apply_land_mask(raw_algae, land_mask)
        raw_sst = apply_land_mask(raw_sst, land_mask)
        
        # 3. Gap Filling (Update State)
        filled_algae = fill_gaps_current(raw_algae, state_algae, land_mask)
        filled_sst = fill_gaps_current(raw_sst, state_sst, land_mask)
        
        # Update State
        state_algae: Image = filled_algae.copy()
        state_sst = filled_sst.copy()
        state_algae.save(f"{PROCESSED_DATA_ALGAE}/{date_str}.png")
        state_sst.save(f"{PROCESSED_DATA_SST}/{date_str}.png")

if __name__ == "__main__":
    main()