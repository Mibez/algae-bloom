import os
import io
import copy
import json
import math
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---

# Coordinate Mapping Constants
BBOX_MIN_LON = 13.5231764283148
BBOX_MIN_LAT = 54.3568091987064
BBOX_MAX_LON = 28.8519583183615
BBOX_MAX_LAT = 63.2446999640676

# Date range and frequency
START_YEAR = 2016
END_YEAR = 2025
SEASON_START_MONTH = 6
SEASON_END_MONTH = 9
DATE_STEP_DAYS = 1

OUTPUT_JSON_FILENAME = "bloom_report.json"
OUTPUT_PREDICTION_JSON_FILENAME = "bloom_report_prediction.json"
RAW_DATA_FOLDER = "prediction_data/algae/"
PREDICTION_DATA_FOLDER = "prediction_data/prediction/"


# COLORS & MASKS
LAND_COLOR = (50, 50, 50) # Dark grey for land


# --- DATE UTILITIES ---

def is_valid_date(dt):
    """
    Checks if a date is within the configured years AND within the active season (months).
    """
    year_valid = START_YEAR <= dt.year <= END_YEAR
    month_valid = SEASON_START_MONTH <= dt.month <= SEASON_END_MONTH
    return year_valid and month_valid

def get_seasonal_dates(start_year, end_year, start_month, end_month, step_days):
    """
    Generates a list of YYYY-MM-DD strings.
    Iterates through each year, starting at the 1st of the start_month,
    and stepping by step_days until the end_month is passed.
    """
    date_list = []
    
    for year in range(start_year, end_year + 1):
        # Reset current_date to the start of the season for this year
        current_date = datetime(year, start_month, 1)
        
        # Loop while we are still within the season of the same year
        while current_date.month <= end_month and current_date.year == year:
            date_list.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=step_days)
            
    return date_list

# --- MASK GENERATION & CACHING ---

def fetch_raw_image(date_str):
    """Fetches, caches, and opens a single image."""
    filepath = os.path.join(RAW_DATA_FOLDER, f"{date_str}.png")

    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return Image.open(io.BytesIO(f.read())).convert('RGB')
        except Exception as e:
            print(f"  [Cache] Failed to load {date_str}: {e}. Retrying.")
            os.remove(filepath)

# --- BLOOM ANALYSIS & JSON REPORTING ---

def get_pixel_lat_lon(x, y, width, height):
    """
    Converts pixel coordinates to Lat/Lon (Equirectangular Projection).
    """
    # X percent (0 to 1) -> Lon
    lon_pct = x / width
    lon = BBOX_MIN_LON + (lon_pct * (BBOX_MAX_LON - BBOX_MIN_LON))
    
    # Y percent (0 to 1) -> Lat
    # Note: Y=0 is Top (Max Lat), Y=Height is Bottom (Min Lat)
    lat_pct = y / height
    lat = BBOX_MAX_LAT - (lat_pct * (BBOX_MAX_LAT - BBOX_MIN_LAT))
    
    return lon, lat

def classify_bloom_index(r, g, b):
    """
    Determines bloom index (0-4) based on RGB color.
    4: Red (Severe)
    3: Yellow (Moderate)
    2: Green (Minor)
    1: Blue (No Bloom)
    0: Land/Invalid
    """
    # 0. Check for Land (Dark Grey)
    if r < 60 and g < 60 and b < 60:
        return 0, "Land"
        
    # 4. Red (Severe): Red dominant
    if r > 150 and g < 100:
        return 4, "Severe algal bloom detected, caution advised."
    
    # 3. Yellow (Moderate): Red and Green high
    if r > 140 and g > 140 and b < 100:
        return 3, "Moderate algal bloom detected, monitor conditions."

    # 2. Green (Minor): Green dominant
    if g > r and g > b:
        return 2, "Minor algal bloom detected."
        
    # 1. Blue (No Bloom): Blue dominant (including our fallback color)
    # Fallback (77, 124, 166) fits here (B=166 is dominant)
    if b > r or b > g:
        return 1, "No bloom detected."
        
    # Default fallback
    return 1, "No bloom detected."

def generate_bloom_report(frames_data):
    """
    Analyzes all frames and generates a JSON report of bloom events.
    Sampling area: 5km x 5km approx.
    """
    print(f"\nGenerating Bloom Analysis JSON Report...")
    
    events = []
    
    # Dimensions
    img_width = frames_data[0][1].width
    img_height = frames_data[0][1].height
    
    # Calculate Grid Steps for ~5km
    # 1 deg lat ~ 111km -> 5km ~ 0.045 deg
    # 1 deg lon ~ 55km (at 60N) -> 5km ~ 0.09 deg
    
    total_lon_deg = BBOX_MAX_LON - BBOX_MIN_LON
    total_lat_deg = BBOX_MAX_LAT - BBOX_MIN_LAT
    
    px_per_deg_lon = img_width / total_lon_deg
    px_per_deg_lat = img_height / total_lat_deg
    
    # Step size in pixels (approx 5km)
    step_x = int(0.09 * px_per_deg_lon) # approx 4-5 pixels
    step_y = int(0.045 * px_per_deg_lat) # approx 2-3 pixels
    
    # Ensure at least 1 pixel step
    step_x = max(1, step_x)
    step_y = max(1, step_y)
    
    for date_str, img in frames_data:
        pixels = img.load()
        
        # Iterate grid
        for y in range(0, img_height, step_y):
            for x in range(0, img_width, step_x):
                
                # Sample average color of the block
                r_sum, g_sum, b_sum, count = 0, 0, 0, 0
                
                # Check pixels within the block
                for by in range(y, min(y + step_y, img_height)):
                    for bx in range(x, min(x + step_x, img_width)):
                        r, g, b = pixels[bx, by]
                        # Ignore pure land color exactly
                        if (r, g, b) != LAND_COLOR:
                            r_sum += r
                            g_sum += g
                            b_sum += b
                            count += 1
                
                if count > 0:
                    avg_r = int(r_sum / count)
                    avg_g = int(g_sum / count)
                    avg_b = int(b_sum / count)
                    
                    idx, desc = classify_bloom_index(avg_r, avg_g, avg_b)
                    
                    # Only log relevant blooms (Index >= 3) to keep JSON size manageable?
                    # Or log all non-land?
                    # Request implies logging the state. Let's log Index > 1 (Minor+) 
                    # to keep file size sane, or all valid sea points.
                    # Let's log if idx >= 2 (Green/Yellow/Red) to filter "No Bloom" noise.
                    # EDIT: User prompt shows "bloom_index": 3, so let's include >= 2.
                    
                    if idx >= 2:
                        # Center of the block for coordinates
                        center_x = x + (step_x / 2)
                        center_y = y + (step_y / 2)
                        lon, lat = get_pixel_lat_lon(center_x, center_y, img_width, img_height)
                        
                        event = {
                            "timestamp": f"{date_str}T12:00:00.000000", # Appending arbitrary time
                            "longtitude": round(lon, 6),
                            "latitude": round(lat, 6),
                            "radius": 5, # 5km block
                            "bloom_index": idx,
                            "description": desc
                        }
                        events.append(event)

    report = {"events": events}
    
    with open(OUTPUT_JSON_FILENAME, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ JSON Report saved as {OUTPUT_JSON_FILENAME} ({len(events)} events)")

# --- CORE LOGIC ---

def fetch_single_image(date_str, processed_dates):
    """Fetches raw image and applies Land Mask."""
    if date_str in processed_dates:
        return None, 0.0, "DUPLICATE"
    
    processed_dates.add(date_str)
    img = fetch_raw_image(date_str)
    
    if img is None:
        return None, "FETCH_FAILED"
    return img, "OK"
    


def fetch_and_process_images():
    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
    
    weekly_dates = get_seasonal_dates(START_YEAR, END_YEAR, SEASON_START_MONTH, SEASON_END_MONTH, DATE_STEP_DAYS)
    collected_frames = []
    processed_dates = set()
    daily_scan_queue = set()
    
    print(f"\n--- Weekly Scan ({START_YEAR}-{SEASON_START_MONTH} to {END_YEAR}-{SEASON_END_MONTH}) ---")
    
    for date_str in weekly_dates:
        img, msg = fetch_single_image(date_str, processed_dates)
        if msg == "OK":
            collected_frames.append((date_str, img))
            print(f"Date {date_str}: OK.")
            center_date = datetime.strptime(date_str, "%Y-%m-%d")
            for i in range(-7, 8):
                daily_date = center_date + timedelta(days=i)
                daily_date_str = daily_date.strftime("%Y-%m-%d")
                if is_valid_date(daily_date):
                    daily_scan_queue.add(daily_date_str)

    # dates_to_fetch_daily = sorted(list(daily_scan_queue - processed_dates))
    # print(f"\n--- Daily Scan ({len(dates_to_fetch_daily)} dates) ---")
    
    # for i, date_str in enumerate(dates_to_fetch_daily):
    #     img, coverage, msg = fetch_single_image(date_str, processed_dates, global_land_mask)
    #     if msg == "OK":
    #         collected_frames.append((date_str, img))
    #         print(f"{date_str}: OK ({coverage:.1f}%)")
    #     else:
    #         print(f"{date_str}: {msg} ({coverage:.1f}%)", end='\r')

    collected_frames.sort(key=lambda x: x[0])
    
    if len(collected_frames) > 1:
        # 1. Interpolate / Inpaint
        # final_frames_data = apply_bidirectional_inpainting(collected_frames)
        
        # 2. Generate JSON Report (Using the restored frames BEFORE adding timestamps to avoid OCR-ing text)
        generate_bloom_report(collected_frames)
        
    #     # 3. Add Timestamps for Visual GIF
    #     print("\nAdding Timestamps to GIF frames...")
    #     final_images_for_gif = []
    #     for date_str, img in final_frames_data:
    #         # Create a copy so we don't stamp the image used for JSON analysis (though analysis is done)
    #         stamped_img = add_timestamp(img.copy(), date_str)
    #         final_images_for_gif.append(stamped_img)
        
    #     # 4. Save GIF
    #     print(f"\nGenerating GIF with {len(final_images_for_gif)} frames...")
    #     final_images_for_gif[0].save(
    #         OUTPUT_GIF_FILENAME,
    #         save_all=True,
    #         append_images=final_images_for_gif[1:],
    #         optimize=True,
    #         duration=GIF_FRAME_DURATION_MS,
    #         loop=0
    #     )
    #     print(f"✅ Success! Saved as {OUTPUT_GIF_FILENAME}")
    # else:
    #     print("\n❌ Not enough frames found.")

if __name__ == "__main__":
    fetch_and_process_images()