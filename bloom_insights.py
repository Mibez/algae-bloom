import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# --- CONFIGURATION ---
INPUT_JSON_FILENAME = "bloom_report.json"
OUTPUT_JSON_FILENAME = "bloom_insights.json"

BBOX_MIN_LON = 13.5231764283148
BBOX_MIN_LAT = 54.3568091987064
BBOX_MAX_LON = 28.8519583183615
BBOX_MAX_LAT = 63.2446999640676
START_YEAR = 2016
END_YEAR = 2025
SEASON_START_MONTH = 6  # June
SEASON_END_MONTH = 9    # September

# Grid Dimensions (Approximate)
TARGET_GRID_WIDTH_KM = 100
TARGET_GRID_HEIGHT_KM = 60

# Date to simulate "Now" (YYYY-MM-DD)
TARGET_DATE = "2024-06-15"

# 1. SETUP: Mock Data (Fallback if file missing)
mock_data = {
    "events": [
        # Location A (61.30, 21.22) - Inside BBOX
        {
            "timestamp": "2023-06-01T12:00:00.000000", 
            "longtitude": 21.227486, "latitude": 61.30734,
            "radius": 5, "bloom_index": 2, "description": "Minor bloom"
        },
        {
            "timestamp": "2024-06-05T12:00:00.000000", # Next Year
            "longtitude": 21.227486, "latitude": 61.30734,
            "radius": 5, "bloom_index": 4, "description": "High bloom"
        },
        # Location B (61.10, 21.22) - Close to A, might fall in same grid
        {
            "timestamp": "2023-07-15T12:00:00.000000",
            "longtitude": 21.227486, "latitude": 61.107612,
            "radius": 5, "bloom_index": 3, "description": "Moderate bloom"
        },
        # Location C (Outside BBOX for testing)
        {
            "timestamp": "2023-06-01T12:00:00.000000",
            "longtitude": 50.000000, "latitude": 10.000000, # Way outside
            "radius": 5, "bloom_index": 1, "description": "No bloom"
        }
    ]
}

def load_data():
    """Loads data from JSON file or falls back to mock data."""
    if os.path.exists(INPUT_JSON_FILENAME):
        print(f"Loading data from {INPUT_JSON_FILENAME}...")
        try:
            with open(INPUT_JSON_FILENAME, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading file: {e}. Using mock data.")
            return mock_data
    else:
        print(f"File {INPUT_JSON_FILENAME} not found. Using mock data.")
        return mock_data

def calculate_grid_steps(center_lat):
    """
    Approximates the degree steps for grid cells based on km requirements.
    1 deg lat ~= 111.32 km
    1 deg lon ~= 40075km * cos(lat) / 360
    """
    km_per_deg_lat = 111.32
    lat_step = TARGET_GRID_HEIGHT_KM / km_per_deg_lat
    
    # Calculate km per degree longitude at the center latitude
    rad_lat = np.radians(center_lat)
    km_per_deg_lon = (40075 * np.cos(rad_lat)) / 360
    lon_step = TARGET_GRID_WIDTH_KM / km_per_deg_lon
    
    return lat_step, lon_step

def assign_grid(df, lat_step, lon_step):
    """Assigns a grid ID to each row based on BBOX origin."""
    df['grid_x'] = ((df['longtitude'] - BBOX_MIN_LON) / lon_step).astype(int)
    df['grid_y'] = ((df['latitude'] - BBOX_MIN_LAT) / lat_step).astype(int)
    df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)
    return df

def process_bloom_data(json_data):
    if not json_data or 'events' not in json_data:
        return pd.DataFrame(), pd.DataFrame(), (0,0)

    df = pd.DataFrame(json_data['events'])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), (0,0)

    df['dt'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month

    # --- FILTERING ---
    # 1. Spatial Filter (BBOX)
    df = df[
        (df['longtitude'] >= BBOX_MIN_LON) & (df['longtitude'] <= BBOX_MAX_LON) &
        (df['latitude'] >= BBOX_MIN_LAT) & (df['latitude'] <= BBOX_MAX_LAT)
    ]
    
    # 2. Temporal Filter (Years & Season)
    df = df[
        (df['year'] >= START_YEAR) & (df['year'] <= END_YEAR) &
        (df['month'] >= SEASON_START_MONTH) & (df['month'] <= SEASON_END_MONTH)
    ]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), (0,0)

    # --- GRIDDING ---
    # Calculate dynamic steps based on average latitude of the BBOX
    center_lat = (BBOX_MIN_LAT + BBOX_MAX_LAT) / 2
    lat_step, lon_step = calculate_grid_steps(center_lat)
    df = assign_grid(df, lat_step, lon_step)

    # --- DEBUGGING: Print Monthly Average per Year ---
    yearly_monthly_avg = df.groupby(['grid_id', 'year', 'month'])['bloom_index'].mean().reset_index()
    print("\n--- DEBUG: Monthly Averages per Year (for verification) ---")
    print(yearly_monthly_avg.sort_values(['grid_id', 'year', 'month']).to_string(index=False))
    print("-----------------------------------------------------------\n")

    # --- AGGREGATION ---
    # 1. Historical Average per Month per Grid (Aggregated across all years)
    historical_avg = df.groupby(['grid_id', 'month'])['bloom_index'].mean().reset_index()
    historical_avg.rename(columns={'bloom_index': 'avg_hist_index'}, inplace=True)

    return historical_avg, df, (lat_step, lon_step)

def create_prompt_for_grid(grid_id, center_lat, center_lon, historical_df, full_df):
    """
    Generates the prompt string for a specific grid ID.
    """
    # 1. Get Historical Context
    grid_history = historical_df[historical_df['grid_id'] == grid_id].sort_values('month')
    
    # 2. Get "Current" Status (Latest data point in this grid)
    grid_raw = full_df[full_df['grid_id'] == grid_id]
    
    if grid_raw.empty:
         return "No data available."
        

    # Filter grid_raw to strictly include dates <= TARGET_DATE
    target_dt = pd.to_datetime(TARGET_DATE)
    relevant_data = grid_raw[grid_raw['dt'] <= target_dt].reset_index()

    if relevant_data.empty:
         print(f"Relevant data empty from {grid_id}")
         # Fallback: If user requested a date way in the past, just say no data available.
         return f"No data available for this grid on or before {TARGET_DATE}."
    latest_entry = relevant_data.sort_values('dt', ascending=False).iloc[0]
    current_bloom = latest_entry['bloom_index']
    current_date = latest_entry['dt'].strftime('%Y-%m-%d')

    # 3. Format Data for Prompt
    history_str = ""
    for _, row in grid_history.iterrows():
        month_name = datetime(2000, int(row['month']), 1).strftime('%B')
        history_str += f"- {month_name}: Avg Index {row['avg_hist_index']:.2f}\n"

    # 4. Construct the Prompt
    prompt = f"""
You are an expert environmental data analyst generating a formal report for a dashboard interface.

CONTEXT:
Location Grid ID: {grid_id}
Coordinates: {center_lat:.4f} N, {center_lon:.4f} E
Latest Observation Date: {current_date}
Current Bloom Index: {current_bloom}

HISTORICAL DATA (Monthly Averages):
{history_str}

INSTRUCTIONS:
Generate a concise, strictly formatted report for a UI modal. 
- Do NOT use emojis or conversational filler.
- Use the exact headings provided below.
- Keep sections brief and data-driven.

REQUIRED FORMAT:
### Current Status Assessment
[Analyze the current bloom index relative to historical norms for this specific time of year.]

### Short-Term Projection
[Predict algae levels for the remainder of the current month and the next month based on historical trends.]

### Business Advisory: Coastal Retail
[Provide specific operational advice for an ice cream kiosk (e.g., optimal operating hours, marketing adjustments) based on the predicted water quality and associated tourism traffic.]
"""
    return prompt

def generate_all_grid_insights(historical_df, full_df, lat_step, lon_step):
    """
    Iterates over all populated grids and generates the output list.
    """
    output_list = []
    unique_grids = full_df['grid_id'].unique()

    print(f"Generating insights for {len(unique_grids)} unique grids...")

    for grid_id in unique_grids:
        # Parse grid indices from ID "x_y"
        try:
            gx_str, gy_str = grid_id.split('_')
            gx, gy = int(gx_str), int(gy_str)
        except ValueError:
            continue # Skip malformed IDs

        # Calculate Grid Bounding Box
        # Grid Start (Bottom-Left reference for calculation, though logic implies Top-Left for display usually)
        # Based on assign_grid logic: 
        # x derived from (lon - MIN_LON)
        # y derived from (lat - MIN_LAT)
        
        g_min_lon = BBOX_MIN_LON + (gx * lon_step)
        g_max_lon = g_min_lon + lon_step
        g_min_lat = BBOX_MIN_LAT + (gy * lat_step)
        g_max_lat = g_min_lat + lat_step

        # Define Corners
        # Top-Left: Max Latitude, Min Longitude
        # Bottom-Right: Min Latitude, Max Longitude
        top_left = {
            "latitude": float(g_max_lat),
            "longitude": float(g_min_lon)
        }
        bottom_right = {
            "latitude": float(g_min_lat),
            "longitude": float(g_max_lon)
        }

        # Calculate Center for Prompt context
        center_lat = (g_min_lat + g_max_lat) / 2
        center_lon = (g_min_lon + g_max_lon) / 2

        # Generate Prompt
        prompt_text = create_prompt_for_grid(grid_id, center_lat, center_lon, historical_df, full_df)

        # Add to list
        grid_data = {
            "grid_id": grid_id,
            "grid_bounds": {
                "top_left": top_left,
                "bottom_right": bottom_right
            },
            "prompt": prompt_text
        }
        output_list.append(grid_data)
    
    return output_list

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("--- Starting Algae Bloom Analysis ---")
    
    # 1. Load Data
    raw_json = load_data()
    
    # 2. Process Data (Filter, Grid, Aggregate)
    hist_stats, filtered_df, steps = process_bloom_data(raw_json)
    
    if not filtered_df.empty:
        # 3. Generate Insights for ALL grids
        insights = generate_all_grid_insights(hist_stats, filtered_df, steps[0], steps[1])
        
        # 4. Save to JSON
        try:
            with open(OUTPUT_JSON_FILENAME, 'w') as f:
                json.dump(insights, f, indent=4)
            print(f"\nSuccess! Insights for {len(insights)} grids saved to '{OUTPUT_JSON_FILENAME}'.")
        except Exception as e:
            print(f"Error saving output file: {e}")
            
    else:
        print("No data found within the specified BBOX and Date filters.")