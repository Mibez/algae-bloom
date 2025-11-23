#### Algae bloom data source

Install python packages:
```pip3 install -r requirements.txt```

Set Google AI Studio api key:
```export GEMINI_API_KEY="<api key>"```

Scripts:
1. pull_data_final.py - pull satellite data, fill in missing data using previous values and Nearest Neighbour
2. (Optional) algae_predict_ml.py - predict next-day algae bloom based on current status and surface water temperature
    State: early prototype using U-Net architecture
3. generate_events_final.py - generate JSON events for each 5kmx5km point on the map for visualization
4. bloom_insights.py - preprocess data to find historical averages etc. and generate prompts for retrieving insights
5. pull_insights.py - use generated prompts to create insights for each 100kmx60km map grid based on current and historical data


