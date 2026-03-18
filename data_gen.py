import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    np.random.seed(42)
    junctions = [
        "Technopolis Crossing",
        "Wipro More",
        "Biswa Bangla Gate",
        "Ultadanga Flyover",
        "Chinar Park",
        "Park Street",
        "Sealdah",
        "Howrah Bridge"
    ]
    weather_conditions = ["Sunny", "Rainy", "Foggy"]
    
    start_date = datetime(2023, 1, 1)
    
    data = []
    
    # 105 days * 24 hours * 8 junctions = 20,160 rows
    for day in range(105):
        for hour in range(24):
            current_time = start_date + timedelta(days=day, hours=hour)
            is_weekend = 1 if current_time.weekday() >= 5 else 0
            
            # Peak hours: 8 AM - 11 AM & 5 PM - 8 PM
            is_peak = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
            
            for j in junctions:
                base_speed = 60
                
                weather = np.random.choice(weather_conditions, p=[0.7, 0.2, 0.1])
                speed_penalty = 0
                
                if is_peak:
                    speed_penalty += 25
                    if j in ["Technopolis Crossing", "Wipro More", "Ultadanga Flyover", "Sealdah"]:
                        speed_penalty += 15  # High congestion bottlenecks during rush hour
                
                # Weekend patterns: lower office traffic
                if is_weekend and j in ["Technopolis Crossing", "Wipro More", "Biswa Bangla Gate"]:
                    speed_penalty -= 20  # Fast on weekends in IT hubs
                    
                average_speed = base_speed - speed_penalty + np.random.normal(0, 5)
                
                # Weather = 40% reduction for adverse conditions
                if weather in ["Rainy", "Foggy"]:
                    average_speed = average_speed * 0.60
                    
                average_speed = max(5, average_speed)
                
                vehicle_count = max(0, int((60 - average_speed) * 3 + np.random.normal(0, 10)))
                
                data.append({
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "junction_id": j,
                    "vehicle_count": vehicle_count,
                    "average_speed": average_speed,
                    "weather": weather,
                    "is_peak_hour": is_peak,
                    "hour": hour,
                    "day_of_week": current_time.weekday()
                })
                
    df = pd.DataFrame(data)
    df.to_csv("kolkata_mega_traffic.csv", index=False)
    print(f"Generated {len(df)} rows and saved to kolkata_mega_traffic.csv")

if __name__ == "__main__":
    generate_data()
