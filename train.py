import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

def train():
    df = pd.read_csv("kolkata_mega_traffic.csv")
    
    le_weather = LabelEncoder()
    le_junction = LabelEncoder()
    
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    df['junction_encoded'] = le_junction.fit_transform(df['junction_id'])
    
    X = df[['junction_encoded', 'hour', 'day_of_week', 'weather_encoded', 'is_peak_hour']]
    y = df['average_speed']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model trained. Saving to model.pkl")
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "le_weather": le_weather,
            "le_junction": le_junction
        }, f)

if __name__ == "__main__":
    train()
