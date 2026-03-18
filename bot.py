import os
import re
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd
import pickle
import requests
from datetime import datetime, timedelta
from pyngrok import ngrok
import google.generativeai as genai

# Streamlit secrets TOML shim for standalone running
def load_local_secrets():
    try:
        with open(".streamlit/secrets.toml", "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
    except:
        pass
load_local_secrets()

app = Flask(__name__)

# Load the AI model once when the bot starts
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
le_weather = data["le_weather"]
le_junction = data["le_junction"]

def get_live_weather():
    """Fetches real-time weather, or falls back to sensible mock data."""
    api_key = "MOCK_API_KEY_PLACEHOLDER"
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Kolkata&appid={api_key}&units=metric"
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            cond = data['weather'][0]['main']
            if cond in ['Rain', 'Drizzle', 'Thunderstorm']:
                mapped_cond = 'Rainy'
            elif cond in ['Fog', 'Mist', 'Haze']:
                mapped_cond = 'Foggy'
            else:
                mapped_cond = 'Sunny'
            return mapped_cond
    except:
        pass
    return "Rainy"

def get_fastest_route():
    """Runs the ML model for 3 Kolkata routes and returns the fastest one."""
    routes = [
        {"name": "Technopolis Bypass", "junction": "Technopolis Crossing"},
        {"name": "New Town Link", "junction": "Biswa Bangla Gate"},
        {"name": "Mahisbathan Internal", "junction": "Chinar Park"}
    ]
    
    now = datetime.now()
    current_hour = now.hour
    day_of_week = now.weekday()
    is_peak_hour = 1 if (8 <= current_hour <= 11) or (17 <= current_hour <= 20) else 0
    
    model_weather = get_live_weather()
    weather_encoded = le_weather.transform([model_weather])[0]
    
    dist_km = 12.0
    best_time = float('inf')
    best_route = ""
    
    for r in routes:
        j_encoded = le_junction.transform([r["junction"]])[0]
        X_input = pd.DataFrame({
            'junction_encoded': [j_encoded],
            'hour': [current_hour],
            'day_of_week': [day_of_week],
            'weather_encoded': [weather_encoded],
            'is_peak_hour': [is_peak_hour]
        })
        
        speed = model.predict(X_input)[0]
        travel_time_mins = (dist_km / max(float(speed), 1.0)) * 60
        
        if travel_time_mins < best_time:
            best_time = travel_time_mins
            best_route = r["name"]
            
    return best_route, best_time, model_weather

@app.route("/webhook", methods=['POST'])
def whatsapp_webhook():
    """Endpoint for Twilio WhatsApp Webhook."""
    incoming_msg = request.values.get('Body', '').strip().lower()
    latitude = request.values.get('Latitude')
    longitude = request.values.get('Longitude')
    
    # Initialize Twilio response
    resp = MessagingResponse()
    msg = resp.message()
    
    if latitude and longitude:
        import json
        live_data = {"lat": float(latitude), "lon": float(longitude), "timestamp": str(datetime.now())}
        with open("live_location.json", "w") as f:
            json.dump(live_data, f)
            
        msg.body("📍 Path Saathi has received your live location. Calculating the fastest 'Green' route to Sector V now...")
        return str(resp)
        
    if 'status' in incoming_msg or 'route' in incoming_msg:
        # Run AI logic
        fastest_route, travel_time, weather = get_fastest_route()
        
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY":
            try:
                genai.configure(api_key=gemini_key)
                model_gen = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                You are 'Path Saathi' - an intelligent commuter AI. Current weather: {weather}. 
                Our local Traffic ML Model processed 3 routes and found the fastest bypass is {fastest_route} taking precisely {travel_time:.0f} mins.
                
                Write a 2-3 sentence WhatsApp message to the user delivering this exact status. Be witty, warm, and hyper-helpful (mention grasping a quick chai or coffee if they save time!).
                Include exact times and routes natively in the text.
                """
                response = model_gen.generate_content(prompt)
                response_text = response.text.strip()
            except Exception as e:
                response_text = f"🚦 *Path Saathi Live AI Update*\n\n⛅ Weather: {weather}\n🚀 Fastest Route: *{fastest_route}*\n⏱️ Est. Travel Time: {travel_time:.0f} mins"
        else:
            response_text = f"🚦 *Path Saathi Live AI Update*\n\n⛅ Weather: {weather}\n🚀 Fastest Route: *{fastest_route}*\n⏱️ Est. Travel Time: {travel_time:.0f} mins\n\nDrive safe!"
            
        msg.body(response_text)
    else:
        msg.body("Hi! I'm your Path Saathi AI. Send 'Status' to get your live fastest commute route for Kolkata.")
        
    return str(resp)

if __name__ == "__main__":
    # Start ngrok tunnel auto-provisioning
    port = 5000
    public_url = ngrok.connect(port).public_url
    print(f"\\n{'='*50}\\n✅ NGROK TUNNEL ACTIVE: Copy this Webhook URL into Twilio Sandbox:")
    print(f"👉 {public_url}/webhook\\n{'='*50}\\n")
    
    app.run(port=port, debug=True, use_reloader=False)
