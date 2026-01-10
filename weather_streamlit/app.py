import streamlit as st
from weather_agent import WeatherAgent

# API Key
API_KEY = "your_openweather_api_key"

# Initialize agent
agent = WeatherAgent(API_KEY)

# Streamlit UI
st.set_page_config(page_title="Weather Agent", page_icon="🌦")

st.title("🌦 Weather Agent")
st.write("Enter a city name to get live weather details")

city = st.text_input("Enter City Name")

if st.button("Get Weather"):
    if city:
        weather = agent.get_weather(city)

        if weather:
            st.success(f"Weather in {weather['city']}")
            st.write(f"🌡 Temperature: {weather['temperature']} °C")
            st.write(f"💧 Humidity: {weather['humidity']} %")
            st.write(f"☁ Condition: {weather['condition']}")
        else:
            st.error("City not found!")
    else:
        st.warning("Please enter a city name")
