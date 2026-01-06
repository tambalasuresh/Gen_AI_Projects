import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import datetime

# ================== TEXT TO SPEECH ==================
def speak(text):
    print("Assistant:", text)

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    # Split long text into small chunks
    max_length = 150
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    for chunk in chunks:
        engine.say(chunk)
        engine.runAndWait()

# ================== SPEECH TO TEXT ==================
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You:", text)
        return text.lower()
    except:
        return ""

# ================== LOAD AI MODEL ==================
ai_model = pipeline(
    "text-generation",
    model="gpt2",
    pad_token_id=50256
)

# ================== AI BRAIN ==================
def brain(text):
    # Rule-based commands
    if "time" in text:
        return datetime.datetime.now().strftime("The time is %H:%M")

    if "date" in text:
        return datetime.datetime.now().strftime("Today is %B %d, %Y")

    if "your name" in text:
        return "My name is Suresh Assistant"

    # AI response
    response = ai_model(
        text,
        max_length=60,
        temperature=0.7,
        num_return_sequences=1
    )

    reply = response[0]["generated_text"]

    # Remove repeated input
    reply = reply.replace(text, "").strip()

    return reply

# ================== MAIN LOOP ==================
speak("Hello Suresh. I am ready.")

while True:
    command = listen()

    if command == "":
        continue

    if "stop" in command or "exit" in command:
        speak("Goodbye Suresh")
        break

    reply = brain(command)
    print("AI:", reply)
    speak(reply)
