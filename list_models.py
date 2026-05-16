import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("\n=== Models that support embedContent ===\n")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(f"  {m.name}")

print("\n=== Models that support generateContent ===\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")
