from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get the API key
api_key = os.getenv("DEEPSEEK_API_KEY")

if api_key:
    print("✅ DEEPSEEK_API_KEY found:", api_key)
else:
    print("❌ DEEPSEEK_API_KEY NOT found. Check your .env file.")
