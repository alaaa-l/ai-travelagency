required = [
    "streamlit",
    "dotenv",
    "langgraph",
    "langchain_core",
    "numpy",
    "requests"  # or any libraries your agents use
]

for lib in required:
    try:
        __import__(lib)
        print(f"✅ {lib} is installed.")
    except ImportError:
        print(f"❌ {lib} is NOT installed. Run: pip install {lib}")
