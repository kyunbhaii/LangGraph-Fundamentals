import os
import sys
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY is not set in the .env file.")
    sys.exit(1)

print("✅ Found GEMINI_API_KEY in .env file. Testing connection...\n")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Initialize the model (gemini-pro is widely available)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=api_key
    )
    
    print("Sending a test prompt to Gemini: 'Say Hello World!'")
    response = llm.invoke("Say Hello World!")
    
    print("\n🎉 SUCCESS! Your Gemini API key is valid and has credits.")
    print(f"🤖 Gemini replied: {response.content}")

except Exception as e:
    error_msg = str(e).lower()
    print("\n❌ FAILED: API call did not succeed.")
    
    if "429" in error_msg or "resource exhausted" in error_msg or "quota" in error_msg:
        print("💡 Diagnosis: Your API key is out of credits or has exceeded its quota/rate limits.")
    elif "403" in error_msg or "invalid api key" in error_msg or "permission denied" in error_msg:
        print("💡 Diagnosis: Your API key is invalid, disabled, or doesn't have permissions.")
    elif "400" in error_msg and "billing" in error_msg:
         print("💡 Diagnosis: Your project requires a billing account to be linked.")
    else:
        print(f"💡 Diagnosis: Unknown error. Details:\n{e}")
