import os
from dotenv import load_dotenv
# Corrected import for your version
from langfuse import observe, get_client

# 1. Load env with override
load_dotenv(override=True)

@observe(name="Project_Confirmation_Test")
def run_test():
    print(f"Sending test to: {os.getenv('LANGFUSE_BASE_URL')}")
    print(f"Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY')[:10]}...")
    return "Handshake Success"

if __name__ == "__main__":
    run_test()
    # In v4, we get the client instance and flush it
    get_client().flush()
    print("\n✅ Sent! Now check the NEW project in Langfuse.")
