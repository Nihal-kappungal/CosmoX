from openai import OpenAI
import os
print("OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))
# quick client ping (Responses client)
client = OpenAI()
resp = client.models.list()  # lists models (permission required)
print("Models count:", len(resp.data))
