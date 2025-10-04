import os
from openai import OpenAI

# Create client with your secret key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Example: chat completion
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": "You are a helpful bot."},
        {"role": "user", "content": "What is a good easy chicken recipe?"}
    ]
)

print(response.choices[0].message.content)



