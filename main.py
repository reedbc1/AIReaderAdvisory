import os
from openai import OpenAI
import csv
from dotenv import load_dotenv

load_dotenv()

def format_results(results):
    formatted_results = ''
    for result in results.data:
        formatted_result = f"<result file_id='{result.file_id}' file_name='{result.file_name}'>"
        for part in result.content:
            formatted_result += f"<content>{part.text}</content>"
        formatted_results += formatted_result + "</result>"
    return f"<sources>{formatted_results}</sources>"

def lookup():
    api_key = os.getenv('OPENAI_API_KEY')

    # Create client with your secret key
    client = OpenAI()

    # vector_store = client.vector_stores.create(
    #     name = "WR Items"
    # )

    # client.vector_stores.files.upload_and_poll(
    #         vector_store_id = vector_store.id,
    #         file = open('wr.txt', mode = 'rb')
    # )

    # print(vector_store.id)

    user_query = "Book"

    results = client.vector_stores.search(
        vector_store_id="vs_68e10330bd0c8191aa31018bf4f228ac",
        query=user_query,
    )

    print(results)

    formatted_results = format_results(results.data)

    '\n'.join('\n'.join(c.text) for c in result.content for result in results.data)

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "developer",
                "content": "Produce a concise answer to the query based on the provided sources."
            },
            {
                "role": "user",
                "content": f"Sources: {formatted_results}\n\nQuery: '{user_query}'"
            }
        ],
    )

    print(completion.choices[0].message.content)

def query():  
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5-nano",
        input="Could you give me a summary of a random book from the vector store from the provided website?",
        tools=[
            {
            "type": "file_search",
            "vector_store_ids": ["vs_68e10330bd0c8191aa31018bf4f228ac"]
            },
            {
            "type": "web_search",
            "filters": {
                "allowed_domains": [
                    "slouc.na2.iiivega.com"
                ]
            }
            }
        ]
    )
    print(response)

query()


