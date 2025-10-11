import os
from openai import OpenAI
import csv
from dotenv import load_dotenv
import requests
import json

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

def vega_api():

    url = "https://na2.iiivega.com/api/search-result/search/format-groups"

    headers = {
        "authority": "na2.iiivega.com",
        "method": "POST",
        "path": "/api/search-result/search/format-groups",
        "scheme": "https",
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "anonymous-user-id": "c6aeabfe-dcc0-4e1a-8fa2-3934d465cb70",
        "api-version": "2",
        "iii-customer-domain": "slouc.na2.iiivega.com",
        "iii-host-domain": "slouc.na2.iiivega.com",
        "origin": "https://slouc.na2.iiivega.com",
        "priority": "u=1, i",
        "referer": "https://slouc.na2.iiivega.com/",
        "sec-ch-ua": '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/141.0.0.0 Safari/537.36"
        ),
        "content-type": "application/json",
    }

    payload = {
        "searchText": "*",
        "sorting": "relevance",
        "sortOrder": "asc",
        "searchType": "everything",
        "universalLimiterIds": ["at_library"],
        "materialTypeIds": ["35", "1"],
        "intendedAudienceIds": ["adolescent", "adult"],
        "locationIds": ["59"],
        "pageNum": 0,
        "pageSize": 40,
        "resourceType": "FormatGroup",
    }

    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Check the result
    if response.ok:
        data = response.json()
        with open("vega_results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(data)
    else:
        print("❌ Request failed:", response.status_code, response.text)


# vega_api()

def extract_json():
    with open('vega_results.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    results = []

    for item in data["data"]:
        title = item["title"]
        publicationDate = item.get("publicationDate")
        author = item.get("primaryAgent", {}).get("label")
        callNumber = item.get("materialTabs", [{}])[0].get("callNumber").replace("  "," ") # where name == book
        shelfLocation = item.get("materialTabs", [{}])[0].get("itemShelfLocation")
        collectionLocation = item.get("materialTabs", [{}])[0].get("itemCollectionLocation")
        isbn = item.get("identifiers", {}).get("isbn")

        # isbn = "9780439139595"

        base_url = "https://openlibrary.org"
        append = f"/isbn/{isbn}.json"
        url = base_url + append

        response = requests.get(url)

        if response.ok:
            data = response.json()
            works = data.get("works", [{}])
            if works:
                new_append = works[0]["key"] + ".json"
                # print(new_append)
            else:
                new_append = None

        if new_append:
            url = base_url + new_append
            response = requests.get(url)
            if response.ok:
                data = response.json()
                # print(data)
                description = data.get("description")
                subject_places = data.get("subject_places", [])
                subjects = data.get("subjects", [])
            else:
                print("oopsie daisies")
        
        data_json = {
            "title": title,
            "publicationDate": publicationDate,
            "author": author,
            "callNumber": callNumber,
            "shelfLocation": shelfLocation,
            "collectionLocation": collectionLocation,
            "isbn": isbn,
            "description": description,
            "subject_places": subject_places,
            "subjects": subjects
        }

        results.append(data_json)

    return results

# print(extract_json())

def item_level():
    import requests

    url = "https://na2.iiivega.com/api/search-result/search/items"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "iii-customer-domain": "slouc.na2.iiivega.com",
        "origin": "https://slouc.na2.iiivega.com",
        "anonymous-user-id": "your-copied-uuid",
        "user-agent": "Mozilla/5.0 ..."
    }
    payload = {
        "searchText": "pottery",
        "sorting": "relevance",
        "sortOrder": "asc",
        "searchType": "everything",
        "universalLimiterIds": ["at_library"],
        "locationIds": ["59"],
        "pageNum": 0,
        "pageSize": 40,
        "resourceType": "Item"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    print(data)

item_level()







