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
    '\n'.join('\n'.join(c.text) for c in results.content
              for result in results.data)

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role":
            "developer",
            "content":
            "Produce a concise answer to the query based on the provided sources."
        }, {
            "role":
            "user",
            "content":
            f"Sources: {formatted_results}\n\nQuery: '{user_query}'"
        }],
    )

    print(completion.choices[0].message.content)


def query():
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5-nano",
        input=
        "Could you give me a summary of a random book from the vector store from the provided website?",
        tools=[{
            "type": "file_search",
            "vector_store_ids": ["vs_68e10330bd0c8191aa31018bf4f228ac"]
        }, {
            "type": "web_search",
            "filters": {
                "allowed_domains": ["slouc.na2.iiivega.com"]
            }
        }])
    print(response)


# query()


def vega_api():

    url = "https://na2.iiivega.com/api/search-result/search/format-groups"

    headers = {
        "authority":
        "na2.iiivega.com",
        "method":
        "POST",
        "path":
        "/api/search-result/search/format-groups",
        "scheme":
        "https",
        "accept":
        "application/json, text/plain, */*",
        "accept-encoding":
        "gzip, deflate, br, zstd",
        "accept-language":
        "en-US,en;q=0.9",
        "anonymous-user-id":
        "c6aeabfe-dcc0-4e1a-8fa2-3934d465cb70",
        "api-version":
        "2",
        "iii-customer-domain":
        "slouc.na2.iiivega.com",
        "iii-host-domain":
        "slouc.na2.iiivega.com",
        "origin":
        "https://slouc.na2.iiivega.com",
        "priority":
        "u=1, i",
        "referer":
        "https://slouc.na2.iiivega.com/",
        "sec-ch-ua":
        '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
        "sec-ch-ua-mobile":
        "?0",
        "sec-ch-ua-platform":
        '"Windows"',
        "sec-fetch-dest":
        "empty",
        "sec-fetch-mode":
        "cors",
        "sec-fetch-site":
        "same-site",
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/141.0.0.0 Safari/537.36"),
        "content-type":
        "application/json",
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
        callNumber = item.get("materialTabs",
                              [{}])[0].get("callNumber").replace(
                                  "  ", " ")  # where name == book
        shelfLocation = item.get("materialTabs",
                                 [{}])[0].get("itemShelfLocation")
        collectionLocation = item.get("materialTabs",
                                      [{}])[0].get("itemCollectionLocation")
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


def create_json():
    import pandas as pd
    import json

    # Load your exported SimplyReports CSV
    df = pd.read_csv("csv/wr_items.csv")

    # Convert to JSON records
    records = df.to_dict(orient='records')

    # Save to file
    with open('library_books.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def enrich_data():
    import requests
    import pandas as pd
    import json
    import time

    # --- CONFIGURATION ---
    INPUT_CSV = "csv/wr_items.csv"
    OUTPUT_JSON = "library_books_enriched.json"
    SLEEP_BETWEEN_REQUESTS = 0.5  # seconds, to avoid rate limits

    # --- LOAD SIMPLYREPORTS EXPORT ---
    df = pd.read_csv(INPUT_CSV)
    df = df[0:10]
    print(type(df["ISBN"][0]))
    df['ISBN'] = df['ISBN'].apply(lambda x: '%.9f' % x)
    df["ISBN"] = [item.split(".")[0] for item in df["ISBN"]]
    print(df["ISBN"])

    # Try to detect likely column names automatically
    possible_isbn_cols = [col for col in df.columns if "isbn" in col.lower()]
    possible_title_cols = [col for col in df.columns if "title" in col.lower()]
    possible_author_cols = [
        col for col in df.columns if "author" in col.lower()
    ]

    isbn_col = possible_isbn_cols[0] if possible_isbn_cols else None
    title_col = possible_title_cols[0] if possible_title_cols else None
    author_col = possible_author_cols[0] if possible_author_cols else None

    print(
        f"Using ISBN column: {isbn_col}, Title column: {title_col}, Author column: {author_col}"
    )

    # --- FUNCTION TO QUERY GOOGLE BOOKS API ---
    def get_book_data(isbn=None, title=None, author=None):
        if isbn:
            query = f"isbn:{isbn}"
        elif title:
            query = f"intitle:{title}"
            if author:
                query += f"+inauthor:{author}"
        else:
            return None

        url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
        response = requests.get(url)
        print(response)
        if not response.ok:
            return None

        data = response.json()
        if "items" not in data:
            return None

        info = data["items"][0]["volumeInfo"]

        return {
            "title": info.get("title"),
            "authors": info.get("authors", []),
            "publishedDate": info.get("publishedDate"),
            "publisher": info.get("publisher"),
            "pageCount": info.get("pageCount"),
            "categories": info.get("categories", []),
            "description": info.get("description"),
            "industryIdentifiers": info.get("industryIdentifiers", []),
        }

    # --- ENRICH DATA ---
    records = []
    for _, row in df.iterrows():
        isbn = row[isbn_col] if isbn_col else None
        title = row[title_col] if title_col else None
        author = row[author_col] if author_col else None

        book_data = get_book_data(isbn=isbn, title=title, author=author)
        if book_data:
            record = {
                "local_title": title,
                "local_author": author,
                "local_isbn": isbn,
                "google_books_info": book_data
            }
            records.append(record)
            print(f"✅ Found: {book_data['title']}")
        else:
            print(f"❌ No data found for: {title}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # --- SAVE TO JSON ---
    print(len(records))
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✨ Enriched data saved to {OUTPUT_JSON}")


# enrich_data()

import json

with open("library_books_enriched.json", "r", encoding="utf-8") as f:
    books = json.load(f)


def book_to_text(book):
    return f"""Title: {book['local_title']}
        Author: {book['local_author']}
        Categories: {', '.join(book.get('categories', []))}
        Summary: {book.get('description', 'No summary available')}"""


book_texts = [book_to_text(book) for book in books]

print(book_texts)
