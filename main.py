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

# query()

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
    possible_author_cols = [col for col in df.columns if "author" in col.lower()]
    
    isbn_col = possible_isbn_cols[0] if possible_isbn_cols else None
    title_col = possible_title_cols[0] if possible_title_cols else None
    author_col = possible_author_cols[0] if possible_author_cols else None
    
    print(f"Using ISBN column: {isbn_col}, Title column: {title_col}, Author column: {author_col}")
    
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
