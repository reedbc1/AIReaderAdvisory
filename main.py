import aiohttp
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio  # loading bars for asyncio

# Files
RESULTS_FILE = "vega_results.json"
ENHANCED_FILE = "enhanced_results.json"
INFO_FILE = "info.json"

# URLs
BASE_SEARCH_URL = "https://na2.iiivega.com/api/search-result/search/format-groups"
BASE_EDITION_URL = "https://na2.iiivega.com/api/search-result/editions"

# Concurrency
CONCURRENCY = 5

# Headers (search)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
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
    "Content-Type": "application/json"
}

async def fetch_page(session, page_num, page_size):
    payload = {
        "searchText": "ILIAD",
        "sorting": "relevance",
        "sortOrder": "asc",
        "searchType": "everything",
        "pageNum": page_num,
        "pageSize": page_size,
        "resourceType": "FormatGroup"
    }
    async with session.post(BASE_SEARCH_URL, json=payload) as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            text = await resp.text()
            print(f"❌ Error {resp.status} on page {page_num}: {text}")
            return None

def parse_results(records):
    parsed = []
    for r in records:
        parsed.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "publicationDate": r.get("publicationDate"),
            "author": r.get("primaryAgent", {}).get("label"),
            "materials": [
                {
                    "name": m.get("name"),
                    "type": m.get("type"),
                    "callNumber": m.get("callNumber"),
                    "editions": [
                        {"id": e.get("id"), "publicationDate": e.get("publicationDate")}
                        for e in m.get("editions", [])
                    ]
                }
                for m in r.get("materialTabs", [])
            ]
        })
    return parsed

def write_json_record(record, filename, first):
    with open(filename, "a", encoding="utf-8") as f:
        if not first:
            f.write(",\n")
        json.dump(record, f, ensure_ascii=False, indent=2)

async def vega_search():
    page_size = 40
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        first_data = await fetch_page(session, 0, page_size)
        if not first_data:
            print("❌ Failed to fetch first page.")
            return

        total_pages = first_data.get("totalPages", 1)
        total_results = first_data.get("totalResults", 0)
        print(f"✅ Found {total_results} results across {total_pages} pages.")

        with open(INFO_FILE, "w", encoding="utf-8") as f:
            json.dump({"totalPages": total_pages, "totalResults": total_results}, f, indent=2)

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("[\n")

        first_record = True
        results = parse_results(first_data.get("data", []))
        for r in results:
            write_json_record(r, RESULTS_FILE, first_record)
            first_record = False

        pages = list(range(1, total_pages))
        for i in range(0, len(pages), CONCURRENCY):
            batch = pages[i:i+CONCURRENCY]
            tasks = [fetch_page(session, p, page_size) for p in batch]
            responses = await tqdm_asyncio.gather(*tasks, desc=f"Fetching pages {batch[0]}-{batch[-1]}")
            for data in responses:
                if data:
                    results = parse_results(data.get("data", []))
                    for r in results:
                        write_json_record(r, RESULTS_FILE, first_record)
                        first_record = False

        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("\n]\n")

async def get_edition(session, edition_id):
    async with session.get(f"{BASE_EDITION_URL}/{edition_id}") as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            text = await resp.text()
            print(f"❌ Error {resp.status} for edition {edition_id}: {text}")
            return None

def parse_and_flatten_edition(edition, sep="."):
    data = edition.get("edition", {})
    extracted = {
        "subjects": {k: v for k, v in data.items() if k.startswith("subj")},
        "notes": {k: v for k, v in data.items() if k.startswith("note")},
        "contributors": data.get("contributors", [])
    }

    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat = flatten_dict(extracted)
    flat = {k: ", ".join(v) if isinstance(v, list) else v for k, v in flat.items()}
    notes = " ".join(v for k, v in flat.items() if k.startswith("notes."))
    subjects = "; ".join(v for k, v in flat.items() if k.startswith("subjects."))
    flat.update({"notes": notes, "subjects": subjects})
    flat = {k: v for k, v in flat.items() if not k.startswith("notes.") and not k.startswith("subjects.")}
    return flat

async def enhance_results():
    HEADERS_EDITION = {
        "authority": "na2.iiivega.com",
        "method": "GET",
        "scheme": "https",
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "anonymous-user-id": "c6aeabfe-dcc0-4e1a-8fa2-3934d465cb70",
        "api-version": "1",
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
        "User-Agent": HEADERS["User-Agent"]
    }

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    async with aiohttp.ClientSession(headers=HEADERS_EDITION) as session:
        with open(ENHANCED_FILE, "w", encoding="utf-8") as f:
            f.write("[\n")
        first_record = True

        # Only show progress for the overall results
        for result in tqdm_asyncio(results, desc="Enhancing results"):
            updated_materials = []
            for material in result.get("materials", []):
                editions = material.get("editions", [])
                new_editions = []

                # Fetch editions without a separate progress bar
                tasks = [get_edition(session, e.get("id")) for e in editions if e.get("id")]
                edition_responses = await asyncio.gather(*tasks)
                for e, data in zip(editions, edition_responses):
                    if not data:
                        continue
                    parsed = parse_and_flatten_edition(data)
                    new_editions.append({**e, **parsed})

                updated_materials.append({**material, "editions": new_editions})

            result["materials"] = updated_materials
            write_json_record(result, ENHANCED_FILE, first_record)
            first_record = False

        with open(ENHANCED_FILE, "a", encoding="utf-8") as f:
            f.write("\n]\n")


if __name__ == "__main__":
    asyncio.run(vega_search())
    asyncio.run(enhance_results())
