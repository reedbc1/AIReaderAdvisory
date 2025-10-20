import aiohttp
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio  # loading bars for asyncio
import random
from tqdm.asyncio import tqdm_asyncio

RESULTS_FILE = "horror.json"
ENHANCED_FILE = "horror_enhanced.json"
INFO_FILE = "info.json"
BASE_SEARCH_URL = "https://na2.iiivega.com/api/search-result/search/format-groups"
BASE_EDITION_URL = "https://na2.iiivega.com/api/search-result/editions"
CONCURRENCY = 5
searchText = "horror"

### Vega search results ###
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


async def fetch_page(session, page_num, page_size, semaphore):
    payload = {
        "searchText": searchText,
        "sorting": "relevance",
        "sortOrder": "asc",
        "searchType": "everything",
        "universalLimiterIds": ["at_library"],
        "materialTypeIds": ["33"],
        "locationIds": ["59"],
        "pageNum": page_num,
        "pageSize": page_size,
        "resourceType": "FormatGroup"
    }

    async with semaphore:
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
    page_size = 1000
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        # Fetch first page for metadata
        first_data = await fetch_page(session, 0, page_size, semaphore)
        if not first_data:
            print("❌ Failed to fetch first page.")
            return

        total_pages = first_data.get("totalPages", 1)
        total_results = first_data.get("totalResults", 0)
        print(f"✅ Found {total_results} results across {total_pages} pages.")

        with open(INFO_FILE, "w", encoding="utf-8") as f:
            json.dump({"totalPages": total_pages, "totalResults": total_results}, f, indent=2)

        # Start output file
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("[\n")

        first_record = True

        # Write first page
        results = parse_results(first_data.get("data", []))
        for r in results:
            write_json_record(r, RESULTS_FILE, first_record)
            first_record = False

        # Create all tasks for remaining pages
        tasks = [
            fetch_page(session, page_num, page_size, semaphore)
            for page_num in range(1, total_pages)
        ]

        # Process pages concurrently with progress bar
        for coro in tqdm_asyncio.as_completed(tasks, desc="Fetching remaining pages"):
            data = await coro
            if data:
                results = parse_results(data.get("data", []))
                for r in results:
                    write_json_record(r, RESULTS_FILE, first_record)
                    first_record = False

        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("\n]\n")

        print(f"✅ Results saved to {RESULTS_FILE}")


### Get edition info ###
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


# Config
CONCURRENCY = 10          # how many "results" to process at once
MAX_RETRIES = 3           # retry failed edition fetches
RETRY_BACKOFF = (1, 4)    # seconds between retries (min, max)


async def fetch_with_retries(session, url, max_retries=MAX_RETRIES):
    """Fetch a URL with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    print(f"❌ {resp.status} for {url}: {text[:100]}...")
        except Exception as e:
            print(f"⚠️ Exception for {url}: {e}")

        if attempt < max_retries - 1:
            sleep_time = random.uniform(*RETRY_BACKOFF) * (2 ** attempt)
            await asyncio.sleep(sleep_time)
    return None


async def get_edition(session, edition_id):
    url = f"https://na2.iiivega.com/api/search-result/editions/{edition_id}"
    return await fetch_with_retries(session, url)


async def process_result(semaphore, session, result):
    async with semaphore:
        updated_materials = []
        for material in result.get("materials", []):
            editions = material.get("editions", [])
            if not editions:
                updated_materials.append(material)
                continue

            # Fetch all editions concurrently for this material
            tasks = [get_edition(session, e.get("id")) for e in editions if e.get("id")]
            edition_responses = await asyncio.gather(*tasks, return_exceptions=True)

            new_editions = []
            for e, data in zip(editions, edition_responses):
                if isinstance(data, Exception) or not data:
                    continue
                parsed = parse_and_flatten_edition(data)
                new_editions.append({**e, **parsed})

            updated_materials.append({**material, "editions": new_editions})

        result["materials"] = updated_materials
        return result


async def enhance_results():
    """Enhance result JSON by fetching editions concurrently with retries."""
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
        "User-Agent": HEADERS["User-Agent"],
    }

    # Load your existing search results
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(headers=HEADERS_EDITION) as session:
        # Start new JSON array output file
        with open(ENHANCED_FILE, "w", encoding="utf-8") as f:
            f.write("[\n")

        first_record = True
        tasks = [process_result(semaphore, session, r) for r in results]

        # Single clean progress bar for total results
        for coro in tqdm_asyncio.as_completed(tasks, desc="Enhancing results", total=len(tasks)):
            result = await coro
            write_json_record(result, ENHANCED_FILE, first_record)
            first_record = False

        with open(ENHANCED_FILE, "a", encoding="utf-8") as f:
            f.write("\n]\n")



if __name__ == "__main__":
    asyncio.run(vega_search())
    asyncio.run(enhance_results())
