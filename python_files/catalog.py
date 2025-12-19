import aiohttp
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio  # loading bars for asyncio
import os
from choose_dir import replace_with_utf8_hex

BASE_SEARCH_URL = "https://na2.iiivega.com/api/search-result/search/format-groups"
BASE_EDITION_URL = "https://na2.iiivega.com/api/search-result/editions"
CONCURRENCY = 5

searchText = "*"  # use "*" to get all results

materialTypeIds = 33 # DVDs
locationIds = 59 # WR

searchTextFormat = searchText.replace(" ", "_")
searchTextFormat = replace_with_utf8_hex(searchText) # replace invalid characters with utf8-hex

directory_name = f"data/{searchTextFormat}_{materialTypeIds}_{locationIds}"

RESULTS_FILE = f"{directory_name}/wr.json"
ENHANCED_FILE = f"{directory_name}/wr_enhanced.json"
INFO_FILE = f"{directory_name}/info.json"

# see payload below for more parameters

### Vega search results ###
HEADERS = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
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
    "sec-ch-ua":
    '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "Content-Type": "application/json"
}


async def create_dir(directory_name = directory_name):
    os.makedirs(directory_name, exist_ok=True)


async def fetch_page(session, page_num, page_size, semaphore):
    payload = {
        "searchText": f"{searchText}",
        "sorting": "title",
        "sortOrder": "asc",
        "searchType": "everything",
        "universalLimiterIds": ["at_library"],  # available materials only
        "locationIds": locationIds,  
        "materialTypeIds": [materialTypeIds],  
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
            "id":
            r.get("id"),
            "title":
            r.get("title"),
            "publicationDate":
            r.get("publicationDate"),
            "author":
            r.get("primaryAgent", {}).get("label"),
            "materials": [{
                "name":
                m.get("name"),
                "type":
                m.get("type"),
                "callNumber":
                m.get("callNumber"),
                "editions": [{
                    "id": e.get("id"),
                    "publicationDate": e.get("publicationDate")
                } for e in m.get("editions", [])]
            } for m in r.get("materialTabs", [])]
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
            json.dump(
                {
                    "totalPages": total_pages,
                    "totalResults": total_results
                },
                f,
                indent=2)

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
        for coro in tqdm_asyncio.as_completed(tasks,
                                              desc="Fetching remaining pages"):
            data = await coro
            if data:
                results = parse_results(data.get("data", []))
                for r in results:
                    write_json_record(r, RESULTS_FILE, first_record)
                    first_record = False

        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("\n]\n")

        print(f"✅ Results saved to {RESULTS_FILE}")


### for editions ###
async def fetch_edition(session, edition_id):
    url = ("https://na2.iiivega.com/api/search-result"
           f"/editions/{edition_id}")

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "anonymous-user-id": "c6e7697b-de9b-4def-aab7-9994c4725500",
        "api-version": "1",
        "iii-customer-domain": "slouc.na2.iiivega.com",
        "iii-host-domain": "slouc.na2.iiivega.com",
        "priority": "u=1, i",
        "sec-ch-ua":
        "\"Google Chrome\";v=\"141\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"141\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Referer": "https://slouc.na2.iiivega.com/"
    }

    async with session.get(url, headers=headers) as response:
        return await response.json()


def process_edition(edition, sep="."):
    data = edition.get("edition", {})
    extracted = {
        "subjects": {
            k: v
            for k, v in data.items() if k.startswith("subj")
        },
        "notes": {
            k: v
            for k, v in data.items() if k.startswith("note")
        },
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
    flat = {
        k: ", ".join(v) if isinstance(v, list) else v
        for k, v in flat.items()
    }
    notes = " ".join(v for k, v in flat.items() if k.startswith("notes."))
    subjects = "; ".join(v for k, v in flat.items()
                         if k.startswith("subjects."))
    flat.update({"summary": notes, "subjects": subjects})
    flat = {
        k: v
        for k, v in flat.items()
        if not k.startswith("notes.") and not k.startswith("subjects.")
    }
    return flat


async def process_record(record, session, semaphore):
    edition_id = (record.get("materials", [])[0].get("editions",
                                                     [])[0].get("id"))
    async with semaphore:
        edition_info = await fetch_edition(session, edition_id)

    processed_edition = process_edition(edition_info)
    record.update(processed_edition)
    return record


async def editions_main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = [process_record(record, session, semaphore) for record in data]

        # tqdm_asyncio.gather gives you a live progress bar
        updated_records = await tqdm_asyncio.gather(*tasks,
                                                    desc="Fetching editions")

    with open(ENHANCED_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(vega_search())
    asyncio.run(editions_main())
