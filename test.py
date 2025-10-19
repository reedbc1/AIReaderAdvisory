import aiohttp
import asyncio
import async_timeout
import json
import os
from tqdm.asyncio import tqdm_asyncio
import requests

# ---------------- CONFIG ----------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/141.0.0.0 Safari/537.36",
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
    "sec-fetch-site": "same-site"
}

API_BASE = "https://na2.iiivega.com/api/search-result/editions/"
SEARCH_URL = "https://na2.iiivega.com/api/search-result/search/format-groups"

REQUESTS_PER_SECOND = 2
MAX_CONCURRENT = 10
MAX_RETRIES = 3
RESUME_FILE = "enhanced_results.jsonl"
OUTPUT_FILE = "enhanced_results.json"

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ---------------- VEGA SEARCH WITH PAGINATION ----------------
def vega_search_all(page_size=40):
    """Fetch all format groups from Vega using pagination."""
    all_results = []
    page_num = 0
    while True:
        data = {
            "searchText": "*",
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
        headers = HEADERS.copy()
        headers.update({"method": "POST", "content-type": "application/json"})
        response = requests.post(SEARCH_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        data_page = result.get("data", [])
        if not data_page:
            break
        all_results.extend(data_page)
        print(f"Fetched page {page_num} ({len(data_page)} records)")
        page_num += 1
    print(f"✅ Total records fetched: {len(all_results)}")
    with open("vega_results_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return all_results

# ---------------- PARSE RESULTS ----------------
def parse_results(results):
    """Parse Vega search results into a simplified structure."""
    new_results = []
    for record in results:
        new_results.append({
            "id": record.get("id"),
            "title": record.get("title"),
            "publicationDate": record.get("publicationDate"),
            "author": record.get("primaryAgent", {}).get("label"),
            "materials": [
                {
                    "name": tab.get("name"),
                    "type": tab.get("type"),
                    "callNumber": tab.get("callNumber"),
                    "editions": [{
                        "id": edition.get("id"),
                        "publicationDate": edition.get("publicationDate")
                    } for edition in tab.get("editions", [])]
                } for tab in record.get("materialTabs", [])
            ]
        })
    with open("iliad_partial.json", "w", encoding="utf-8") as f:
        json.dump(new_results, f, indent=2, ensure_ascii=False)
    return new_results

# ---------------- ASYNC EDITION FETCH ----------------
async def get_edition_async(edition_id, session, r_idx, m_idx, e_idx):
    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                await asyncio.sleep(1 / REQUESTS_PER_SECOND)
                async with async_timeout.timeout(15):
                    url = f"{API_BASE}{edition_id}"
                    async with session.get(url, headers=HEADERS) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return edition_id, data, r_idx, m_idx, e_idx
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"❌ Failed {edition_id} after {MAX_RETRIES} attempts: {e}")
                    return edition_id, None, r_idx, m_idx, e_idx
                await asyncio.sleep(attempt * 2)

def parse_and_flatten_edition(edition, sep='.'):
    data = edition.get("edition", {})
    extracted = {
        "subjects": {k: v for k, v in data.items() if k.startswith("subj")},
        "notes": {k: v for k, v in data.items() if k.startswith("note")},
        "contributors": data.get("contributors", [])
    }
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    flat = flatten_dict(extracted)
    flat = {k: ', '.join(v) if isinstance(v, list) else v for k, v in flat.items()}
    notes_parts = [v for k, v in flat.items() if k.startswith("notes.")]
    flat["notes"] = " ".join(notes_parts)
    subject_parts = [v for k, v in flat.items() if k.startswith("subjects.")]
    flat["subjects"] = "; ".join(subject_parts)
    flat = {k: v for k, v in flat.items() if not (k.startswith("notes.") or k.startswith("subjects."))}
    return flat

def load_resume_data():
    processed_ids, existing_data = set(), {}
    if not os.path.exists(RESUME_FILE):
        return processed_ids, existing_data
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                eid = entry.get("edition_id")
                if eid:
                    processed_ids.add(eid)
                    existing_data[eid] = entry.get("data")
            except:
                continue
    return processed_ids, existing_data

async def enhance_results_parallel(results):
    processed_ids, existing_data = load_resume_data()

    # Update results with already fetched data
    for result in results:
        for material in result.get("materials", []):
            for idx, edition in enumerate(material.get("editions", [])):
                eid = edition.get("id")
                if eid in existing_data:
                    material["editions"][idx] = {**edition, **existing_data[eid]}

    edition_map = [
        (r_idx, m_idx, e_idx, eid)
        for r_idx, result in enumerate(results)
        for m_idx, material in enumerate(result.get("materials", []))
        for e_idx, edition in enumerate(material.get("editions", []))
        if (eid := edition.get("id")) and eid not in processed_ids
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [get_edition_async(eid, session, r_idx, m_idx, e_idx) for r_idx, m_idx, e_idx, eid in edition_map]
        for fetched_id, data, r_idx, m_idx, e_idx in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Fetching editions"):
            if not data:
                continue
            parsed_data = parse_and_flatten_edition(data)
            results[r_idx]["materials"][m_idx]["editions"][e_idx] = {
                **results[r_idx]["materials"][m_idx]["editions"][e_idx],
                **parsed_data
            }
            with open(RESUME_FILE, "a", encoding="utf-8") as f_out:
                json.dump({
                    "result_index": r_idx,
                    "material_index": m_idx,
                    "edition_index": e_idx,
                    "edition_id": fetched_id,
                    "data": results[r_idx]["materials"][m_idx]["editions"][e_idx]
                }, f_out, ensure_ascii=False)
                f_out.write("\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_final:
        json.dump(results, f_final, ensure_ascii=False, indent=2)
    print(f"✅ Finished! Combined JSON saved to {OUTPUT_FILE}")
    return results

# ---------------- RUN FULL PIPELINE ----------------
def main():
    vega_results = vega_search_all()
    parsed_results = parse_results(vega_results)
    asyncio.run(enhance_results_parallel(parsed_results))

if __name__ == "__main__":
    main()
