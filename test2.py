# import dvd json
import json
import requests

### Get results from vega search ###
def vega_search():
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
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
      "content-type": "application/json"
  }

  data = {
      "searchText": "*",
      "sorting": "relevance",
      "sortOrder": "asc",
      "searchType": "everything",
      "universalLimiterIds": ["at_library"],
      "materialTypeIds": ["33"],
      "locationIds": ["59"],
      "pageNum": 0,
      "pageSize": 40,
      "resourceType": "FormatGroup"
  }

  response = requests.post(url, headers=headers, data=json.dumps(data))

  if response.ok:
      print("✅ Success!")
      data = response.json()
      with open("vega_results.json", "w", encoding="utf-8") as f:
          json.dump(data, f, indent=2, ensure_ascii=False)

  else:
      print(f"❌ Error {response.status_code}")
      print(response.text)

##### The Iliad #####
# iliad.json is from vega_search
def iliad_test():
  with open("iliad.json", "r", encoding="utf-8") as f:
    results = json.load(f)

  data = results.get("data", [])
  new_results = []
  for record in data:
    new_results.append(
      {
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
                 "publicationDate": edition.get("publicationDate")} 
                            for edition in tab.get("editions",[])]
          }
          for tab in record.get("materialTabs")
        ]
      }
    )

  with open("iliad_partial.json", "w", encoding="utf-8") as f:
     json.dump(new_results, f, indent=2)
  
  return new_results


# find edition info
def get_edition(id):

  # Create a session to persist headers and cookies
  session = requests.Session()
  session.headers.update({
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/141.0.0.0 Safari/537.36"
  })

  # Define the URL
  url = f"https://na2.iiivega.com/api/search-result/editions/{id}"

  # Add custom headers
  headers = {
      "authority": "na2.iiivega.com",
      "method": "GET",
      "path": f"/api/search-result/editions/{id}",
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

  # Send GET request
  response = session.get(url, headers=headers)

  # return result
  return response.json()


def parse_and_flatten_edition(edition, sep='.'):
    """
    Extracts, flattens, and processes edition metadata.

    - Extracts 'subjects', 'notes', and 'contributors' from edition['edition']
    - Flattens nested dicts (dot notation)
    - Joins list values into readable strings
    - Merges all 'notes.' fields into one 'notes' string
    - Merges all 'subjects.' fields into one 'subjects' string
    """

    data = edition.get("edition", {})

    # Step 1: Extract relevant sections
    extracted = {
        "subjects": {k: v for k, v in data.items() if k.startswith("subj")},
        "notes": {k: v for k, v in data.items() if k.startswith("note")},
        "contributors": data.get("contributors", [])
    }

    # Step 2: Flatten nested dictionaries
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

    # Step 3: Convert list values to strings
    flat = {k: ', '.join(v) if isinstance(v, list) else v for k, v in flat.items()}

    # Step 4: Merge all 'notes.' fields into one string
    notes_parts = [v for k, v in flat.items() if k.startswith("notes.")]
    flat["notes"] = " ".join(notes_parts)

    # Step 5: Merge all 'subjects.' fields into one string
    subject_parts = [v for k, v in flat.items() if k.startswith("subjects.")]
    flat["subjects"] = "; ".join(subject_parts)

    # Step 6: Remove individual 'notes.' and 'subjects.' keys
    flat = {k: v for k, v in flat.items() if not (k.startswith("notes.") or k.startswith("subjects."))}

    return flat

def enhance_results(results):
    for result in results:
        updated_materials = []  # will hold new versions of materials

        for material in result.get("materials", []):
            editions = material.get("editions", [])
            new_editions = []

            for edition in editions:
                edition_id = edition.get("id")
                if not edition_id:
                    continue  # skip invalid edition entries

                data = get_edition(edition_id)
                if not data:
                    continue  # skip if no edition data returned
                
                # optionally write raw edition data to file
                # with open("raw_edition_data_iliad.json", "a", encoding="utf-8") as f:
                #    json.dump(data, f, ensure_ascii=False, indent=2)

                parsed_data = parse_and_flatten_edition(data)

                # Combine old edition data with parsed data
                updated_edition = {**edition, **parsed_data}
                new_editions.append(updated_edition)

            # Create a new material with updated editions
            updated_material = {**material, "editions": new_editions}
            updated_materials.append(updated_material)

        # ✅ Replace the entire materials list for this result
        result["materials"] = updated_materials
    
    with open("enhanced_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results
   

### testing ###
if __name__ == "__main__":

    results = iliad_test()
    enhance_results(results)

    



       
    

  
   

