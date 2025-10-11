# import dvd json
import json

### Get json from vega API ###

import requests

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


### Extract relevevant info from json ###
def load_json():

  # Load JSON from a local file
  with open("dvds/wrdvds.json", "r", encoding="utf-8") as f:
    data = json.load(f)

  # parse dvd json for relevant info
  results = data.get("data", [{}])
  return results


# first result for example
def get_first_result(results):
  first_result = results[0]
  return first_result


def extract_from_json(result):

  # build relevant json info
  id = result.get("id")
  title = result.get("title")
  publicationDate = result.get("publicationDate")
  # get author if record is book

  # make materialTabs for reference
  materialTabs = result.get("materialTabs", [{}])[0]

  # using materialTabs
  materialType = materialTabs.get("name")
  callNumber = materialTabs.get("callNumber")

  # editions_id for MARC record
  editionsId = materialTabs.get("editions")[0] \
    .get("id")

  return {
      "id": id,
      "title": title,
      "publicationDate": publicationDate,
      "materialType": materialType,
      "callNumber": callNumber,
      "editionsId": editionsId
  }


### Testing
def test():
  results = load_json()
  first_result = get_first_result(results)
  first_result_json = extract_from_json(first_result)
  return first_result_json


result = test()

vega_api()

### Get MARC record info from API ###

# retrieve editionsId
id = result.get("editionsId")

### Enhance json file

# query marc record info to extract necessary data
## description, subjects
## anything else that is missing and important

# add marc record info to parsed dvd json info

# now we are ready for loading the data into an ai

# maintenance (future)
# for availability info: query library data from each branch
# maybe each day. have it rolling in batches of 1000, so 1000 queries spread throughout the day
# when recommending items, check availability of the item before recommending
