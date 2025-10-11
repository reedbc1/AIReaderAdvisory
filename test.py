# import dvd json
import json
import requests

### Get json from vega API ###
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


# get first result for example
def get_first_result(results):
  first_result = results[0]
  return first_result

## The following functions would be called to each individual record
# get relevant json info
def filter_result(result):

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

### Get editions info from API ###

# get editions info using editionsId
def get_editions(editionsId):
  # ideas - async with rate limiting, 
  #       - generate and use different user ids
  #       - write to json file periodically to save progress made by api usage

  url = f"https://na2.iiivega.com/api/search-result/editions/{editionsId}"

  headers = {
      "authority": "na2.iiivega.com",
      "method": "GET",
      "path": f"/api/search-result/editions/{editionsId}",
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
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
  }

  # Make the GET request
  response = requests.get(url, headers=headers)

  # Check the response status
  print(f"Status: {response.status_code}")

  # Print the JSON data if available
  try:
      # load json from response
      data = response.json()

      # make json pretty
      # data_string = json.dumps(data, ensure_ascii=False, indent=2)

      # # print pretty json
      # print(data_string)

      # return json as dict
      return data
  except ValueError:
      print("Response is not valid JSON.")
      print(response.text)
      return None

# parse editions
# is this in the right format for the ai?
def parse_editions(editions):
   edition = editions.get("edition", {})

   keys = {"subjTopicalTerm", "subjGenre", "noteSummary"}
   edition_filtered = {k: v for k, v in edition.items() if k in keys}

   rename_map = {"subjTopicalTerm": "subject", "subjGenre": "genre", "noteSummary": "summary"}
   edition_renamed = {rename_map.get(k, k): v for k, v in edition_filtered.items()} 

   return edition_renamed


### get locations info using id ###
# not using currently since dvds are all available at one location
def get_locations(id, materialType):

  url = f"https://na2.iiivega.com/api/search-result/drawer/format-groups/{id}/locations?tab={materialType}"

  headers = {
      "authority": "na2.iiivega.com",
      "method": "GET",
      "path": f"/api/search-result/drawer/format-groups/{id}/locations?tab={materialType}",
      "scheme": "https",
      "accept": "application/json, text/plain, */*",
      "accept-encoding": "gzip, deflate, br, zstd",
      "accept-language": "en-US,en;q=0.9",
      "anonymous-user-id": "1c35d4e2-bdd1-49c4-97a3-9fc3eaa7d120",
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
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
  }

  response = requests.get(url, headers=headers)

  print(f"Status code: {response.status_code}")

  try:
      data = response.json()
      return data
  except ValueError:
      print("Response is not JSON:")
      print(response.text)
      return None


### add additional info to previous json ###
def add_editions_info(result, editions_info):
   result.update(editions_info)
   return result
   

### apply functions to every record in file
def apply_to_results(results):

  data = []

  for result in results:
    result_filtered = filter_result(result)
    
    editionsId = result_filtered.get("editionsId")
    editions = get_editions(editionsId)

    editions_info = parse_editions(editions)

    combined_info = add_editions_info(result_filtered, editions_info)
    data.append(combined_info)
  
  return data

   

# now we are ready for loading the data into an ai

# maintenance (future)
# for availability info: query library data from each branch
# maybe each day. have it rolling in batches of 1000, so 1000 queries spread throughout the day
# when recommending items, check availability of the item before recommending

### Testing
def test():
  # vega_api()
  results = load_json()
  first_result = get_first_result(results)
  first_result_filtered = filter_result(first_result)
  
  ## retrieve editions info
  # editionsId = first_result_json.get("editionsId")
  # editions = get_editions(editionsId)

  ## optionally save as file
  # with open("editions.json", "w", encoding="utf-8") as f:
  #   json.dump(editions, f, ensure_ascii=False, indent=2)

  # load as file
  with open("editions.json", "r", encoding="utf-8") as f:
      editions = json.load(f)

  editions_info = parse_editions(editions)

  combined_info = add_editions_info(first_result_filtered, editions_info)
  # print(combined_info)


  
  # get relevant editions info


  # # see first result
  # print(first_result_json)

  # # retrieve locations info
  # id = first_result_json.get("id")
  # materialType = first_result_json.get("materialType")
  # locations = get_locations(id, materialType)

  # # optionally save as file
  # with open("locations.json", "w", encoding="utf-8") as f:
  #   json.dump(locations, f, ensure_ascii=False, indent=2)

  data = apply_to_results(results)
  with open("wrdvds_full", "w") as f:
     json.dump(data, f, indent=2)
  print("Done!")


### testing ###
if __name__ == "__main__":
   test()

