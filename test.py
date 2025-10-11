# import dvd json
import json

### Get json from vega API ###


### Extract relevevant info from json ###
def load_json():

  # Load JSON from a local file
  with open("dvds/wrdvds.json", "r") as f:
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
print(result)

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
