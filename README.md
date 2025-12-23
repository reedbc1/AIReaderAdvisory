# AI Library Search
## Description
- Uses OpenAI to provide recommended library items based on user requests.
- Currently CLI (Command Line Interface)
- Currently only finds DVDs (movies, TV shows, documentaries)

## Future Releases
A future program is in development. Features will include:
- Search through all items
- Program's database is updated automatically and regularly
- User-friendly Web Interface
- Availability is checked before recommendations are made to guarantee accuracy
- Links are available to the item in the library catalog

## What Files Do
- catalog.py: retrieves library records from Vega catalog
- embeddings.py: creates embeddings for library records
- conversation.py: creates conversation with OpenAI API, retrieves library records based on query
- pipeline.py: integrates all three

## Running the Program
- The program is ran by calling pipeline.py
- Pipeline uses argparse. You pass arguments when runnind pipeline.py depending on what you want to do.
- Arguments:
  - --fetch "Fetch data from Vega and enrich editions"
  - --embed "Generate embeddings and FAISS index"
  - --chat "Start the interactive chat loop"
- For example, the following command does all three:
  - python3 pipeline.py --fetch --embed --chat
- Use the --help tag to also see these descriptions in the shell.


