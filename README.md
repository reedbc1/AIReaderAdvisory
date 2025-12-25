# AI Library Search
## Overview
<b>Problem:</b>
Finding library materials on our online catalog is easy, but in many cases we don't have the exact item a patron wants at our library and need to order it from a different library. Additionally, finding recommendations for patrons based on what they like is easy, but it is often unlikely we have those recommenations available.

<b>Solution:</b>  
This program can search the library catalog and choose items most similar to what the patron is looking for. This can be much faster than manually navigating the online catalog and filters.

<b>Limitations</b>  
- Currently CLI (Command Line Interface)
- Currently only supports DVDs (movies, TV shows, documentaries)  

## Description
Uses OpenAI to provide recommended library items based on user requests. The program does the following:  

<b>Catalog Retrieval</b>
- Gets current items and item descriptions from a library catalog (only DVDs currently supported)

<b>Creating embeddings</b>
- Creates embeddings for each item to later find semantic similarity to user queries

<b>AI Responses</b>
- Prompts a user to describe what items they   are looking for
- Embeds query and searches catalog using FAISS
- Chooses final results with ChatGPT 4o 

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

## What Files Do
- catalog.py: retrieves library records from Vega catalog
- embeddings.py: creates embeddings for library records
- conversation.py: creates conversation with OpenAI API, retrieves library records based on query
- pipeline.py: integrates all three
- stateful_pipeline.py: uses state to keep track of ETL operations

## Future Releases
A future program is in development. Features will include:
- User-friendly Web Interface
- Search through all types of items
- Program's database is updated automatically and regularly
- Item availability is checked before recommendations are made to guarantee accuracy
- Links to the catalog record are available for recommended items


