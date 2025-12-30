# AI Library Search
## Overview
<b>Problem:</b> It is sometimes difficult to connect customers with physical library items on the same day.
- Making recommendations for library items depend on staff knowledge or catalog/internet searches.
- Customers are commonly looking for a particular item, but we often have to get it from another branch. 
- The library has online reader advisory tools, but they are not able to filter by availability at a single branch.

<b>Solution:</b> 
- This program uses AI to search a specific library’s collection for available items related to a customer’s query, improving customer convenience while increasing library circulation.

<b>Current Limitations</b>  
- CLI (Command Line Interface) for prototyping
- Only supports DVDs (movies, TV shows, documentaries)  

## Description
Uses OpenAI to provide recommended library items based on user requests. The program does the following:  

<b>Catalog Retrieval</b>
- Gets current items and item descriptions from a library catalog

<b>Creating embeddings</b>
- Creates embeddings for each item to find semantic similarity to user queries

<b>AI Responses</b>
- Prompts a user to describe what items they are looking for
- Embeds query and searches catalog using FAISS (Facebook AI Similarity Search)
- Chooses final results with ChatGPT 4o and explains relevance to query

## Running the Program
- The program runs by calling pipeline.py
- Pipeline uses argparse. You pass arguments when running pipeline.py depending on what you want to do.
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


