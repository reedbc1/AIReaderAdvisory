# AIReaderAdvisory
## Description
Uses Gemini to provide reader's advisory within a specific library.

## What Files Do
- catalog.py: retrieves library records from Vega catalog
- embeddings.py: creates embeddings for library records
- conversation.py: creates conversation with the Gemini API, retrieves library records based on query
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

## Configuration
- Set the `GEMINI_API_KEY` environment variable for embeddings and chat.
