# WARNING: Only a test branch. Not fully functional.
# AIReaderAdvisory
## Description
Uses OpenAI embeddings and FAISS to search library items within a specific library.

## What Files Do
- catalog.py: retrieves library records from Vega catalog
- embeddings.py: creates embeddings for library records
- conversation.py: FAISS-powered search loop that returns the top 100 most similar items
- pipeline.py: integrates all three and exposes the search loop

## Running the Program
- The program is ran by calling pipeline.py
- Pipeline uses argparse. You pass arguments when runnind pipeline.py depending on what you want to do.
- Arguments:
  - --fetch "Fetch data from Vega and enrich editions"
  - --embed "Generate embeddings and FAISS index"
  - --search "Start the interactive FAISS search loop"
- For example, the following command does all three:
  - python3 pipeline.py --fetch --embed --search
- Use the --help tag to also see these descriptions in the shell.

