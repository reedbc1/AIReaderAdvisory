# AIReaderAdvisory
## Description
Uses OpenAI to provide reader's advisory within a specific library.

## What Files Do
- catalog.py: retrieves library records from Vega catalog
- embeddings.py: creates embeddings for library records (from wr_enhanced.json)
- conversation.py: interactive chat loop that reads the generated FAISS index
- pipeline.py: orchestrates fetch → enrich → embed → chat

## Running the Program
- The program is ran by calling pipeline.py
- Pipeline uses argparse. You pass arguments when runnind pipeline.py depending on what you want to do.
- Arguments:
  - --fetch "Fetch data from Vega and enrich editions into wr_enhanced.json"
  - --embed "Generate embeddings and FAISS index from wr_enhanced.json"
  - --chat "Start the interactive chat loop"
  - --run-dir "Directory to store / read artifacts (defaults to data/wr_run)"
- For example, the following command does all three:
  - python3 pipeline.py --fetch --embed --chat --run-dir data/wr_run
- Use the --help tag to also see these descriptions in the shell.

