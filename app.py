#import logging

#import pathway as pw
#from dotenv import load_dotenv
#from pathway.xpacks.llm.question_answering import SummaryQuestionAnswerer
#from pathway.xpacks.llm.servers import QASummaryRestServer
#from pydantic import BaseModel, ConfigDict, InstanceOf

import logging
import os
import sys
import click
import pathway as pw
import yaml
from dotenv import load_dotenv
from pathway.udfs import DiskCache
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
from pathway.stdlib.indexing import BruteForceKnnFactory, HybridIndexFactory
from pathway.stdlib.indexing.bm25 import TantivyBM25Factory
from pathway.xpacks.llm import embedders, llms, parsers, splitters
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.io.fs import read
from pathway import schema_from_csv
from pathway import schema_from_dict


# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("758B3A-15AD89-9660E2-247D24-562496-V3")

logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s %(name)s %(levelname)s %(message)s",
   datefmt="%Y-%m-%d %H:%M:%S",
)


# Load environment variables (e.g., API keys) from the .env file.
load_dotenv()


# Command-line interface (CLI) function to run the app with a specified config file.
@click.command()
@click.option("--config_file", default="app.yaml", help="Config file to be used.")
def run(config_file: str = "app.yaml"):
   # Load the configuration from the YAML file.
    with open(config_file) as f:
        config = pw.load_yaml(f)
    schema = schema = schema_from_dict({
        "match_id": str,
        "player": str,
        "row": str,
        "returnable": int,
        "shallow": int,
        "deep": int,
        "very_deep": int,
        "unforced": int,
        "err_net": int,
        "err_deep": int,
        "err_wide": int,
        "err_wide_deep": int
    })

    sources = [
        read(
        path="data/charting-m-stats-ReturnDepth.csv",
        format="csv",
        with_metadata=False,
        schema=schema
    )
    ]

#    with open(config_file) as f:
    # config = pw.load_yaml(f)


    #sources = config["sources"]


   # llm = llms.OpenAIChat(model="gpt-4o-mini", cache_strategy=DiskCache())
   # llm = llms.LiteLLMChat(model="gemini/gemini-pro", cache_strategy=DiskCache())


   # Initialize the OpenAI Embedder to handle embeddings with caching enabled.
    embedder = embedders.GeminiEmbedder(model="models/text-embedding-004")




    parser = parsers.UnstructuredParser()

    index = HybridIndexFactory(
           [
               TantivyBM25Factory(),
               BruteForceKnnFactory(embedder=embedder),
           ]
       )
    llm = llms.LiteLLMChat(
       model="gemini/gemini-2.0-flash", # Choose the model you want
       api_key=os.environ["GEMINI_API_KEY"], # Read GEMINI API key from environmental variables
   )


    text_splitter = splitters.TokenCountSplitter(max_tokens=400)


   # Host and port configuration for running the server.
   # Get host from app.yaml config and port from .env file
    pathway_host = config.get("host", "0.0.0.0")
    pathway_port = int(os.environ.get("PATHWAY_PORT", 8000))


   # Initialize the vector store for storing document embeddings in memory.
   # This vector store updates the index dynamically whenever the data source changes
   # and can scale to handle over a million documents.
    doc_store = DocumentStore(
           docs=sources,
           splitter=text_splitter,
           parser=parser,
           retriever_factory=index
       )


   # Create a RAG (Retrieve and Generate) question-answering application.
    rag_app = BaseRAGQuestionAnswerer(llm=llm, indexer=doc_store)


   # Build the server to handle requests at the specified host and port.
    rag_app.build_server(host=pathway_host, port=pathway_port)


   # Run the server with caching enabled, and handle errors without shutting down.
    rag_app.run_server(with_cache=True, terminate_on_error=False)


# Entry point to execute the app if the script is run directly.
if __name__ == "__main__":
   run()
