# node.py
from typing import Dict, Any
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_compressors import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from urlextract import URLExtract
from src.state import query_type_structure, AgentState 
from src.llm_setup import llm  # custom imports from your llm setup
from src.embeddings_setup import embedding
from src.tools import duck_duck, wiki_tool
# ========== BASIC NODES ==========

def question_func(state:AgentState)->AgentState:
  return state



def query_type_func(state:AgentState)->AgentState:
  question=state['question_state']
  structured_output=llm.with_structured_output(query_type_structure)
  query_response=structured_output.invoke(question).query_refactor
  state['query_type_state']=query_response
  return state


def query_type_checker(state:AgentState)->str:
  query_type=state['query_type_state']['query_complexity']
  return query_type

def simple_query_func(state:AgentState)->AgentState:
  refactor_query=state['query_type_state']['refactored_query']
  state['simple_query_state'] = refactor_query
  return state


def complex_query_func(state:AgentState)->AgentState:
  refactor_query=state['query_type_state']['refactored_query']
  state['complex_query_state'] = refactor_query
  return state


# ========== SEARCH NODES ==========

def wiki_func(state:AgentState)->AgentState:
    refactor_query=state['query_type_state']['refactored_query']
    data=wiki_tool.invoke(refactor_query)
    #state['wiki_state']=data
    return {"wiki_state": data}

    #return state




def duck_duck_func(state:AgentState)->AgentState:
  refactor_query=state['query_type_state']['refactored_query']
  data=duck_duck.invoke(refactor_query)
  #state['duct_duck_state']=data
  return {"duct_duck_state": data}
  #return state


# ========== SIMPLE RERANKER NODE ==========

compressor = FlashrankRerank()


def simple_reranker_func(state:AgentState):
  refactor_query=state['query_type_state']['refactored_query']
  wiki_data=state['wiki_state']
  duck_duck_state=state['duct_duck_state']
  documents =  wiki_data + duck_duck_state
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  texts = text_splitter.create_documents([documents])
  retriever = Chroma.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
  )

  compressed_docs = compression_retriever.invoke(
     refactor_query
)
  #state['reranker_state']=compressed_docs
  return {'reranker_state':compressed_docs}



# ========== COMPLEX SEARCH + SCRAPING NODES ==========

def wiki_complex_func(state:AgentState):
    refactor_query=state['query_type_state']['refactored_query']
    data=wiki_tool.invoke(refactor_query)
    #state['wiki_state']=data
    return {"wiki_complex_state": data}



def duckduck_complex_func(state:AgentState):
  refactor_query=state['query_type_state']['refactored_query']
  data=duck_duck.invoke(refactor_query)
  #state['duct_duck_state']=data
  return {"ductduck_complex_state": data}
  #return state



# ========== SCRAPING NODE ==========

extractor = URLExtract()


def url_extractor(texts):
  urls = extractor.find_urls(texts)
  return urls


def scraping_func(state:AgentState):
  wiki_data = state['wiki_complex_state']
  duct_duck_data = state['ductduck_complex_state']
  if len(wiki_data) < 100:
    wiki_data = ""

  texts= wiki_data + duct_duck_data
  urls=url_extractor(texts)

  
  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()
  texts=[doc.page_content for doc in data if doc.page_content]
  texts = ", ".join(texts)
  return {"scrape_state":texts}


# ========== COMPLEX RERANKER NODE ==========

def complex_reranker_func(state:AgentState):
  refactor_query=state['query_type_state']['refactored_query']
  documents=state['scrape_state']
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  texts = text_splitter.create_documents([documents])
  retriever = Chroma.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
  )

  compressed_docs = compression_retriever.invoke(
     refactor_query
)
  #state['reranker_state']=compressed_docs
  return {'complex_reranker_state':compressed_docs}

