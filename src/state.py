from typing import TypedDict, Optional, List, Annotated
import operator
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class AgentState(TypedDict):
  question_state: str
  query_type_state: dict
  simple_query_state: Optional[str]
  complex_query_state: Optional[str]
  wiki_state:str
  duct_duck_state:str
  wiki_complex_state:str
  ductduck_complex_state:str
  scrape_state:Annotated[str, operator.add]
  reranker_state:Annotated[List[Document], operator.add]
  complex_reranker_state: List[Document]



class query_type_structure(BaseModel):
  query_refactor: dict = Field(description="Please refactor the original user query. also return the query is simple or complex. Again return two things only.First one is refactor query. Secondly return simple if original query is simple else complex is original query is complex")
  #query_type_str: str = Field(description="Please see the user question and just return its a simple question or complex question. just return simple or complex")

