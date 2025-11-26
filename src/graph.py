import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.node import (
    question_func,
    query_type_func,
    simple_query_func,
    complex_query_func,
    wiki_func,
    duck_duck_func,
    wiki_complex_func,
    duckduck_complex_func,
    scraping_func,
    simple_reranker_func,
    complex_reranker_func,
    query_type_checker
)

# ---------------- Graph Definition ----------------

graph=StateGraph(AgentState)

# existing nodes
graph.add_node('question_node', question_func)
graph.add_node('query_type_node', query_type_func)
graph.add_node('simple_query_node', simple_query_func)
graph.add_node('complex_query_node', complex_query_func)

# separate wiki/duck for simple branch
graph.add_node('wiki_simple_node', wiki_func)
graph.add_node('duck_simple_node', duck_duck_func)

# separate wiki/duck for complex branch
graph.add_node('wiki_complex_node', wiki_complex_func)     # can reuse same function
graph.add_node('duck_complex_node', duckduck_complex_func)

# scraping only on complex branch
graph.add_node('scraping_node', scraping_func)

# reranker (shared)
graph.add_node('reranker_node', simple_reranker_func)
graph.add_node('complex_reranker_node', complex_reranker_func)

# entry + routing
graph.set_entry_point('question_node')
graph.add_edge('question_node', 'query_type_node')
graph.add_conditional_edges('query_type_node', query_type_checker,
                            {"simple": "simple_query_node", "complex": "complex_query_node"})

# simple path
graph.add_edge('simple_query_node', 'wiki_simple_node')
graph.add_edge('simple_query_node', 'duck_simple_node')
graph.add_edge('wiki_simple_node', 'reranker_node')
graph.add_edge('duck_simple_node', 'reranker_node')

# complex path: wiki/duck -> scraping -> reranker
graph.add_edge('complex_query_node', 'wiki_complex_node')
graph.add_edge('complex_query_node', 'duck_complex_node')
graph.add_edge('wiki_complex_node', 'scraping_node')
graph.add_edge('duck_complex_node', 'scraping_node')
graph.add_edge('scraping_node', 'complex_reranker_node')
graph.add_edge('reranker_node', END)
graph.add_edge('complex_reranker_node', END)

workflow=graph.compile()