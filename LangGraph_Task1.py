# Import libs
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from IPython.display import Image, display
import tiktoken


# Defining State
class State(TypedDict):
    text: str
    tokens: list[str]
    token_lengths: list[int]
    total_tokens: int


# Building nodes
# Defining Tokenizer, using GPT-4 Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")
# Tokenizer node


def tokenizer_node(state: State):
    token_ids = encoding.encode(state["text"])
    tokens = [encoding.decode([token_id]) for token_id in token_ids]
    return {"tokens": tokens}

# Defining a node that gets the number of characters in each token


def num_of_chars_in_each_token(state: State):
    token_lengths = [len(token) for token in state["tokens"]]
    return {"token_lengths": token_lengths}

# Defining a node that gets the total number of Tokens


def total_num_of_tokens(state: State):
    total_tokens = len(state["tokens"])
    return {"total_tokens": total_tokens}


# Defining builder
builder = StateGraph(State)

# Connecting nodes
builder.add_node("tokens", tokenizer_node)
builder.add_node("token_lengths", num_of_chars_in_each_token)
builder.add_node("total_tokens", total_num_of_tokens)

builder.add_edge(START, "tokens")
builder.add_edge("tokens", "token_lengths")
builder.add_edge("tokens", "total_tokens")
builder.add_edge("token_lengths", END)
builder.add_edge("total_tokens", END)

# Defining Graph
graph = builder.compile()

# Defining Result
result = graph.invoke(
    {"text": "This is my first LangGraph task., I have created three nodes, one as a tokenizer, and the following two nodes are in a parallel connection one calculates the number of characters in each token and the other one calculates the total number of tokens generated."}
)

# Showing the result
print(result)

# Displaying the Graph
display(Image(graph.get_graph().draw_mermaid_png()))
