"""
LangChain Modern Overview (v0.2+ style)

Install:
pip install -U langchain langchain-openai langchain-core
pip install -U langchain-community langchain-pinecone
pip install -U langchain-experimental python-dotenv pinecone-client
"""

# ============================================================
# 1. Environment Setup
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ============================================================
# 2. ChatOpenAI (Modern Way)
# ============================================================

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

response = llm.invoke("Explain large language models in one sentence.")
print("\n--- ChatOpenAI Response ---")
print(response.content)

# ============================================================
# 3. Prompt Templates + LCEL
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "You are an expert AI engineer. Explain {concept} clearly."
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"concept": "autoencoders"})
print("\n--- LCEL Chain Output ---")
print(result)

# ============================================================
# 4. RunnableSequence
# ============================================================

from langchain_core.runnables import RunnableSequence

simplify_prompt = ChatPromptTemplate.from_template(
    "Explain the following like I am 5 years old: {text}"
)

chain_one = prompt | llm | parser
chain_two = simplify_prompt | llm | parser

overall_chain = RunnableSequence(
    first=chain_one,
    last=chain_two
)

simplified_output = overall_chain.invoke({"concept": "neural networks"})
print("\n--- RunnableSequence Output ---")
print(simplified_output)

# ============================================================
# 5. Embeddings (Latest OpenAI)
# ============================================================

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector = embeddings.embed_query("LangChain modern embeddings example")

print("\n--- Embedding Vector Length ---")
print(len(vector))

# ============================================================
# 6. Pinecone (v3 Client)
# ============================================================

if PINECONE_API_KEY:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "langchain-modern-demo"
    # ⚠️ Make sure this index exists in Pinecone dashboard

    vectorstore = PineconeVectorStore.from_texts(
        texts=["LangChain enables building LLM applications easily."],
        embedding=embeddings,
        index_name=index_name
    )

    results = vectorstore.similarity_search("What is LangChain?")

    print("\n--- Pinecone Similarity Search Result ---")
    print(results[0].page_content)

else:
    print("\nPINECONE_API_KEY not found. Skipping Pinecone section.")

# ============================================================
# 7. Modern ReAct Agent
# ============================================================

from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

python_tool = PythonREPLTool()

agent_prompt = PromptTemplate.from_template(
    "Answer the question. You may use tools if necessary. Question: {input}"
)

agent = create_react_agent(
    llm=llm,
    tools=[python_tool],
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[python_tool],
    verbose=True
)

print("\n--- Agent Execution ---")
agent_executor.invoke({"input": "Solve 2*x^2 + 3*x - 2"})

print("\nExecution complete.")