from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

DB_PATH = "Chroma-DB"
COLLECTION_NAME = "dna_book"
embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
    },
)

llm = ChatMistralAI(model="ministral-3b-2512")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
        Use ONLY the provided context to answer the question.

        If the answer is not present in the context,
        say: "I could not find the answer in the document."
        """,
    ),
    (
        "human",
        """context:

        {context}

        question:
        {question}
        """,
    ),
])

print("press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
    })
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
