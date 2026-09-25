from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data = PyPDFLoader("document_loader/Deeplearning.pdf")
docs = data.load()


splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [("system","you are a AI that summarizes the text"),
    ("human", "{data}")]
)


model = ChatMistralAI(model='ministral-3b-2512')
prompt = template. format_messages(data = docs[0].page_content)
response = model.invoke(prompt)
print(response.content)