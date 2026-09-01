from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
load_dotenv()

loader = CSVLoader(file_path='products.csv', encoding='utf-8')
documents = loader.load()

print(f"Conteúdo do CSV: {len(documents)} linhas")
print(f"Primeiro produto: {documents[0].page_content}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"Chunks criados: {len(chunks)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def search_products(query: str) -> str:
    """Busca produtos no catálogo através de busca semântica.

    Útil quando o usuário pergunta sobre disponibilidade, preço, categoria
    ou características de produtos (como notebooks, teclados, smartphones, etc.).

    Args:
        query (str): Termo de busca ou pergunta sobre os produtos.

    Returns:
        str: Informações dos produtos mais relevantes encontrados.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "Nenhum produto encontrado para a pesquisa"
    
    result = [doc.page_content for doc in docs]
    return "\n\n".join(result)

system_prompt = ("Você é um assistente especializado em produtos de tecnologia e escritório. Usee a ferraenta search_products para consultar o catalogo antes de responder pergntas sobre os produtos, sempre mencione o preço quando disponivel.")

agent = create_agent(
    model="gpt-4o-mini",
    system_prompt=system_prompt,
    tools=[search_products]
)

questions = [
    "quais propdutos voce tem para melhorar a ergonomia do home office?",
    "estou montando um setup gamer, tem algo par arecomendar?",
    "preciso de perifericos sem fio, quais pcoes voces tem?",
    "qual produto mais barato do catalogo?"
]

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"Pergunta: {question}")
    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Resposta: {response['messages'][-1].content}")
    

