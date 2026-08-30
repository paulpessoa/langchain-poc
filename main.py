from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.7)


messages = [
    SystemMessage(
        content="Você é um assistente que responde tudo na lingua do P."
    ),
    HumanMessage(content="O que é o langchain framework?")
]


# response = model.invoke(messages)
# print(response.content)

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
