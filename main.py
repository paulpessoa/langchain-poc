from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.7)
agent = create_agent(model=model)

messages = [HumanMessage(content="Oi, me chamo Paul")]

response = agent.invoke({"messages": messages})
messages.append(response['messages'][-1])
print(response['messages'][-1].content)

messages.append(HumanMessage(content="Qual o meu  nome? "))
response = agent.invoke({"messages": messages})
messages.append(response['messages'][-1])
print(response['messages'][-1].content)

