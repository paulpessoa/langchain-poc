from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pprint import pprint


class PersonalInfo(BaseModel):
    name: str = Field(description="Name of the user")
    age: int = Field(description="age of the user")
    profession: str = Field(description="profession of the user")



load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.7)
agent = create_agent(model=model, response_format=PersonalInfo)

response = agent.invoke({"messages": [HumanMessage(content="Extraia os dados: Paul Pessoa, 36 anos, engenheiro de software")]})

personal_info = response['structured_response']

print(personal_info)
