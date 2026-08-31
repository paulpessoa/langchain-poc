from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool 
from langchain.agents import create_agent
from pprint import pprint

@tool
def calculate_bmi(weight: float, height: float) -> float:
    """Calcula o IMC dado o peso em kg e a altura em metros
    Args: 
        weight (float): peso em kg
        height (float): altura em metros

    Returns:
        str: O IMC calculado com 2 casas decimais e uma breve descrição do resultado com um icone de balança.
    """
    bmi = weight / (height ** 2)
    return round(bmi, 2)



from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.7)
# models_with_tools = model.bind_tools([calculate_bmi]) 
agent = create_agent(
    model=model,
    tools=[calculate_bmi],
    system_prompt="Você é um assistente de calculo de IMC. Use a tool de calculo de imc quando necessário."
)
messages = [
    # SystemMessage(
    #     content="Você é um assistente que responde tudo na lingua do P."
    # ),
    HumanMessage(content="Qual o imc de alguem com 60 quilos e 1,70 metros")
]

# response = models_with_tools.invoke(messages)
response = agent.invoke({"messages": messages})
# messages.append(response)

# for tool_call in response.tool_calls:
#     result = calculate_bmi.invoke(tool_call)
#     messages.append(result)

# final = models_with_tools.invoke(messages)

# print(response['messages'][-1].content)
# pprint(response)
# for chunk in model.stream(messages):
#     print(chunk.content, end="", flush=True)
for chunk in agent.stream({"messages": messages}, stream_mode="values" ):
    latest_message = chunk.get("messages")[-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"Humano: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"AI: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
                  
            