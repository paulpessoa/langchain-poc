from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool 

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
models_with_tools = model.bind_tools([calculate_bmi]) 

messages = [
    # SystemMessage(
    #     content="Você é um assistente que responde tudo na lingua do P."
    # ),
    HumanMessage(content="Qual o imc de alguem com 60 quilos e 1,70 metros")
]

response = models_with_tools.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    result = calculate_bmi.invoke(tool_call)
    messages.append(result)

final = models_with_tools.invoke(messages)

print(final.content)

# for chunk in model.stream(messages):
#     print(chunk.content, end="", flush=True)
