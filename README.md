Вызов функций с помощью Gemini API


Вызов функций позволяет подключать модели к внешним инструментам и API. Вместо генерации текстовых ответов модель понимает, когда вызывать определенные функции, и предоставляет необходимые параметры для выполнения реальных действий. Это позволяет модели выступать в качестве моста между естественным языком и реальными действиями и данными. Вызов функций имеет 3 основных варианта использования:

Расширяйте знания: получайте доступ к информации из внешних источников, таких как базы данных, API и базы знаний.
Расширение возможностей: используйте внешние инструменты для выполнения вычислений и расширения ограничений модели, например, с помощью калькулятора или создания диаграмм.
Выполнение действий: взаимодействие с внешними системами с помощью API, например планирование встреч, создание счетов, отправка электронных писем или управление устройствами умного дома.
Получить прогноз погоды Расписание встреч Создать диаграмму

Питон
JavaScript
ОТДЫХ

 from google import genai
 from google.genai import types

 # Define the function declaration for the model
 schedule_meeting_function = {
     "name": "schedule_meeting",
     "description": "Schedules a meeting with specified attendees at a given time and date.",
     "parameters": {
         "type": "object",
         "properties": {
             "attendees": {
                 "type": "array",
                 "items": {"type": "string"},
                 "description": "List of people attending the meeting.",
             },
             "date": {
                 "type": "string",
                 "description": "Date of the meeting (e.g., '2024-07-29')",
             },
             "time": {
                 "type": "string",
                 "description": "Time of the meeting (e.g., '15:00')",
             },
             "topic": {
                 "type": "string",
                 "description": "The subject or topic of the meeting.",
             },
         },
         "required": ["attendees", "date", "time", "topic"],
     },
 }

 # Configure the client and tools
 client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
 tools = types.Tool(function_declarations=[schedule_meeting_function])
 config = types.GenerateContentConfig(tools=[tools])

 # Send request with function declarations
 response = client.models.generate_content(
     model="gemini-2.0-flash",
     contents="Schedule a meeting with Bob and Alice for 03/14/2025 at 10:00 AM about the Q3 planning.",
     config=config,
 )

 # Check for a function call
 if response.candidates[0].content.parts[0].function_call:
     function_call = response.candidates[0].content.parts[0].function_call
     print(f"Function to call: {function_call.name}")
     print(f"Arguments: {function_call.args}")
     #  In a real app, you would call your function here:
     #  result = schedule_meeting(**function_call.args)
 else:
     print("No function call found in the response.")
     print(response.text)
Как работает вызов функций 

Обзор вызова функций

Вызов функции подразумевает структурированное взаимодействие между вашим приложением, моделью и внешними функциями. Вот разбивка процесса:

Определите декларацию функции: Определите декларацию функции в коде вашего приложения. Декларации функций описывают имя функции, параметры и цель для модели.
Вызов LLM с декларациями функций: отправка приглашения пользователя вместе с декларацией(ями) функций в модель. Она анализирует запрос и определяет, будет ли полезен вызов функции. Если да, она отвечает структурированным объектом JSON.
Выполнение кода функции (ваша ответственность): Модель не выполняет функцию сама по себе. Это ответственность вашего приложения — обрабатывать ответ и проверять вызов функции, если
Да : извлеките имя и аргументы функции и выполните соответствующую функцию в вашем приложении.
Нет: Модель предоставила прямой текстовый ответ на подсказку (этот поток менее подчеркнут в примере, но является возможным результатом).
Создать удобный для пользователя ответ: Если функция была выполнена, захватить результат и отправить его обратно в модель в последующем повороте диалога. Она будет использовать результат для генерации окончательного удобного для пользователя ответа, который включает информацию из вызова функции.
Этот процесс может повторяться в течение нескольких ходов, что позволяет осуществлять сложные взаимодействия и рабочие процессы. Модель также поддерживает вызов нескольких функций в одном ходе ( параллельный вызов функций ) и последовательно ( композиционный вызов функций ).

Шаг 1: Определение объявления функции

Определите функцию и ее объявление в коде вашего приложения, которое позволяет пользователям устанавливать значения света и делать запрос API. Эта функция может вызывать внешние службы или API.

Питон
JavaScript

from google.genai import types

# Define a function that the model can call to control smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {"brightness": brightness, "colorTemperature": color_temp}

Шаг 2: Вызов модели с объявлениями функций

После того, как вы определили объявления функций, вы можете предложить модели использовать функцию. Она анализирует приглашение и объявления функций и решает, ответить напрямую или вызвать функцию. Если функция вызывается, объект ответа будет содержать предложение вызова функции.

Питон
JavaScript

from google import genai

# Generation Config with Function Declaration
tools = types.Tool(function_declarations=[set_light_values_declaration])
config = types.GenerateContentConfig(tools=[tools])

# Configure the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Turn the lights down to a romantic level")]
    )
]

# Send request with function declarations
response = client.models.generate_content(
    model="gemini-2.0-flash", config=config, contents=contents
)

print(response.candidates[0].content.parts[0].function_call)
Затем модель возвращает объект functionCall в схеме, совместимой с OpenAPI, определяющей, как вызвать одну или несколько объявленных функций, чтобы ответить на вопрос пользователя.

Питон
JavaScript

id=None args={'color_temp': 'warm', 'brightness': 25} name='set_light_values'
Шаг 3: Выполнить код функции set_light_values

Извлеките сведения о вызове функции из ответа модели, проанализируйте аргументы и выполните функцию set_light_values ​​в нашем коде.

Питон
JavaScript

# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "set_light_values":
    result = set_light_values(**tool_call.args)
    print(f"Function execution result: {result}")
Шаг 4: Создайте удобный для пользователя ответ с результатом функции и вызовите модель еще раз.

Наконец, отправьте результат выполнения функции обратно в модель, чтобы она могла включить эту информацию в свой окончательный ответ пользователю.

Питон
JavaScript

# Create a function response part
function_response_part = types.Part.from_function_response(
    name=tool_call.name,
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(response.candidates[0].content) # Append the content from the model's response.
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

final_response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=config,
    contents=contents,
)

print(final_response.text)
Это завершает поток вызова функции. Модель успешно использовала функцию set_light_values ​​для выполнения действия запроса пользователя.

Декларации функций

При реализации вызова функции в приглашении вы создаете объект tools , который содержит одно или несколько function declarations . Вы определяете функции с помощью JSON, в частности, с выбранным подмножеством формата схемы OpenAPI . Одно объявление функции может включать следующие параметры:

name (string): Уникальное имя для функции ( get_weather_forecast , send_email ). Используйте описательные имена без пробелов и специальных символов (используйте подчеркивания или camelCase).
description (string): Четкое и подробное объяснение цели и возможностей функции. Это важно для модели, чтобы понять, когда использовать функцию. Будьте конкретны и приведите примеры, если это полезно («Находит кинотеатры на основе местоположения и, возможно, названия фильма, который в данный момент идет в кинотеатрах»).
parameters (объект): определяет входные параметры, ожидаемые функцией.
type (строка): определяет общий тип данных, например, object .
properties (объект): Перечисляет отдельные параметры, каждый из которых содержит:
type (строка): тип данных параметра, например string , integer , boolean, array .
description (string): Описание назначения и формата параметра. Предоставьте примеры и ограничения («Город и штат, например, «Сан-Франциско, Калифорния» или почтовый индекс, например, «95616».)
enum (массив, необязательно): Если значения параметров из фиксированного набора, используйте "enum" для перечисления допустимых значений вместо того, чтобы просто описывать их в описании. Это повышает точность ("enum": ["daylight", "cool", "warm"]).
required (массив): массив строк, в котором перечислены имена параметров, обязательные для работы функции.
Параллельный вызов функций

Помимо вызова функции с одним ходом, вы также можете вызывать несколько функций одновременно. Параллельный вызов функций позволяет выполнять несколько функций одновременно и используется, когда функции не зависят друг от друга. Это полезно в таких сценариях, как сбор данных из нескольких независимых источников, например, получение данных о клиентах из разных баз данных или проверка уровней запасов на разных складах или выполнение нескольких действий, например, переоборудование квартиры в дискотеку.

Питон
JavaScript

power_disco_ball = {
    "name": "power_disco_ball",
    "description": "Powers the spinning disco ball.",
    "parameters": {
        "type": "object",
        "properties": {
            "power": {
                "type": "boolean",
                "description": "Whether to turn the disco ball on or off.",
            }
        },
        "required": ["power"],
    },
}

start_music = {
    "name": "start_music",
    "description": "Play some music matching the specified parameters.",
    "parameters": {
        "type": "object",
        "properties": {
            "energetic": {
                "type": "boolean",
                "description": "Whether the music is energetic or not.",
            },
            "loud": {
                "type": "boolean",
                "description": "Whether the music is loud or not.",
            },
        },
        "required": ["energetic", "loud"],
    },
}

dim_lights = {
    "name": "dim_lights",
    "description": "Dim the lights.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "number",
                "description": "The brightness of the lights, 0.0 is off, 1.0 is full.",
            }
        },
        "required": ["brightness"],
    },
}
Вызовите модель с инструкцией, которая может использовать все указанные инструменты. В этом примере используется tool_config . Чтобы узнать больше, вы можете прочитать о настройке вызова функций .

Питон
JavaScript

from google import genai
from google.genai import types

# Set up function declarations
house_tools = [
    types.Tool(function_declarations=[power_disco_ball, start_music, dim_lights])
]

config = {
    "tools": house_tools,
    "automatic_function_calling": {"disable": True},
    # Force the model to call 'any' function, instead of chatting.
    "tool_config": {"function_calling_config": {"mode": "any"}},
}

# Configure the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat = client.chats.create(model="gemini-2.0-flash", config=config)
response = chat.send_message("Turn this place into a party!")

# Print out each of the function calls requested from this single call
print("Example 1: Forced function calling")
for fn in response.function_calls:
    args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
    print(f"{fn.name}({args})")
Каждый из напечатанных результатов отражает один вызов функции, запрошенный моделью. Чтобы отправить результаты обратно, включите ответы в том же порядке, в котором они были запрошены.

Python SDK поддерживает функцию, называемую автоматическим вызовом функции , которая преобразует функцию Python в объявления, обрабатывает выполнение вызова функции и цикл ответа для вас. Ниже приведен пример для нашего варианта использования disco.

Примечание: Автоматический вызов функций на данный момент является единственной функцией Python SDK.
Питон

from google import genai
from google.genai import types

# Actual implementation functions
def power_disco_ball_impl(power: bool) -> dict:
    """Powers the spinning disco ball.

    Args:
        power: Whether to turn the disco ball on or off.

    Returns:
        A status dictionary indicating the current state.
    """
    return {"status": f"Disco ball powered {'on' if power else 'off'}"}

def start_music_impl(energetic: bool, loud: bool) -> dict:
    """Play some music matching the specified parameters.

    Args:
        energetic: Whether the music is energetic or not.
        loud: Whether the music is loud or not.

    Returns:
        A dictionary containing the music settings.
    """
    music_type = "energetic" if energetic else "chill"
    volume = "loud" if loud else "quiet"
    return {"music_type": music_type, "volume": volume}

def dim_lights_impl(brightness: float) -> dict:
    """Dim the lights.

    Args:
        brightness: The brightness of the lights, 0.0 is off, 1.0 is full.

    Returns:
        A dictionary containing the new brightness setting.
    """
    return {"brightness": brightness}

config = {
    "tools": [power_disco_ball_impl, start_music_impl, dim_lights_impl],
}

chat = client.chats.create(model="gemini-2.0-flash", config=config)
response = chat.send_message("Do everything you need to this place into party!")

print("\nExample 2: Automatic function calling")
print(response.text)
# I've turned on the disco ball, started playing loud and energetic music, and dimmed the lights to 50% brightness. Let's get this party started!
Вызов композиционной функции

Gemini 2.0 поддерживает композиционный вызов функций, что означает, что модель может объединять несколько вызовов функций вместе. Например, чтобы ответить «Получить температуру в моем текущем местоположении», API Gemini может вызвать как функцию get_current_location() , так и функцию get_weather() , которая принимает местоположение в качестве параметра.

Примечание: Вызов композиционных функций в настоящее время доступен только в Live API . Объявление функции run() , которое обрабатывает асинхронную настройку веб-сокета, опущено для краткости.
Питон
JavaScript

# Light control schemas
turn_on_the_lights_schema = {'name': 'turn_on_the_lights'}
turn_off_the_lights_schema = {'name': 'turn_off_the_lights'}

prompt = """
  Hey, can you write run some python code to turn on the lights, wait 10s and then turn off the lights?
  """

tools = [
    {'code_execution': {}},
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]}
]

await run(prompt, tools=tools, modality="AUDIO")
Режимы вызова функций

API Gemini позволяет вам контролировать, как модель использует предоставленные инструменты (декларации функций). В частности, вы можете установить режим в function_calling_config .

AUTO (Default) : модель решает, генерировать ли ответ на естественном языке или предлагать вызов функции на основе подсказки и контекста. Это наиболее гибкий режим, рекомендуемый для большинства сценариев.
ANY : Модель ограничена тем, чтобы всегда предсказывать вызов функции и гарантировать соответствие схеме функции. Если allowed_function_names не указано, модель может выбирать из любого из предоставленных объявлений функций. Если allowed_function_names указано в виде списка, модель может выбирать только из функций в этом списке. Используйте этот режим, когда требуется вызов функции в ответ на каждый запрос (если применимо).
NONE : модели запрещено делать вызовы функций. Это эквивалентно отправке запроса без каких-либо объявлений функций. Используйте это, чтобы временно отключить вызов функций, не удаляя определения инструментов.
Питон
JavaScript

from google.genai import types

# Configure function calling mode
tool_config = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["get_current_temperature"]
    )
)

# Create the generation config
config = types.GenerateContentConfig(
    temperature=0,
    tools=[tools],  # not defined here.
    tool_config=tool_config,
)
Автоматический вызов функций (только Python)

При использовании Python SDK вы можете предоставлять функции Python напрямую как инструменты. SDK автоматически преобразует функцию Python в объявления, обрабатывает выполнение вызова функции и цикл ответа для вас. Затем Python SDK автоматически:

Обнаруживает ответы на вызовы функций из модели.
Вызовите соответствующую функцию Python в вашем коде.
Отправляет ответ функции обратно в модель.
Возвращает окончательный текстовый ответ модели.
Чтобы использовать это, определите свою функцию с подсказками типов и строкой документации, а затем передайте саму функцию (не декларацию JSON) в качестве инструмента:

Питон

from google import genai
from google.genai import types

# Define the function with type hints and docstring
def get_current_temperature(location: str) -> dict:
    """Gets the current temperature for a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA

    Returns:
        A dictionary containing the temperature and unit.
    """
    # ... (implementation) ...
    return {"temperature": 25, "unit": "Celsius"}

# Configure the client and model
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # Replace with your actual API key setup
config = types.GenerateContentConfig(
    tools=[get_current_temperature]
)  # Pass the function itself

# Make the request
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What's the temperature in Boston?",
    config=config,
)

print(response.text)  # The SDK handles the function call and returns the final text
Вы можете отключить автоматический вызов функций с помощью:

Питон

# To disable automatic function calling:
config = types.GenerateContentConfig(
    tools=[get_current_temperature],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
)
Декларация схемы автоматической функции

Автоматическое извлечение схемы из функций Python работает не во всех случаях. Например: оно не обрабатывает случаи, когда вы описываете поля вложенного словаря-объекта. API может описывать любой из следующих типов:

Питон

AllowedType = (int | float | bool | str | list['AllowedType'] | dict[str, AllowedType])
Чтобы увидеть, как выглядит выведенная схема, вы можете преобразовать ее с помощью from_callable :

Питон

def multiply(a: float, b: float):
    """Returns a * b."""
    return a * b

fn_decl = types.FunctionDeclaration.from_callable(callable=multiply, client=client)

# to_json_dict() provides a clean JSON representation.
print(fn_decl.to_json_dict())
Многофункциональное использование: объединение собственных инструментов с вызовом функций

С Gemini 2.0 вы можете включить несколько инструментов, объединяющих собственные инструменты с вызовом функций одновременно. Вот пример, который включает два инструмента, Grounding with Google Search и code execution , в запросе с использованием Live API .

Примечание: Использование нескольких инструментов в настоящее время доступно только в Live API . Объявление функции run() , которая обрабатывает асинхронную настройку веб-сокета, опущено для краткости.
Питон
JavaScript

# Multiple tasks example - combining lights, code execution, and search
prompt = """
  Hey, I need you to do three things for me.

    1.  Turn on the lights.
    2.  Then compute the largest prime palindrome under 100000.
    3.  Then use Google Search to look up information about the largest earthquake in California the week of Dec 5 2024.

  Thanks!
  """

tools = [
    {'google_search': {}},
    {'code_execution': {}},
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]} # not defined here.
]

# Execute the prompt with specified tools in audio modality
await run(prompt, tools=tools, modality="AUDIO")
Разработчики Python могут опробовать это в блокноте Live API Tool Use .

Модель контекстного протокола (MCP)

Model Context Protocol (MCP) — открытый стандарт для подключения приложений ИИ к внешним инструментам и данным. MCP предоставляет общий протокол для моделей для доступа к контексту, такому как функции (инструменты), источники данных (ресурсы) или предопределенные подсказки.

Gemini SDK имеют встроенную поддержку MCP, что сокращает шаблонный код и предлагает автоматический вызов инструментов для инструментов MCP. Когда модель генерирует вызов инструмента MCP, клиентский SDK Python и JavaScript может автоматически выполнить инструмент MCP и отправить ответ обратно модели в последующем запросе, продолжая этот цикл до тех пор, пока модель не перестанет делать вызовы инструментов.

Здесь вы можете найти пример использования локального сервера MCP с Gemini и mcp SDK.

Питон
JavaScript
Убедитесь, что на выбранной вами платформе установлена ​​последняя версия mcp SDK .


pip install mcp
Примечание: Python поддерживает автоматический вызов инструмента, передавая ClientSession в параметры tools . Если вы хотите отключить его, вы можете предоставить automatic_function_calling с отключенным True .

import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",  # Executable
    args=["-y", "@philschmid/weather-mcp"],  # MCP Server
    env=None,  # Optional environment variables
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Prompt to get the weather for the current day in London.
            prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%d')}?"
            # Initialize the connection between client and server
            await session.initialize()
            # Send request to the model with MCP function declarations
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # uses the session, will automatically call the tool
                    # Uncomment if you **don't** want the sdk to automatically call the tool
                    # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    #     disable=True
                    # ),
                ),
            )
            print(response.text)

# Start the asyncio event loop and run the main function
asyncio.run(run())
Ограничения при встроенной поддержке MCP

Встроенная поддержка MCP является экспериментальной функцией в наших SDK и имеет следующие ограничения:

Поддерживаются только инструменты, а не ресурсы или подсказки.
Он доступен для Python и JavaScript/TypeScript SDK.
В будущих версиях могут произойти критические изменения.
Ручная интеграция серверов MCP всегда возможна, если это ограничивает ваши возможности.

Поддерживаемые модели

Экспериментальные модели не включены. Их возможности вы можете найти на странице обзора моделей .

Модель	Вызов функции	Параллельный вызов функций	Вызов композиционной функции
(Только API в реальном времени)
Близнецы 2.0 Флэш	✔️	✔️	✔️
Gemini 2.0 Flash-Lite	Х	Х	Х
Близнецы 1.5 Флэш	✔️	✔️	✔️
Близнецы 1.5 Про	✔️	✔️	✔️
Лучшие практики

Описания функций и параметров: будьте предельно ясны и конкретны в своих описаниях. Модель опирается на них, чтобы выбрать правильную функцию и предоставить соответствующие аргументы.
Именование: используйте описательные имена функций (без пробелов, точек и тире).
Строгая типизация: используйте определенные типы (целое число, строка, перечисление) для параметров, чтобы уменьшить количество ошибок. Если параметр имеет ограниченный набор допустимых значений, используйте перечисление.
Выбор инструмента: Хотя модель может использовать произвольное количество инструментов, предоставление слишком большого количества может увеличить риск выбора неправильного или неоптимального инструмента. Для достижения наилучших результатов стремитесь предоставлять только соответствующие инструменты для контекста или задачи, в идеале сохраняя активный набор максимум в 10-20. Рассмотрите динамический выбор инструментов на основе контекста разговора, если у вас большое общее количество инструментов.
Оперативное проектирование:
Предоставьте контекст: расскажите модели о ее роли (например, «Вы полезный помощник по прогнозу погоды»).
Дайте инструкции: укажите, как и когда использовать функции (например, «Не угадывайте даты; всегда используйте будущую дату для прогнозов»).
Поощряйте уточнения: попросите модель задавать уточняющие вопросы при необходимости.
Температура: используйте низкую температуру (например, 0) для более детерминированных и надежных вызовов функций.
Проверка: если вызов функции имеет существенные последствия (например, размещение заказа), проверьте вызов у ​​пользователя перед его выполнением.
Обработка ошибок : реализуйте надежную обработку ошибок в своих функциях, чтобы изящно обрабатывать неожиданные входные данные или сбои API. Возвращайте информативные сообщения об ошибках, которые модель может использовать для генерации полезных ответов пользователю.
Безопасность: Помните о безопасности при вызове внешних API. Используйте соответствующие механизмы аутентификации и авторизации. Избегайте раскрытия конфиденциальных данных при вызовах функций.
Ограничения по токенам: Описания функций и параметры учитываются в вашем лимите входных токенов. Если вы достигаете лимитов по токенам, рассмотрите возможность ограничения количества функций или длины описаний, разбейте сложные задачи на более мелкие, более целевые наборы функций.
Примечания и ограничения

Поддерживается только подмножество схемы OpenAPI .
Поддерживаемые типы параметров в Python ограничены.
Автоматический вызов функций доступен только в Python SDK.
Генерация текста


API Gemini может генерировать текстовый вывод из различных входных данных, включая текст, изображения, видео и аудио, используя модели Gemini.

Вот простой пример, требующий ввода одного текста:

Питон
JavaScript
Идти
Ещё

from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)
print(response.text)
Системные инструкции и конфигурации

Вы можете управлять поведением моделей Gemini с помощью системных инструкций. Для этого передайте объект GenerateContentConfig .

Питон
JavaScript
Идти
Ещё

from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko."),
    contents="Hello there"
)

print(response.text)
Объект GenerateContentConfig также позволяет переопределять параметры генерации по умолчанию, такие как температура .

Питон
JavaScript
Идти
Ещё

from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1
    )
)
print(response.text)
Полный список настраиваемых параметров и их описания см. в разделе GenerateContentConfig в нашем справочнике по API.

Мультимодальные входы

API Gemini поддерживает мультимодальные входы, позволяя вам комбинировать текст с медиафайлами. Следующий пример демонстрирует предоставление изображения:

Питон
JavaScript
Идти
Ещё

from PIL import Image
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, "Tell me about this instrument"]
)
print(response.text)
Для альтернативных методов предоставления изображений и более продвинутой обработки изображений см. наше руководство по пониманию изображений . API также поддерживает ввод и понимание документов , видео и аудио .

Потоковые ответы

По умолчанию модель возвращает ответ только после завершения всего процесса генерации.

Для более плавного взаимодействия используйте потоковую передачу, чтобы получать экземпляры GenerateContentResponse постепенно по мере их генерации.

Питон
JavaScript
Идти
Ещё

from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"]
)
for chunk in response:
    print(chunk.text, end="")
Многократные беседы (чат)

Наши SDK предоставляют функционал для сбора нескольких раундов запросов и ответов в чат, предоставляя вам простой способ отслеживать историю разговоров.

Примечание: Функциональность чата реализована только как часть SDK. За кулисами он по-прежнему использует API generateContent . Для многооборотных разговоров полная история разговоров отправляется в модель с каждым последующим ходом.
Питон
JavaScript
Идти
Ещё

from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
chat = client.chats.create(model="gemini-2.0-flash")

response = chat.send_message("I have 2 dogs in my house.")
print(response.text)

response = chat.send_message("How many paws are in my house?")
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}',end=": ")
    print(message.parts[0].text)
Потоковую передачу также можно использовать для многопоточных разговоров.

Питон
JavaScript
Идти
Ещё

from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
chat = client.chats.create(model="gemini-2.0-flash")

response = chat.send_message_stream("I have 2 dogs in my house.")
for chunk in response:
    print(chunk.text, end="")

response = chat.send_message_stream("How many paws are in my house?")
for chunk in response:
    print(chunk.text, end="")

for message in chat.get_history():
    print(f'role - {message.role}', end=": ")
    print(message.parts[0].text)
Поддерживаемые модели

Все модели семейства Gemini поддерживают генерацию текста. Чтобы узнать больше о моделях и их возможностях, посетите страницу Модели .

Лучшие практики

Полезные советы

Для создания простого текста часто бывает достаточно подсказки с нуля , без необходимости в примерах, системных инструкциях или специальном форматировании.

Для более индивидуальных результатов:

Используйте системные инструкции для управления моделью.
Предоставьте несколько примеров входов и выходов для руководства моделью. Это часто называют подсказкой с небольшим количеством выстрелов .
Рассмотрите возможность тонкой настройки для расширенных вариантов использования.
Дополнительные советы можно найти в нашем руководстве по быстрому проектированию .

Структурированный вывод

В некоторых случаях вам может понадобиться структурированный вывод, такой как JSON. Обратитесь к нашему руководству по структурированному выводу , чтобы узнать, как это сделать.

Что дальше?

Попробуйте API Gemini, чтобы начать работу с Colab .
Изучите возможности Gemini по распознаванию изображений , видео , аудио и документов .
Узнайте о стратегиях мультимодального запроса файлов .

## Автономный агент

Скрипт `agent.py` демонстрирует использование модели gemini-2.0-flash для построения плана и выполнения команд в рабочей директории. Перед запуском установите зависимости `pip install -r requirements.txt` и задайте переменную окружения `GEMINI` (поддерживается также `GEMINI_API_KEY`).

Агент строит подробный план, а после каждой выполненной команды пересылает историю модели, чтобы скорректировать оставшиеся шаги. Таким образом, план можно динамически изменять или пропускать пункты.

Пример запуска:

```
python agent.py "создать файл example.txt и вывести его содержимое"
```

Для тестирования можно попробовать более сложную цель:

```
python agent.py "Создать небольшую игру на C++ с компиляцией с помощью cmake"
```

