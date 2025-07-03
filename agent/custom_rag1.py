import json
from typing import List

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.utils.json import parse_json_markdown
from langchain_ollama import ChatOllama

# 模型
llm = ChatOllama(
    model="deepseek-r1:14b",
    temperature=0,
    base_url="http://21.6.90.166.devcloud.woa.com:11434"
)

# 定义角色模板
system_template = "你是一个客服助手，请使用所有能用的工具，查询回用户的问题。然后给出答案"
human_template = """
尽可能的帮助用户回答任何问题。

您可以使用以下工具来帮忙解决问题，如果已经知道了答案，也可以直接回答：

{tools}

回复格式说明
----------------------------

回复我时，请以以下两种格式之一输出回复：

选项 1：如果您希望人类使用工具，请使用此选项。
采用以下JSON模式格式化的回复内容：

```json
{{
    "reason": string, \\ 叙述使用工具的原因
    "action": string, \\ 要使用的工具。 必须是 {tool_names} 之一
    "action_input": string \\ 工具的输入
}}
```

选项2：如果您认为你已经有答案或者已经通过使用工具找到了答案，想直接对人类做出反应，请使用此选项。 采用以下JSON模式格式化的回复内容：

```json
{{
  "action": "Final Answer",
  "answer": string \\最终答复问题的答案放到这里！
}}
```

用户的输入
--------------------
这是用户的输入（请记住通过单个选项，以JSON模式格式化的回复内容，不要回复其他内容）：

{input}


"""

from langchain.agents import tool


@tool
def get_temperature(city):
    """返回城市的温度信息"""
    if (city.find('广') > -1):
        return 50
    else:
        return 12


@tool
def get_weather(city):
    """返回城市的天气信息"""
    if (city.find('广') > -1):
        return "下雨"
    else:
        return "下雪"


get_temperature.invoke('abccd')
get_weather.invoke('南京')

tools = [get_temperature, get_weather]

# 创建模板实例
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 组合成聊天模板
chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt, MessagesPlaceholder(variable_name="agent_scratchpad")])

# print('*' * 20 + "prompt 结果" + '*' * 20)
# print(chat_prompt.input_variables)

### 填充tools
from langchain.tools.render import render_text_description

chat_prompt = chat_prompt.partial(
    tools=render_text_description(list(tools)),
    tool_names=",".join([t.name for t in tools]),
)
#
# print('*' * 20 + "prompt 填充 tool 结果" + '*' * 20)
# print(chat_prompt.input_variables)
# print(chat_prompt.messages[1])


TEMPLATE_TOOL_RESPONSE = """工具响应：
---------------------
{observation}

用户的输入：
---------------------
请根据工具的响应判断，是否能够回答问题：

{input}

请根据工具响应的内容，思考接下来回复。回复格式严格按照前面所说的2种JSON回复格式，选择其中1种进行回复。请记住通过单个选项，以JSON模式格式化的回复内容，不要回复其他内容。
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def format_log_to_messages(
        query,
        intermediate_steps,
        template_tool_response,
):
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(input=query, observation=observation)
        )
        thoughts.append(human_message)
    return thoughts


from langchain.agents.agent import AgentOutputParser
from langchain_core.exceptions import OutputParserException


class JSONAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in JSON format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    {
      "action": "search",
      "action_input": "2+2"
    }
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    {
      "action": "Final Answer",
      "answer": "4"
    }
    ```
    """

    def parse(self, text):
        try:
            response = parse_json_markdown(text)
            # response = json.loads(text)
            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
                response = response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["answer"]}, text)
            else:
                return AgentAction(
                    response["action"], response.get("action_input", {}), text
                )
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "json-agent"


from langchain_core.runnables import RunnablePassthrough
from langchain.schema import AgentAction, AgentFinish
from langchain_core.runnables import RunnableLambda


def print_info(info: str):
    print(f"info: {info}")
    return info


agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["input"],
                x["intermediate_steps"],
                template_tool_response=TEMPLATE_TOOL_RESPONSE
            )
        )
        | chat_prompt
        | RunnableLambda(print_info)
        | llm
        | JSONAgentOutputParser()
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "广州的天气怎么样，温度是多少？如果没有查询到不要返回"})

# 填充变量并生成提示
# prompt = chat_prompt.format(product="姓名", question="张三的名字是什么")
# print(prompt)


# chain = chat_prompt | llm
# res = chain.invoke({"input": "广州天气怎么样"})
# print(res.content)

# chunks = []
# for chunk in chain.stream({"input": "张三"}):
#     chunks.append(chunk)
#     # 打印每个块的内容
#     print(chunk, end="|", flush=True)
