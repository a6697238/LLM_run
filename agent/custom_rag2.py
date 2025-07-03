from typing import List

from langchain_ollama import ChatOllama

# 模型
llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0,
    base_url="http://21.6.90.166.devcloud.woa.com:11434"
)

from langchain.agents import tool


@tool
def get_word_length(word: str) -> int:
    """返回一个单词的长度"""
    return len(word)


@tool
def get_word_start(word: str) -> str:
    """返回一个单词的首字符"""
    return word[0]


get_word_length.invoke("abc")

tools = [get_word_length,get_word_start]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

system_template = "你是一个客服助手，回答用户的问题。"
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

# # 创建模板实例
# system_prompt = SystemMessagePromptTemplate.from_template(system_template)
# human_prompt = HumanMessagePromptTemplate.from_template(human_template)
#
# # 组合成聊天模板
# chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])


# 创建模板实例
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 组合成聊天模板

chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt])

# chat_prompt = ChatPromptTemplate.from_messages(
#     [system_prompt, human_prompt, MessagesPlaceholder(variable_name="agent_scratchpad")])

# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你是一个客服助手，回答用户的问题。"),
#         ("user", human_template),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

from langchain.tools.render import render_text_description

chat_prompt = chat_prompt.partial(
    tools=render_text_description(list(tools)),
    tool_names=",".join([t.name for t in tools]),
)

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


from langchain_core.runnables import RunnablePassthrough

TEMPLATE_TOOL_RESPONSE = """工具响应：
---------------------
{observation}

用户的输入：
---------------------
请根据工具的响应判断，是否能够回答问题：

{input}

请根据工具响应的内容，思考接下来回复。回复格式严格按照前面所说的2种JSON回复格式，选择其中1种进行回复。请记住通过单个选项，以JSON模式格式化的回复内容，不要回复其他内容。
"""

from langchain.agents.agent import AgentOutputParser
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish


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


agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["input"],
                x["intermediate_steps"],
                template_tool_response=TEMPLATE_TOOL_RESPONSE
            )
        )
        | chat_prompt
        | llm
        | JSONAgentOutputParser()
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "这个单词里的长度是多少 eudca"})
# list(agent_executor.stream({"input": "这个单词里的长度是多少 eudca"}))


