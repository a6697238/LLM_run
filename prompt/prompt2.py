from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
import re
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 模型
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
    base_url="http://21.6.90.166.devcloud.woa.com:11434"
)

# 定义角色模板
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

from langchain.agents import tool


@tool
def get_word_length(word):
    """retrun word length"""
    return len(word)


@tool
def get_user_age(name):
    """retrun user age"""
    if (name.find('张') > -1):
        return 1
    else:
        return 100


get_word_length.invoke('abccd')
get_user_age.invoke('张三')

tools = [get_word_length, get_user_age]

# 创建模板实例
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 组合成聊天模板
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# print('*' * 20 + "prompt 结果" + '*' * 20)
# print(chat_prompt.input_variables)

### 填充tools
from langchain.tools.render import render_text_description

chat_prompt = chat_prompt.partial(
    tools=render_text_description(list(tools)),
    tool_names=",".join([t.name for t in tools]),
)

# print('*' * 20 + "prompt 填充 tool 结果" + '*' * 20)
# print(chat_prompt.input_variables)
# print(chat_prompt.messages[1])


# 填充变量并生成提示
# prompt = chat_prompt.format(product="姓名", question="张三的名字是什么")
# print(prompt)


chain = chat_prompt | llm
res = chain.invoke({"input": "广州天气怎么样"})
print(res.content)

# chunks = []
# for chunk in chain.stream({"input": "张三"}):
#     chunks.append(chunk)
#     # 打印每个块的内容
#     print(chunk, end="|", flush=True)
