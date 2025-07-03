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


# 模型
llm = ChatOllama(
    model = "qwen3:8b",
    temperature = 0,
    base_url="http://21.6.90.166.devcloud.woa.com:11434"
)

chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        # SystemMessage(
        #     content="你叫小智，是一名乐于助人的助手。"
        # ),
        # HumanMessage(content="你帮我算下，3.941592623412424+4.3434532535353的结果"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

messages_list = []
question = "你帮我算下，3.941592623412424+4.3434532535353的结果"
messages_list.append(HumanMessage(content=question))

basic_qa_chain = chatbot_prompt | llm

result = basic_qa_chain.invoke({"messages": messages_list})
print(result.content)


###定义function_tool

from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool


# 下面开始使用ReAct机制，定义工具，让LLM使用工具做专业的事情。

# 定义工具，要继承自LangChain的BaseTool
class SumNumberTool(BaseTool):
    name: str = "数字相加计算工具"
    description: str = "当你被要求计算2个数字相加时，使用此工具"

    def _run(self, a, b):
        return a + b


# 工具合集
tools = [SumNumberTool()]
# 提示词，直接从langchain hub上下载，因为写这个ReAct机制的prompt比较复杂，直接用现成的。
prompt = hub.pull("hwchase17/structured-chat-agent")
# 定义AI Agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
# 使用Memory记录上下文
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)
# 定义AgentExecutor，必须使用AgentExecutor，才能执行代理定义的工具
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)
# 测试使用到工具的场景
agent_executor.invoke({"input": "你帮我算下3.941592623412424+4.3434532535353的结果"})

# 测试不使用工具的场景
agent_executor.invoke({
                          "input": "请你充当稿件审核师，帮我看看'''号里的内容有没有错别字，如果有的话帮我纠正下。'''今天班级里的学生和老实要去哪里玩'''"})