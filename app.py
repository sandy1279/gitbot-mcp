import os
import asyncio
import streamlit as st
from textwrap import dedent
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from agno.agent import Agent
from agno.tools.mcp import MCPTools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load env variables (for HF_TOKEN)
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="🐙 GitHub MCP Agent + LLaMA 3", page_icon="🦙", layout="wide")

st.title("🦙 LLaMA 3 GitHub Agent")
st.markdown("Explore GitHub repos with LLaMA 3 + MCP")

# Sidebar for GitHub Token
with st.sidebar:
    st.header("🔑 Authentication")
    github_token = st.text_input("GitHub Token", type="password")
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token

    st.markdown("---")
    st.caption("Use natural language queries like:\n- Show me recent issues\n- Analyze repo activity")

# Main input
repo = st.text_input("GitHub Repository", value="repo link")
query = st.text_area("Your Prompt", placeholder="e.g., Show me recent pull requests in the repo")

# Load LLaMA 3 8B pipeline (only once)
@st.cache_resource(show_spinner="Loading LLaMA 3 model...")
def load_llama():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", use_auth_token=os.getenv("HF_TOKEN"))
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

llama_pipe = load_llama()

# Custom Agent using LLaMA
class LlamaAgent(Agent):
    def __init__(self, tools, pipeline, **kwargs):
        super().__init__(tools=tools, **kwargs)
        self.pipeline = pipeline

    async def call_llm(self, message):
        prompt = message if isinstance(message, str) else message.render()
        response = self.pipeline(prompt, max_new_tokens=512, temperature=0.7, do_sample=True)
        return response[0]["generated_text"]

# Main agent runner
async def run_llama_agent(message):
    if not os.getenv("GITHUB_TOKEN"):
        return "⚠️ GitHub token missing!"

    try:
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                mcp_tools = MCPTools(session=session)
                await mcp_tools.initialize()

                agent = LlamaAgent(
                    tools=[mcp_tools],
                    pipeline=llama_pipe,
                    instructions=dedent("""\
                        You are a GitHub expert assistant powered by LLaMA 3.
                        - Summarize insights from GitHub repositories.
                        - Provide tables and lists when helpful.
                        - Use markdown formatting.
                    """),
                    markdown=True,
                    show_tool_calls=True
                )

                response = await agent.arun(message)
                return response.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Run Query
if st.button("🚀 Run Query"):
    if not github_token:
        st.error("Enter your GitHub token")
    elif not query:
        st.error("Enter a prompt")
    else:
        with st.spinner("Running LLaMA Agent..."):
            full_query = query if repo in query else f"{query} in {repo}"
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_llama_agent(full_query))
            st.markdown("### 💡 Results")
            st.markdown(result)

# Notes
st.markdown("---")
st.info("""
This app combines:
- **LLaMA 3 (8B Instruct)** for reasoning
- **Model Context Protocol (MCP)** for GitHub repo access
- **Streamlit** UI for natural language exploration
""")
