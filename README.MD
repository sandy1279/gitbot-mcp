# 🦙 GitHub MCP Agent with LLaMA 3

Explore any public GitHub repository using natural language with the power of **Meta's LLaMA 3** and **Model Context Protocol (MCP)**.

Built with:
- 🦙 **LLaMA 3 8B Instruct** via Hugging Face
- 🧠 **Model Context Protocol (MCP)** for GitHub insights
- 🎈 **Streamlit** for an interactive UI
- 🔧 **Agno Framework** for agent management

![GitHub MCP Agent Screenshot](https://your-screenshot-link.png)

---

## 🚀 Features

- 🔍 Ask natural language questions about any GitHub repository
- 📊 Get live insights using MCP: issues, PRs, metrics, trends
- 🤖 Answers powered by LLaMA 3's reasoning and summarization
- 🧰 Fully local + API-based, uses your GitHub token securely

---

## 🧩 App Flow

Here’s a breakdown of how the app works:

1. **Authentication**:
   - When you first open the app, you’ll need to provide your GitHub token in the **sidebar**.
   - The token allows the app to access private/public repositories on your behalf. (Create one at [GitHub Token Settings](https://github.com/settings/tokens))

2. **Repository and Query Input**:
   - You can enter the **GitHub repository** (e.g., `facebook/react`) that you want to analyze.
   - You can also choose a predefined query type (such as **Issues**, **Pull Requests**, or **Repository Activity**) or type your custom query.

3. **Query Processing**:
   - After you click "Run Query", the app sends your prompt to a powerful **LLaMA 3 8B Instruct** model hosted via Hugging Face.
   - The query is processed using both the LLaMA 3 model and **Model Context Protocol (MCP)** to fetch data about the repository from GitHub (like issues, PRs, repo activity).

4. **Results Display**:
   - The results are then presented in a **streamlit UI**, formatted neatly with markdown, tables, and links to GitHub.
   - You can see insights like **open issues**, **PR statuses**, **recent repository activity**, and more.

5. **Interactive Exploration**:
   - You can continue interacting by modifying the query, selecting a different repo, or using different query types.

---

## 📦 Installation

```bash
https://github.com/sandy1279/gitbot-mcp
cd gitbot-mcp
pip install -r req.txt
