# 📚 Article Screening Tool for Bibliometric Research

A web-based application for screening academic articles for bibliometric studies. Automatically evaluates articles based on customizable criteria using rule-based matching or AI-powered analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **📁 Excel Upload** - Drag & drop support for WoS/Scopus exports
- **👀 Data Preview** - View and validate your data before processing
- **⚙️ Customizable Criteria** - Define your own inclusion/exclusion keywords
- **🔍 Rule-Based Screening** - Fast keyword matching
- **🤖 AI-Powered Screening** - Gemini or DeepSeek integration
- **📊 Results Filtering** - Filter by accepted/rejected
- **📥 Excel Export** - Download processed results

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/article-screener.git
cd article-screener

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage

1. Open `http://localhost:8501` in your browser
2. Upload your Excel file with article data
3. Configure screening criteria (optional)
4. Select analysis mode:
   - **Rule-Based**: Fast keyword matching
   - **Gemini AI**: Requires [Google AI API key](https://aistudio.google.com/apikey)
   - **DeepSeek AI**: Requires [DeepSeek API key](https://platform.deepseek.com)
5. Click "Start Analysis"
6. Download results as Excel

## 📋 Default Screening Criteria

Articles are **accepted** if they meet ALL criteria:

| Criterion | Description |
|-----------|-------------|
| **AI Technology** | Contains ChatGPT, GPT, LLM, chatbot, AI, etc. |
| **Web/Online Environment** | Contains online, e-learning, MOOC, distance learning, etc. |
| **Human Factor** | Contains teacher, student, learner, educator, etc. |

## 📁 Excel Format

Your Excel file should contain at minimum:

| Column | Required |
|--------|----------|
| Title | ✅ Yes |
| Abstract / Abstract Note | ✅ Yes |
| Acceptance | Optional (will be filled) |
| Reason | Optional (will be filled) |

## 🛠️ Requirements

- Python 3.8+
- pandas
- openpyxl
- streamlit
- google-genai (for Gemini)
- openai (for DeepSeek)

## 📄 License

MIT License - feel free to use for your research!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue.
(Developed by Ilyas Akkus)