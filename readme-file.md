# 🏨 Hotel Reviews Assistant

A Streamlit-powered application that allows users to query hotel reviews using the Gemini AI model. This assistant helps users find relevant information about specific hotels based on customer reviews.

## 📋 Features

- Interactive chat interface for querying hotel reviews
- Hotel selection from available options
- AI-powered responses based on real customer feedback
- Contextual awareness with conversation history
- Vector-based similarity search for accurate information retrieval

## 🔧 Technology Stack

- **Streamlit**: Web application framework
- **LangChain**: Framework for developing applications powered by language models
- **Google Gemini AI**: Advanced large language model
- **FAISS**: Vector store for efficient similarity search
- **Beautiful Soup**: HTML parsing and sanitization
- **Pandas**: Data manipulation and analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Google API key for Gemini AI

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hotel-reviews-assistant.git
cd hotel-reviews-assistant
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### Running the Application

```bash
streamlit run gemini_2.py
```

The application will be available at `http://localhost:8501`.

## 🌐 Development with Codespaces

This repository is configured to work with GitHub Codespaces. You can develop directly in your browser by:

1. Clicking the "Code" button on the repository
2. Selecting the "Codespaces" tab
3. Clicking "Create codespace on main"

The devcontainer configuration will:
- Set up Python 3.11
- Install required dependencies
- Start the Streamlit server automatically

Remember to add your Google API key to a `.env` file in the codespace.

## 📂 Project Structure

- `gemini_2.py`: Main application file
- `htmlTemplates.py`: HTML templates for chat UI
- `requirements.txt`: Project dependencies
- `.devcontainer/`: Configuration for development containers
- `data/`: Directory containing the hotel reviews data (veri.pkl)

## 🔄 How It Works

1. **Data Loading**: The application loads hotel review data from a pickle file
2. **Text Processing**: Reviews are split into chunks and converted to embeddings
3. **Vector Store**: FAISS creates an efficient searchable index
4. **Query Processing**: User questions are processed using similarity search
5. **AI Response**: Gemini AI generates contextual responses based on relevant reviews

## ⚠️ Important Notes

- You need to provide your own Google API key in the `.env` file
- The application expects a `veri.pkl` file in the `data/` directory with hotel review data
- Sensitive information should be stored in `.env` or `secrets.toml` files (both are gitignored)

## 🌍 Language

The application is currently configured for Turkish language hotel reviews and responses.

## 🔒 Security

- HTML content is sanitized using BeautifulSoup to prevent XSS attacks
- API keys are stored in environment variables and not exposed in the code

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
