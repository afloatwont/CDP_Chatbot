# AI Chatbot for Customer Data Platforms (CDPs)

This project is an AI chatbot designed to answer "how-to" questions related to four Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap. The chatbot extracts relevant information from the official documentation of these platforms to guide users in performing tasks or achieving specific outcomes.

## Features

- **Documentation Scraping**: Automatically scrapes and loads documentation data from the official sources of Segment, mParticle, Lytics, and Zeotap.
- **Text Processing**: Cleans and formats the scraped text data for better usability.
- **Vector Database Management**: Manages storage and retrieval of processed embeddings for efficient querying.
- **Local Language Model**: Implements a local language model that does not require an API key, enabling the chatbot to generate responses.
- **Streamlit Interface**: Provides a user-friendly interface for users to interact with the chatbot.

## Project Structure

```
cdp-chatbot
├── src
│   ├── data
│   │   ├── loaders.py         # Documentation scrapers for each CDP
│   │   ├── processors.py      # Text processing utilities
│   │   └── storage.py         # Vector database management
│   ├── models
│   │   ├── embeddings.py      # Embedding models
│   │   ├── llm.py             # Local LLM implementation
│   │   └── chains.py          # LangChain components
│   ├── utils
│   │   ├── config.py          # Configuration settings
│   │   └── helpers.py         # Helper functions
│   ├── app.py                 # Main Streamlit application
│   └── ingest.py              # Script to run data ingestion
├── data
│   ├── raw                    # Raw documentation data
│   └── processed              # Processed embeddings and vector store
├── tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd cdp-chatbot
   ```

2. **Install Dependencies**:
   Use the following command to install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run Data Ingestion**:
   To scrape and process the documentation data, run:
   ```
   python -m src.data.ingest
   ```

4. **Start the Streamlit Application**:
   Launch the chatbot interface with:
   ```
   streamlit run src/app.py
   ```

## Usage

- Once the Streamlit application is running, users can enter their "how-to" questions related to Segment, mParticle, Lytics, and Zeotap.
- The chatbot will provide responses based on the scraped documentation and processed data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
