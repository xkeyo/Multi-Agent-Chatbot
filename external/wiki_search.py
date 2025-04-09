# external/wiki_search.py
import wikipedia
import logging

# Configure logging to display debug messages in the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def search_wikipedia(query: str, sentences: int = 2) -> str:
    # logging.debug(f"Searching Wikipedia for query: {query}")
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        # Log the first 100 characters of the summary for brevity
        # logging.debug(f"Found summary: {summary[:100]}...")
        return summary
    except Exception as e:
        error_message = f"(Wikipedia search failed: {str(e)})"
        # logging.debug(error_message)
        return error_message

if __name__ == '__main__':
    # Direct test mode
    query = input("Enter a Wikipedia search query: ")
    result = search_wikipedia(query)
    print("Result from Wikipedia:")
    print(result)
