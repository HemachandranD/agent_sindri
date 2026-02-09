import os
import random

import pandas as pd
import requests
import wikipedia
from langchain.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# Global variable to store the current task_id (set by the agent)
_current_task_id = None


def set_current_task_id(task_id: str):
    global _current_task_id
    _current_task_id = task_id


def get_current_task_id():
    return _current_task_id


@tool
def web_search(query: str) -> str:
    """
    Perform a live web search using Tavily. Best for finding current information, recent news, specific facts, or details that might not be in general knowledge databases. Returns a string containing the content, title, and source URL of up to 3 relevant web pages.

    Args:
        query: The search query to send to the web search engine.
    """
    print(f"DEBUG: Inside web_search tool. Received query: '{query}'")
    try:
        search_results = TavilySearchResults(max_results=3).invoke(
            query
        )  # Invoke returns list of dicts

        formatted_search_docs = "\n\n---\n\n".join(
            [
                # Access keys of the dictionary instead of .metadata and .page_content
                f'<Document source="{doc.get("url", "")}" title="{doc.get("title", "")}">\n{doc.get("content", "No content available")}\n</Document>'
                for doc in search_results
            ]
        )

        print(
            f"DEBUG: web_search tool successful. Found {len(search_results)} results."
        )
        return {"web_results": formatted_search_docs}
    except Exception as e:
        print(f"ERROR: Error within web_search tool: {e}")
        return f"Error: {e}"


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for information. Excellent for historical information, biographies, scientific concepts, or well-established topics. Uses the latest available version of English Wikipedia. Returns a string containing the content and metadata (source, page) of up to 2 relevant Wikipedia articles.

    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b


@tool
def read_excel_file(task_id: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    To use this file you need to have saved it in a location and pass that location to the function.

    Args:
        task_id: Path to the Excel file
        query: Question about the data

    Returns:
        Analysis result or error message
    """
    DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

    try:
        print(f"🔍 excel_file read with: {task_id}")

        # Smart replacement logic
        actual_task_id = task_id

        if _current_task_id:
            print(f"🔄 Replacing '{task_id}' with actual task_id: {_current_task_id}")
            actual_task_id = _current_task_id
        else:
            return "Error: No valid task_id available for excel file"

        print(f"✅ Using task_id: {actual_task_id}")

        file_path = f"temp_{actual_task_id}.xlsx"
        response = requests.get(
            f"{DEFAULT_API_URL}/files/{actual_task_id}", stream=True
        )
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        df = pd.read_excel(file_path)
        os.remove(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        print(f"✅ File read completed: {df.to_string()[:100]}...")
        return result
    except ImportError:
        return "Error: pandas and openpyxl are not installed. Please install them with 'pip install pandas openpyxl'."
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"


@tool
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"
