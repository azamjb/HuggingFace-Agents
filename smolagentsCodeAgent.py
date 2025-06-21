from huggingface_hub import login
import numpy as np
import time
import datetime

login()

from smolagents import CodeAgent, tool, DuckDuckGoSearchTool, LiteLLMModel

m = LiteLLMModel(
        model_id="ollama_chat/qwen2:7b",  
        api_base="http://127.0.0.1:11434",  # Default Ollama local server
        num_ctx=8192,
    )

# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."
    

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.
    
    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    
    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)
    
    return best_service



#agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=m)
#agent = CodeAgent(tools=[suggest_menu], model=m)
agent = CodeAgent(tools=[], model=m, additional_authorized_imports=['datetime'])


#agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
#agent.run("Prepare a superhero menu for the pary.")
agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)