er 
import pymysql

from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.tools import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents import AgentType
from langchain.agents import AgentType, initialize_agent
from langchain.agents import tool
from langchain.tools import Tool
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.utilities import SerpAPIWrapper


load_dotenv()

def search(text: str) -> str:
    search = SerpAPIWrapper()
    res = search.run(f"{text}")
    return res

def main():
    
    user = 'root'
    host='localhost'
    database='my_test_db'
    local_uri = f"mysql+pymysql://{user}@{host}/{database}"
    
    db = SQLDatabase.from_uri(local_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)) 
    

    sql_agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        Verbose = True,
    )


    python_agent_executor = create_python_agent(
        llm= ChatOpenAI(temperature=0, model_name="gpt-4"), 
        tool=PythonREPLTool(),
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
    )


    csv_agent_executor = create_csv_agent(
        llm= ChatOpenAI(temperature=0, model_name="gpt-4"), 
        path="episode_info.csv",
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
    )


    grand_agent = initialize_agent(
     
        tools=[
        
        Tool(
            name="PythonAgent",
            func=python_agent_executor.run,
            description="""Useful when you need to transform natural language and write from in python and execute the python code
                        returning the results of the code execution
            """,
        ),
        
        Tool(
            name="CSVAgent",
            func=csv_agent_executor.run,
            description="""Useful when you need to answer question over episode_info.csv file,
                        takes an input of the entire question and returns the answer after running pandas calculations
            """,
        ),
        
        Tool(
            name="SearchAgent", #customized agent
            func=search,
            description="""Useful do you general research from Google.
            """,
        ),
        
        Tool(
            name="SQLAgent",
            func=sql_agent_executor.run,
            description="""Useful when you need to answer questions about TV show Friends by querying the database.
            """,
        )
        
        ],
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4"), 
        agent_type = AgentType.OPENAI_FUNCTIONS,
        verbose = True,
        handle_parsing_errors="Check your output and make sure it conforms!",

            
    )
    
    grand_agent.run("Enter your query here")


if __name__ == '__main__':
    main()    


