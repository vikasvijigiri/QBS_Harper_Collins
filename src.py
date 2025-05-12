import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import subprocess
from docx import Document

def from_prompt_gen_ans_main(file_name, system_message, user_message):

    _ = load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    model = 'gpt-4-turbo-preview'
    temperature = 0.7
    max_tokens=50000
    n = 1
    topic = ""

    #print(os.environ.get("OPENAI_API_KEY"))

    # system_message = system_message()
    # prompt = user_message()


    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}

    ]

    completion = client.chat.completions.create(
    model="o1",
    store=True,
    messages=messages
    )

    # print(completion.choices[0].message);
    content = completion.choices[0].message.content
    # Specify the file name

    # Write the content to a text file
    with open("dum.tex", "w", encoding="utf-8") as file:
        file.write(content)

    subprocess.run(['pandoc', '-s', "dum.tex", '-o', file_name], check=True, text=True)
 

    print(f"Content successfully written to {file_name}")
    # print(f"Content is: ")
    # # print()
    # # print()
    # # print(content)




def read_docx(file_path):
    """
    Reads the content of a DOCX file and returns it as a string.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: The content of the DOCX file as a single string.
    """
    try:
        # Load the DOCX file
        doc = Document(file_path)
        
        # Extract all text and join paragraphs
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        return content
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return None

