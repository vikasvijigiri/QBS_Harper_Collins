import streamlit as st
import pypandoc
import os
from openai import OpenAI
from PyPDF2 import PdfMerger
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path
import os
import torch
import tempfile

#from dotenv import load_dotenv, find_dotenv




# Convert a file using pypandoc
def convert_file(input_file, output_file, output_format):
    pypandoc.convert_file(input_file, to=output_format, format='md', outputfile=output_file)


def from_prompt_gen_ans_main(api_key, outfile_name, system_message, user_message):

    #_ = load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=api_key#os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    model = 'gpt-4-turbo-preview'
    temperature = 0.4
    max_tokens=500
    topic = ""

    #print(os.environ.get("OPENAI_API_KEY"))

    # system_message = system_message()
    # prompt = user_message()


    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}

    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=messages,
    temperature=0.7
    )

    # print(completion.choices[0].message);
    content = completion.choices[0].message.content
    # Specify the file name

    # Write the content to a text file
    with open(outfile_name, "w", encoding="utf-8") as file:
        file.write(content)

    return content
    # print(f"Content successfully written to {file_name}")
    # print(f"Content is: ")
    # print()
    # print()
    # print(content)

def pdf_merger(pdf_files):
    # Create an instance of PdfFileMerger
    merger = PdfMerger()

    # List of PDF files to merge
    #pdf_files = ['pdf_files/sample_page1.pdf', 'pdf_files/sample_page2.pdf', 'pdf_files/sample_page3.pdf']

    # Iterate over the list of file paths
    for pdf_file in pdf_files:
        # Append PDF files
        merger.append(pdf_file)

    # Write out the merged PDF file
    merger.write("merged_pages.pdf")

    # Close the merger
    merger.close()
    return "merged_pages.pdf"


def model():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True, device_map="cpu")


    device = torch.device("cpu")  # Set the device to CPU
    # ... (Load your model and data onto the 'device') ...
    # Load the model with float32 and force it to use CPU
    model = AutoModel.from_pretrained(
        "ucaslcl/GOT-OCR2_0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
        device_map="cpu",
        torch_dtype=torch.float32  # Use float32 for CPU compatibility
    )

    # Move the model to CPU and set it to evaluation mode
    model = model.to(torch.device("cpu")).eval()

    # Verify the model is on CPU
    print("Model device:", next(model.parameters()).device)
    return model, tokenizer



# Function to convert PDF to high-quality images
def pdf_to_images(pdf_path, output_folder="output_images", dpi=300):
    # Convert the PDF to images with specified DPI for high quality
    images = convert_from_path(pdf_path, dpi=dpi)

    # Save images to the specified folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = []
    for i, img in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
        img.save(image_path, "JPEG")  # Save as JPEG or PNG for better quality
        image_paths.append(image_path)

    return image_paths



# Function to process uploaded images (e.g., save them as temporary files or process them)
def process_image(image_file) -> str:
    """
    Process an uploaded image file.
    Args:
        image_file: Uploaded image file object.
    Returns:
        str: Path to the temporary image file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_file.getvalue())
        return temp_file.name  # Return temporary file path

# Function to process uploaded images (e.g., save them as temporary files or process them)
def process_pdf(pdf_file) -> str:
    """
    Process an uploaded pdf file.
    Args:
        image_file: Uploaded pdf file object.
    Returns:
        str: Path to the temporary pdf file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getvalue())
        return temp_file.name  # Return temporary file path


# ----------------------------------------------------------------------------------
#   Model
# ----------------------------------------------------------------------------------

# import docx_extractor
# import query_data


st.set_page_config(layout='wide')
# Title for the Streamlit app
st.markdown("""<div>
  <h1 style='text-align: center; font-size: 52px; font-weight: 700; color: #202020; text-transform: uppercase; word-spacing: 1px; letter-spacing: 2px; position: relative;'>
    Content Filler
    </h1>
</div>
""", unsafe_allow_html=True)

# # Upload file options
# st.header("Upload Retrieval Book")
# retrieval_book = st.file_uploader("Choose a retrieval book (PDF or DOCX)", type=["pdf", "docx"])

# st.header("Upload To-be-Filled DOCX")
# to_be_filled_docx = st.file_uploader("Choose the DOCX document to be filled", type=["docx"])

# # API key input fields
# st.subheader("Enter API Keys")

# llm_api_key = st.text_input("LLM Model Access Token", type="password")
# qudrant_api_key = st.text_input("Qudrant Access Token", type="password")

# # Button to trigger document filling
# if st.button("Click to Fill the Document"):
#     if retrieval_book and to_be_filled_docx and llm_api_key and qudrant_api_key:
#         st.success("Document is being processed...")
#         # Insert logic to process the uploaded files using the API keys here
#         # E.g., use the keys and files to fill the document
#     else:
#         st.error("Please upload all files and enter the required API keys.")

# Initialize session state variables if not already set
if "retrieval_book" not in st.session_state:
    st.session_state.retrieval_book = None

if "to_be_filled_docx" not in st.session_state:
    st.session_state.to_be_filled_docx = None

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None

if "qudrant_api_key" not in st.session_state:
    st.session_state.qudrant_api_key = None

if "extracted_context" not in st.session_state:
    st.session_state.extracted_context = None

if "st.session_state.merged_pdf" not in st.session_state:
    st.session_state.merged_pdf = None

if "st.session_state.image_paths" not in st.session_state:
    st.session_state.image_paths = None



# Create tabs
tabs = st.tabs(["Context", "API Keys", "Process Document"])

# Content for "Upload Files" tab
with tabs[0]:
    st.header("Context")
    st.session_state.retrieval_book = st.file_uploader("Upload Retrieval Books (Images)", type=["pdf", "png", "jpg", "docx"], accept_multiple_files=True)
    if st.session_state.retrieval_book:
        for uploaded_pdf in st.session_state.retrieval_book:
            # Process each uploaded image and store its temporary path
            pdf_path = process_pdf(uploaded_pdf)
            #image_paths = pdf_to_images(pdf_path, output_folder="output_images", dpi=300)
            st.write(f"Processed pdf: {uploaded_pdf.name} (Temporary path: {pdf_path})")

            # Add the image path to session state
            # st.session_state.image_paths.append(image_path)

        # Display results
        st.write("All processed pdf paths:")
        #st.write(st.session_state.image_paths)
    else:
        st.write("Please upload pdf files.")
    st.write("OR")
    st.session_state.extracted_context = st.text_area("Paste the extracted content of the above PDF")
    st.session_state.to_be_filled_docx = st.file_uploader("Upload To-be-Filled Document (DOCX)", type=["docx"])
    # Example usage
    #if len(st.session_state.retrieval_book) == 1:
    #st.write(st.session_state.retrieval_book)
 
    
    #print("Saved images:", image_paths)

# Content for "API Keys" tab
with tabs[1]:
    st.header("API Keys")
    st.session_state.openai_api_key = st.text_input("LLM Model Access Token", type="password")
    #st.session_state.qudrant_api_key = st.text_input("Qudrant Access Token", type="password")

# Content for "Process Document" tab
with tabs[2]:
    st.header("Process Document")
    if (st.session_state.retrieval_book or st.session_state.extracted_context):# and st.session_state.to_be_filled_docx: # and st.session_state.llm_api_key and st.session_state.qudrant_api_key:
        st.success("Document is being processed...")
        # File path
        # docx_file = "./filler/example.docx"  # Replace with the path to your file

        # # Extract and display results
        # lesson_title, headers, content = docx_extractor.extract_headers_and_content(docx_file)
        # print("Lesson Title:", lesson_title)  # Display the full lesson title without "Lesson *:"
        # for header, cont in zip(headers, content):  # Iterate over headers and content
        #     print(f"\nHeader: {header}")
        #     print(f"Content: {cont}")

        # Insert document processing logic here
        # Text input


        model, tokenizer = model()
        res = ''        
        if st.session_state.retrieval_book:
            for uploaded_image in st.session_state.retrieval_book:
                # Process each uploaded image and store its temporary path
                image_path = process_image(uploaded_image)
                
                st.write(f"Processed image: {uploaded_image.name} (Temporary path: {image_path})")

                # Display the query


                res += model.chat(tokenizer, image_path, ocr_type='format')         
                # Add the image path to session state
                #st.session_state.image_paths.append(image_path)

            # Display results
            st.write("All processed image paths:")
            #st.write(st.session_state.image_paths)
        else:
            st.write("Please upload image files.")


        #print(res)

        system_message = st.text_area("Enter System Message:")
        user_message = st.text_area(res)
        # Button
        if st.button("Click to Query"):
            if system_message.strip() and user_message.strip():
                # Process the query (replace this logic with your specific functionality)

                # Example usage: Convert markdown to PDF
                # input_file = 'output.tex'
                # output_file = 'output.docx'
                # convert_file(input_file, output_file, 'docx')

                response = from_prompt_gen_ans_main(st.session_state.openai_api_key, 'output.txt', system_message, user_message)


                st.text_area(response)
                # response_text = query_data.query_rag(query)
                # st.success(f"Your query: '{user_input}' has been processed.")
                # st.write(response_text)
            else:
                st.error("Please enter a query before clicking the button.")



    else:
        st.error("Please upload all files and enter the required API keys.")