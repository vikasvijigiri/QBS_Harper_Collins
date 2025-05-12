from docx import Document

def combine_docx_files(file_list, output_file):
    """
    Combines multiple DOCX files into one, maintaining their order.
    
    Args:
        file_list (list of str): List of DOCX file paths in the desired order.
        output_file (str): Path to the output combined DOCX file.
    """
    # Create a new document for the combined output
    combined_doc = Document()

    for i, file_path in enumerate(file_list):
        # Load the current DOCX file
        current_doc = Document(file_path)
        
        # Append content to the combined document
        for element in current_doc.element.body:
            combined_doc.element.body.append(element)
        
        # Optionally, add a page break after each document except the last
        if i < len(file_list) - 1:
            combined_doc.add_page_break()
    
    # Save the combined document
    combined_doc.save(output_file)
    print(f"Combined document saved as: {output_file}")


