# QBS_Harper_Collins

IMP: Make sure conda is installed and the PWD is set to current folder

Step 1: conda create -p ./venv python==3.10

Step 2: conda activate ./venv

Step 3: Fill the .env file with the openAI api key under the variable OPENAI_API_KEY 

Step 4: pip install -r requirements.txt  (Install all the prerequisites)

Step 5: open the Jupyter Notebook content_generator.ipynb

Step 6: In the jupyter notebook, write the topics under the "topics" variable for which the chapter has to be generated and make sure all the variables and file names are correctly fed. Change the user_message and system message accordingly for each cell and for each chapter and it's content. 
NOTE: This is not generic code.  

Step 7: After step 6 is thoroughly taken care, run all the cell in order from top.

Step 8: In the last cell, the corresponding files generated are merged to give a complete chapter content.




