import PyPDF2
import regex as re
import itertools

main_characters = {
    'movie_scripts/american_psycho.pdf': 'BATEMAN'
}

def read_pdf(path_to_pdf: str):
    pdf = open(path_to_pdf, 'rb')
    pdf = PyPDF2.PdfReader(pdf)
    
    print('Path to CSV:', path_to_pdf)
    print('Number of pages of pdf:', len(pdf.pages))
    
    conversations_by_page = []
    
    for page in pdf.pages:
        text = page.extract_text()
        
        #print('Step 1: Remove text within parentheses')
        text_no_parentheses = re.sub(r"\([^)]*\)", "", text)
        
        #print('Step 2: Replace character names surrounded by newlines with <NAME>')
        text_with_names = re.sub(r'(?<=\n)([A-Z]+)(?=\n)', r'<\1>', text_no_parentheses)
        
        #print('Step 3: Remove mid sentence new lines')
        text_no_mid_sentence_newlines = re.sub(r'(?<![.?!])\n(?!\s)', ' ', text_with_names)
        
        #print('Step 4: Remove newlines following a <NAME> pattern')
        text_no_newlines_after_name = re.sub(r'(<[A-Z]+>)\n', r'\1', text_no_mid_sentence_newlines)
        
        #print('Step 5: Keep only the <NAME> followed by the dialogue up until the next newline')
        text_only_name_dialouge = re.sub(r'(<[A-Z]+>.*?)\n', r'\1\n', text_no_newlines_after_name)
        
        #print('Step 6: Remove everything except lines with <NAME> followed by text until newline')
        # This regex keeps only the lines where there's a name followed by dialogue up to the newline
        text_spaces = re.sub(r'^(?!<.*?>).*$', '', text_only_name_dialouge, flags=re.M)
        
        #print('Step 7: Replace multiple spaces with a single space')
        text_final = re.sub(r'\s+', ' ', text_spaces)
        
        #print('Step 8: Add a newline before each <NAME>')
        text_final_with_newlines = re.sub(r'(<[A-Z]+>)', r'\n\1', text_final)
        
        #print('Step 9: Remove the last three characters if they contain a number (1 to 1000) followed by a dot')
        text_final_no_last_number = re.sub(r'(\d{1,3})\.$', '', text_final_with_newlines)
        
        # print('Step 10: Replace the main character's name with <NAME> to generlaise between various movie scripts')
        text_final = re.sub(rf"<{main_characters[path_to_pdf]}>", "<MAIN>", text_final_no_last_number)
        
        # Remove whitespace elements in list
        page_list = [line for line in text_final.splitlines() if line.strip()]
        
        conversations_by_page.append(page_list)
    
    # Remove empty lists
    conversations_by_page = [el for el in conversations_by_page if el]
    
    # Combine all conversations to one list
    all_conversations = list(itertools.chain.from_iterable(conversations_by_page))
         
    return all_conversations