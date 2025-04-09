import streamlit as st
import spacy.cli
spacy.cli.download("en_core_web_sm")
import spacy
from spacy import displacy

from transformers import pipeline
import streamlit.components.v1 as components

nlp = spacy.load("en_core_web_sm")


# Cache the Hugging Face model to avoid reloading each time
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("text2text-generation", model="./t5-small")




# Initialize the model (cached)
generator = load_model()

# Predefined templates for feedback
def provide_feedback(missing_component):
    if missing_component == "subject":
        return """**No subject was detected.** It is advisable to include a subject in your sentence.
        A subject answers **who** or **what** performs the action in the sentence.
        
        **Example**:  
        `I went to the park.`  
        (Here, "I" is the subject performing the action.)
        """
    elif missing_component == "verb":
        return """**No verb was detected.** A verb describes the action being performed in the sentence.
        
        **Example**:  
        `She ate lunch.`  
        (Here, "ate" is the verb describing what the subject did.)
        """
    elif missing_component == "object":
        return """**No object was detected.** The object receives the action of the verb.
        
        **Example**:  
        `She ate the cake.`  
        (Here, "the cake" is the object receiving the action.)
        """
    elif missing_component == "passive_voice":
        return """**The sentence is in passive voice.** We suggest changing it to active voice for clearer expression.
        
        **Example**:  
        `John threw the ball.`  
        (Active voice is often more direct and clear.)
        """
    elif missing_component == "adverb_placement":
        return """**The adverb is incorrectly placed.** Adverbs should generally come after the subject and verb or at the end of the sentence.
        
        **Example**:  
        `She quickly ran to the park.`  
        (Correct placement of adverb "quickly")
        """
    elif missing_component == "negative_sentence":
        return """**Negative sentences detected.** It’s important to use the right placement for negations to make the sentence clear.
        
        **Example**:  
        `She does not like coffee.`  
        (Negative sentences are perfectly valid but should be placed properly for clarity.)
        """
    elif missing_component == "multiple_subjects":
        return """**Multiple subjects detected.** If there are multiple subjects, ensure that the verb agrees with both subjects.
        
        **Example**:  
        `John and Mary went to the park.`  
        (This is a valid sentence, but keep subject-verb agreement in mind.)
        """
    elif missing_component == "complex_sentence":
        return """**Complex sentence detected.** Consider breaking it into shorter sentences or using proper conjunctions for clarity.
        
        **Example**:  
        `She went to the store, and then she went home.`  
        (Proper punctuation and conjunctions can help make complex sentences more readable.)
        """
    elif missing_component == "compound_sentence":
        return """**Compound sentence detected.** Ensure that each clause is properly connected with conjunctions and that subject-verb agreement is maintained.
        
        **Example**:  
        `I went to the store, and I bought coffee.`  
        (Use conjunctions like 'and', 'but', etc., for smooth transitions.)
        """
    elif missing_component == "incorrect_noun_usage":
        return """**Incorrect noun usage detected.** Ensure that your sentence uses nouns correctly, and avoid confusion.
        
        **Example**:  
        `She enjoys running.`  
        (The gerund "running" works better here than the noun "run.")
        """
    elif missing_component == "double_negative":
        return """**Double negative detected.** Double negatives can create confusion. Avoid using two negative words in the same sentence unless necessary.
        
        **Example**:  
        `She doesn't need any help.`  
        (Avoid saying "She doesn't need no help.")
        """
    elif missing_component == "misplaced_modifier":
        return """**Misplaced modifier detected.** Modifiers should be placed near the word they modify to avoid confusion.
        
        **Example**:  
        `She quickly ran to the store.`  
        (The modifier "quickly" should describe the action "ran.")
        """
    elif missing_component == "run_on_sentence":
        return """**Run-on sentence detected.** Consider splitting the sentence into two or more sentences or using proper punctuation.
        
        **Example**:  
        `She went to the store, she bought some coffee.`  
        (This should be two sentences: `She went to the store. She bought some coffee.`)
        """
    elif missing_component == "comma_splice":
        return """**Comma splice detected.** You should avoid using commas to join independent clauses without a conjunction or semicolon.
        
        **Example**:  
        `She went to the store, she bought coffee.`  
        (Corrected: `She went to the store; she bought coffee.`)
        """
    elif missing_component == "inverted_order":
        return """**The sentence order is inverted.** For NLP and SEO purposes, it is advisable to follow the standard Subject-Verb-Object (SVO) structure.
        
        **Example**:  
        `She ran to the park.`  
        (Standard order with subject "She" performing the action "ran".)
        """
    elif missing_component == "wrong_word_order":
        return """**Incorrect word order detected.** It’s important to follow the correct word order for clearer writing and better SEO.
        
        **Example**:  
        `The cat quickly ran to the park.`  
        (In NLP-friendly content, adverbs should come after the verb.)
        """
    elif missing_component == "excessive_passive_voice":
        return """**Excessive use of passive voice detected.** Too many passive constructions can make your content less engaging. Try using active voice for more clarity.
        
        **Example**:  
        `The cake was baked by Mary.`  
        (Rephrased to active voice: `Mary baked the cake.`)
        """
    else:
        return "Error: Component type not recognized."

def extract_svo(doc):
    """ Extract subject, verb, and object from the sentence using dependency parsing. """
    subject = None
    verb = None
    obj = None
    
    # Identify subject, verb, and object using dependency parsing
    for token in doc:
        if "subj" in token.dep_:  # Subject
            subject = token.text
        elif "VERB" in token.pos_:  # Verb
            verb = token.text
        elif "obj" in token.dep_:  # Object
            obj = token.text
    
    return subject, verb, obj

def check_order(doc, subject, verb, obj):
    """ Check if the sentence follows the correct SVO order. """
    if subject and verb and obj:
        # Check if passive voice is detected
        if any(tok.dep_ == "auxpass" for tok in doc):
            return "Passive voice detected. Suggested active sentence."
        else:
            return "Sentence is in correct SVM/SVO order. You can proceed."
    else:
        return "Something is missing from your sentence (subject, verb, or object)."

def process_sentence(sentence):
    """ Main function to process and convert sentence into SVM format. """
    doc = nlp(sentence)
    
    # Extract Subject, Verb, and Object
    subject, verb, obj = extract_svo(doc)
    
    # Check for correct order
    order_feedback = check_order(doc, subject, verb, obj)

    # Identify the conditions and provide feedback
    if not subject:
        feedback = provide_feedback("subject")
    elif not verb:
        feedback = provide_feedback("verb")
    elif not obj:
        feedback = provide_feedback("object")
    elif "not" in sentence:
        feedback = provide_feedback("negative_sentence")
    elif len(subject.split()) > 1:  # Multiple subjects detected
        feedback = provide_feedback("multiple_subjects")
    elif "quickly" in sentence:  # Adverb placement check
        feedback = provide_feedback("adverb_placement")
    elif len(sentence.split()) > 10:  # Complex sentence check
        feedback = provide_feedback("complex_sentence")
    elif len(set([subject, verb, obj])) == 3:  # Order check
        feedback = order_feedback
    elif len(subject.split()) > 2:  # Compound sentence detected
        feedback = provide_feedback("compound_sentence")
    elif "enjoys" in sentence:  # Noun usage check
        feedback = provide_feedback("incorrect_noun_usage")
    elif "doesn't need no" in sentence:  # Double negative check
        feedback = provide_feedback("double_negative")
    elif "almost" in sentence:  # Misplaced modifier check
        feedback = provide_feedback("misplaced_modifier")
    elif "," in sentence:  # Comma splice check
        feedback = provide_feedback("comma_splice")
    else:
        feedback = order_feedback
    
    return feedback

def render_syntax_tree(sentence):
    doc = nlp(sentence)
    html = displacy.render(doc, style="dep", options={"compact": True}, jupyter=False)
    return html
    
# Streamlit app layout
st.title("Full Subject-Verb-Modifier (SVM) Sentence Structure Checker")
st.write("""
    This app was inspired by Matt Diggity's video on creating crazy ChatGPT prompts for SEO.
    Check out his **[Video - ChatGPT Prompt BREAKS Google
](https://www.youtube.com/watch?v=uNGyKZwMTaQ)**, and follow him on Twitter: **[@MattDiggity](https://x.com/mattdiggityseo)**.
    
    This app helps you analyze sentences to see if they are in correct Subject-Verb-Object (SVO) or Subject-Verb-Modifier (SVM) format and provides feedback on how to improve them.
""")

# Display Matt Diggity's Section with Professional Link
st.image("data:image/jpeg;base64,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", caption="Matt Diggity's ChatGPT Prompts Video")
st.write("---🧠 For more resources like this, follow me on [Twitter](https://x.com/SankarGurumurt1")

# User input field
user_sentence = st.text_area("Enter your sentence to check for correct Subject-Verb-Object structure:)")

if st.button("Check Sentence"):
    if user_sentence:
        result = process_sentence(user_sentence)
        st.markdown(result)  # Display feedback as before

        # 🔥 Show syntax tree below the result
        st.markdown("### 🧠 Syntax Tree (Dependency Parse)")
        tree_html = render_syntax_tree(user_sentence)
        components.html(tree_html, height=300, scrolling=True)

    else:
        st.error("Please enter a sentence to process.")


# Footer Section with Professional Details
st.markdown("""
    ---  
    **Prepared with ❤ by Sankar Gurumurthy**  
    Follow me on:
    - [LinkedIn](https://www.linkedin.com/in/sankar-gurumurthy-a1044a136/) 
    - [Substack](https://sankargurumurthy.substack.com/)
    - [GitHub](https://github.com/sg-sankar)

    If you love this tool, don't forget to **star** the repository and share it with your network. Building it like a true **SEO NLP expert** and **AI/ML data science professional**.
""")
