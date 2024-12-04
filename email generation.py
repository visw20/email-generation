# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the appropriate tokenizer and model
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# def generate_email(prompt, max_length=300, temperature=0.8):
#     # Tokenize input
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
    
#     # Generate text
#     outputs = model.generate(
#         inputs, 
#         max_length=max_length, 
#         num_return_sequences=1, 
#         temperature=temperature, 
#         top_k=50, 
#         top_p=0.95, 
#         pad_token_id=tokenizer.eos_token_id
#     )
    
#     # Decode generated text
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Refined prompt to instruct the model to generate a leave request email
# prompt = ("Write a professional and formal leave request email. "
#           "Reason: Attending my brother's marriage. "
#           "Include a polite tone, a subject line, and a closing remark.")


# # Generate email
# email_draft = generate_email(prompt)
# print(email_draft)








# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load model and tokenizer
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# def generate_email(prompt, max_length=150, temperature=0.8):
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
#     outputs = model.generate(
#         inputs, 
#         max_length=max_length, 
#         temperature=temperature, 
#         top_k=50, 
#         top_p=0.95, 
#         pad_token_id=tokenizer.eos_token_id
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Refined prompt
# prompt = ("Write a formal leave request for comming Wednesday. "
#           "Reason: going to native place for brother marriage. Tone: Professional.")

# # Generate improved email
# email_draft = generate_email(prompt)
# print(email_draft)











# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Load Flan-T5
# model_name = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# def generate_email(prompt, max_length=200):
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#     outputs = model.generate(inputs['input_ids'], max_length=max_length, num_beams=5, early_stopping=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Refined prompt
# prompt = "Write a professional leave request email for Wednesday. Reason: Going to native place for marriage."

# # Generate email
# email_draft = generate_email(prompt)
# print(email_draft)


##################################################




# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the appropriate tokenizer and model
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# def generate_email(prompt, subject, tone, reason, closing, max_length=300, temperature=0.7):
#     """
#     Generate a professional email with customizable parameters.

#     Args:
#         prompt (str): Main instruction for the email.
#         subject (str): Subject of the email.
#         tone (str): Tone of the email (e.g., Polite, Professional, Formal).
#         reason (str): Reason or context for the email.
#         closing (str): Closing remark or signature.
#         max_length (int): Maximum length of the generated email.
#         temperature (float): Sampling temperature for text generation.
#     Returns:
#         str: Generated email.
#     """
#     # Create a structured email format
#     full_prompt = (
#         f"Subject: {subject}\n\n"
#         f"Write a {tone.lower()} email addressing the following:\n"
#         f"1. The purpose: {reason}.\n"
#         f"2. A clear and concise body with professional language.\n"
#         f"3. A polite closing remark: {closing}.\n\n"
#         f"{prompt}"
#     )
    
#     # Tokenize the input
#     inputs = tokenizer.encode(full_prompt, return_tensors="pt")
    
#     # Generate text
#     outputs = model.generate(
#         inputs,
#         max_length=max_length,
#         temperature=temperature,
#         num_beams=5,  # Improves consistency
#         early_stopping=True,  # Ensures logical sentence termination
#         pad_token_id=tokenizer.eos_token_id
#     )
    
#     # Decode generated text
#     email_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # Post-process: Ensure proper formatting and remove artifacts
#     email_text = email_text.strip()
#     return email_text

# # Example Usage
# subject = "email for karuppur police station regarding missing by bike "
# tone = "formal"
# reason = "hello im viswa im lost my bike yesterday night im parked my bike in frount of my home , today morning bike was not there.take the complint and find my bike.tn 30 cx 7990 this is my bike number"
# closing = "Thank you for your time and consideration."
# prompt = "Write a formal email to karuppur police station"

# # Generate email
# email_draft = generate_email(prompt, subject, tone, reason, closing)
# print(email_draft)





















# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Initialize the model and tokenizer globally to avoid reloading
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# def format_email(subject, tone, reason, closing, prompt):
#     """
#     Format the email input into a structured prompt for the model.
#     """
#     return (
#         f"Subject: {subject}\n\n"
#         f"Write a {tone.lower()} email addressing the following:\n"
#         f"1. The purpose: {reason}.\n"
#         f"2. A clear and concise body with professional language.\n"
#         f"3. A polite closing remark: {closing}.\n\n"
#         f"{prompt}"
#     )

# def generate_text(prompt, max_length=300, temperature=0.7, num_beams=5):
#     """
#     Generate text using the model.
#     """
#     try:
#         # Tokenize input
#         inputs = tokenizer.encode(prompt, return_tensors="pt")
        
#         # Generate text
#         outputs = model.generate(
#             inputs,
#             max_length=max_length,
#             temperature=temperature,
#             num_beams=num_beams,
#             early_stopping=True,
#             pad_token_id=tokenizer.eos_token_id,
#         )
        
#         # Decode and return text
#         return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     except Exception as e:
#         return f"Error generating text: {str(e)}"

# def generate_email(subject, tone, reason, closing, prompt, **kwargs):
#     """
#     Generate a professional email with customizable parameters.
#     """
#     # Validate input
#     if not subject or not tone or not reason or not closing:
#         raise ValueError("Subject, tone, reason, and closing are required fields.")
    
#     # Format the prompt
#     formatted_prompt = format_email(subject, tone, reason, closing, prompt)
    
#     # Generate email text
#     email_text = generate_text(formatted_prompt, **kwargs)
    
#     return email_text

# # Example Usage
# if __name__ == "__main__":
#     subject = "mail for comissioner about water facility"
#     tone = "professional"
#     reason = ("im viswa, im from salem distict  from vellalapatty area,in my area we not having water facility for last 30 to 50 years. so place take some action in my area")
#     closing = "Thank you for your time and assistance."
#     prompt = "Write a professional email for water facility"
    
#     # Generate the email
#     email = generate_email(
#         subject=subject,
#         tone=tone,
#         reason=reason,
#         closing=closing,
#         prompt=prompt,
#         max_length=500,  # Customize parameters as needed
#         temperature=0.7,
#     )
    
#     print(email)








from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the model and tokenizer globally to avoid reloading
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cpu")
model.to(device)

def format_email(subject, tone, reason, closing, prompt):
    """
    Format the email input into a structured prompt for the model.
    """
    return (
        f"Subject: {subject}\n\n"
        f"Write a {tone.lower()} email addressing the following:\n"
        f"1. The purpose: {reason}.\n"
        f"2. A clear and concise body with professional language.\n"
        f"3. A polite closing remark: {closing}.\n\n"
        f"{prompt}"
    )

def generate_text(prompt, max_length=500, temperature=0.7, num_beams=5, repetition_penalty=1.2):
    """
    Generate text using the model with improved repetition handling.
    """
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Create attention mask: 1 for all tokens since pad_token == eos_token
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # All tokens should be attended to
        
        # Generate text
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,  # Explicitly pass the attention mask
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,  # Avoid repetition
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id as pad_token_id
        )
        
        # Decode and post-process text
        email_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        email_text = remove_redundancy(email_text)  # Remove redundant lines
        return email_text
    except Exception as e:
        return f"Error generating text: {str(e)}"

def remove_redundancy(text):
    """
    Removes repeated content in the generated text.
    """
    sentences = text.split('\n')
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        if sentence.strip() not in seen_sentences:
            unique_sentences.append(sentence.strip())
            seen_sentences.add(sentence.strip())

    return '\n'.join(unique_sentences)


def generate_email(subject, tone, reason, closing, prompt, **kwargs):
    """
    Generate a professional email with customizable parameters.
    """
    # Validate input
    if not subject or not tone or not reason or not closing:
        raise ValueError("Subject, tone, reason, and closing are required fields.")
    
    # Format the prompt
    formatted_prompt = format_email(subject, tone, reason, closing, prompt)
    
    # Generate email text
    email_text = generate_text(formatted_prompt, **kwargs)
    
    return email_text

# Example Usage
if __name__ == "__main__":
    subject = "mail for school pricipal "
    tone = "formal"
    reason = ("hi i need admision for my boy in your school now he was studing 9th")
    closing = "Thank you for your time and assistance."
    prompt = "Write a professional email for admission"
    
    # Generate the email
    email = generate_email(
        subject=subject,
        tone=tone,
        reason=reason,
        closing=closing,
        prompt=prompt,
        max_length=500,  # Customize parameters as needed
        temperature=0.7,
    )
    
    print(email)
