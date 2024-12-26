# Prompt Engineering Techniques

## Overview
In this project, i applied In-Context Learning (ICL) techniques, including few-shot, one-shot, and zero-shot prompting,
to answer multiple-choice questions (MCQs) from the medical MCQ dataset.
These prompting techniques allow us to leverage a large language model (LLM) to solve complex medical questions without the need for additional task-specific training.

# Types of In-Context Learning Applied:
## Zero-Shot Prompting
For zero-shot prompting, the model is given a question and its options without any prior examples.
The model attempts to select the correct answer based purely on its pre-trained knowledge.

Example:

           Prompt: "What is the recommended treatment for hypertension?\n(A) Lisinopril\n(B) Amoxicillin\n(C) Acetaminophen"
           Model Output: "(A) Lisinopril"
-----------------------------
## One-Shot Prompting
In one-shot prompting, we provide one example of a question-answer pair from the dataset before asking the model to choose an answer for the next question. 
This helps the model better understand the task.

Example:

        Prompt: "Example:\nQuestion: What is the treatment for diabetes?\nAnswer: Insulin\n\nNow answer the following:\nWhat is the 
        recommended treatment for hypertension?\n(A)
        Lisinopril\n(B) Amoxicillin\n(C) Acetaminophen"
        Model Output: "(A) Lisinopril"


## Few-Shot Prompting
Few-shot prompting involves providing multiple question-answer pairs as examples to further guide the model in making the correct selection for a new question.
This method tends to improve performance over zero-shot and one-shot approaches by offering more context.

Example:

           Prompt: "Examples:\n1. Question: What is the treatment for diabetes?\n Answer: Insulin\n2. Question: What is the treatment for 
           asthma?\n Answer: 
           Albuterol\n\nNow answer the following:\nWhat is the recommended treatment for hypertension?\n(A) Lisinopril\n(B) 
           Amoxicillin\n(C) Acetaminophen"
           Model Output: "(A) Lisinopril"

This flexible prompting approach allows us to explore different ways of adapting LLMs to specific tasks without requiring task-specific fine-tuning.
--------------------------------------------
## Prompting with Roles

This project demonstrates how to use role-based prompting to guide Large Language Models (LLMs) in generating context-specific responses. The examples highlight how roles can provide additional context, tone, or style to the model's output.
## Features

- Basic Prompting: Example of a simple prompt without any role specification.

- Role-Based Prompting: Introduces a role context to refine the model's responses.

- Custom Tone and Style: Ability to customize the response tone, such as speaking like an English pirate.
### Code Examples

Basic Prompt
A simple prompt asking a question without specifying a role:

                           prompt = """
                           How can I answer this question from my friend:
                           What is the meaning of life?
                           """
                           inputs = tokenizer(prompt, return_tensors="pt")
                           outputs = llm.generate(**inputs, max_length=200)
                           response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                           print(response)
### Role-Based Prompt
A role-based prompt providing context and a specific tone:

                          role = """
                          Your role is a life coach \
                          who gives advice to people about living a good life.\
                          You attempt to provide unbiased advice.
                          You respond in the tone of an English pirate.
                          """
                          prompt_with_role = f"""
                         {role}
                         How can I answer this question from my friend:
                         What is the meaning of life?
                         """
                         inputs = tokenizer(prompt_with_role, return_tensors="pt")
                         outputs = llm.generate(**inputs, max_length=200)
                         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                         print(response)
## Requirements
- Python >= 3.8
- Hugging Face Transformers library
- PyTorch
Install the dependencies using:

                    pip install transformers torch
  ---------------------------------------------------------------------------------------------------------
# Chain-of-Thought Prompting
This notebook provides a foundation for exploring chain-of-thought prompting with an LLM. Modify the prompts and settings as needed to suit specific tasks or use cases.
### Structure

1. Prompting Problem Statement
"15 of us want to go to a restaurant. Two of them have cars, each car can seat 5 people. Two of us have motorcycles, each motorcycle can fit 2 people. Can we all get to the restaurant by car or motorcycle?"

2. Chain-of-Thought Prompt Variants
- Simple problem statement.
- Problem statement with instructions to think step by step.
- Problem statement with a request for step-by-step reasoning and final summary answer.
3. Model Inference Steps
- Tokenize the prompt using a tokenizer.
- Generate output using the language model.
- Decode the output and print the response.
## Requirements
- Tokenizer: Used to preprocess the prompt.
- Language Model (LLM): Used to generate responses.

                             pip install transformers torch
  
## Usage Instructions

1. Define the Problem Prompt
The problem is described as a natural language prompt. It is structured in three variations to test the language model's reasoning ability:
- Basic prompt.
- Prompt with explicit instructions to think step by step.
- Prompt requiring a single answer (yes/no) followed by an explanation.

2. Tokenize the Prompt

                               inputs = tokenizer(prompt, return_tensors="pt")
   This converts the text prompt into input tensors suitable for the LLM.
3. *Generate Response

                               outputs = llm.generate(**inputs, max_length=512)
   
