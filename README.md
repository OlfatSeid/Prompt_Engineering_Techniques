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
