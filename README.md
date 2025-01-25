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
```python
     Prompt: "What is the recommended treatment for hypertension?\n(A) Lisinopril\n(B) Amoxicillin\n(C) Acetaminophen"
     Model Output: "(A) Lisinopril"      
```
---------------------------
## One-Shot Prompting
In one-shot prompting, we provide one example of a question-answer pair from the dataset before asking the model to choose an answer for the next question. 
This helps the model better understand the task.

Example:
```python
    Prompt: "Example:\nQuestion: What is the treatment for diabetes?\nAnswer: Insulin\n\nNow answer the following:\nWhat is the 
    recommended treatment for hypertension?\n(A)
    Lisinopril\n(B) Amoxicillin\n(C) Acetaminophen"
    Model Output: "(A) Lisinopril"    
```

## Few-Shot Prompting
Few-shot prompting involves providing multiple question-answer pairs as examples to further guide the model in making the correct selection for a new question.
This method tends to improve performance over zero-shot and one-shot approaches by offering more context.

Example:
```python
   Prompt: "Examples:\n1. Question: What is the treatment for diabetes?\n Answer: Insulin\n2. Question: What is the treatment for 
   asthma?\n Answer: 
   Albuterol\n\nNow answer the following:\nWhat is the recommended treatment for hypertension?\n(A) Lisinopril\n(B) 
   Amoxicillin\n(C) Acetaminophen"
   Model Output: "(A) Lisinopril"       
```
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
```python
   prompt = """
   How can I answer this question from my friend:
   What is the meaning of life?
   """
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = llm.generate(**inputs, max_length=200)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)             
   print(response)                      
 ```                                            
### Role-Based Prompt
A role-based prompt providing context and a specific tone:
```python
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
```
## Requirements
- Python >= 3.8
- Hugging Face Transformers library
- PyTorch
Install the dependencies using:
  ```python
     pip install transformers torch
  ```
  ---------------------------------------------------------------------------------------------------
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

1. ### Define the Problem Prompt
The problem is described as a natural language prompt. It is structured in three variations to test the language model's reasoning ability:
- Basic prompt.
- Prompt with explicit instructions to think step by step.
- Prompt requiring a single answer (yes/no) followed by an explanation.

2. ### Tokenize the Prompt
```python
   inputs = tokenizer(prompt, return_tensors="pt")
```
   This converts the text prompt into input tensors suitable for the LLM.
3. ### Generate Response
```python
  outputs = llm.generate(**inputs, max_length=512)
```                         
 The LLM generates a response based on the tokenized input.
 4. ### Decode and Print Response
```python
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  print(response)                             
```                          
   This decodes the LLM’s response and prints the result.                             

****************************************************************************
# Solving Riddles using Few-Shot Prompting.

## Overview
This notebook demonstrates the use of LangChain and the Groq LLM to solve riddles using few-shot prompting. The script utilizes LangChain's `FewShotPromptTemplate` to format examples and queries to the Groq model. It showcases solving riddles by leveraging iterative reasoning steps to arrive at a solution.

---

## Requirements
- Python 3.8+
- LangChain
- LangChain-Groq Integration (`langchain_groq`)
Install the required Python packages using the following command:
```bash
pip install langchain langchain-groq
```

---

## Notebook Structure

### 1. **Imports**
The following key modules are imported:
- `PromptTemplate` and `FewShotPromptTemplate` from `langchain.prompts`
- `HumanMessage` from `langchain.schema`
- `ChatGroq` from `langchain_groq`

These modules enable prompt formatting, interaction with the Groq LLM, and structured messaging.

### 2. **Initialize the Groq Model**
```python
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
```
The `ChatGroq` object is initialized with the API key and model name.

### 3. **Define Riddles**
Three riddles are defined:
- Riddle 1: A train speed problem.
- Riddle 2: A farmer's animal count problem.
- Riddle 3: An age difference problem.

Each riddle is passed as a `HumanMessage` to the model.

### 4. **Few-Shot Examples**
The notebook defines two solved examples:
- Example 1: A train speed problem with reasoning steps.
- Example 2: A farmer's animal count problem with equations and reasoning steps.

These examples include intermediate reasoning steps and final answers, demonstrating the format expected from the LLM.

### 5. **Prompt Template**
A `FewShotPromptTemplate` is created using the examples, a specific example prompt structure, and a suffix for new input:
```python
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)
```

### 6. **Query the Model**
For each riddle, a prompt is formatted using the `FewShotPromptTemplate`, and the query is sent to the Groq LLM:
```python
message = HumanMessage(content=prompt.format(input=riddle_1))
print(llm([message]))
```
The responses are printed for each riddle.

---

## Example Outputs

### Riddle 1: Train Speed Problem
**Question:**
_A train leaves a station traveling at 60 km/h. One hour later, another train leaves the same station traveling in the same direction at 90 km/h. How long will it take for the second train to catch up to the first train?_

**Answer:**
```
We will follow up with some questions to get the answer.
Follow up: How far ahead is the first train when the second train starts?
Intermediate answer: In one hour, the first train travels 60 km.
Follow up: What is the relative speed between the two trains?
Intermediate answer: 90 km/h - 60 km/h = 30 km/h.
Follow up: How long will it take for the second train to cover the 60 km gap?
Intermediate answer: 60 km ÷ 30 km/h = 2 hours.
Final Answer: It will take 2 hours for the second train to catch up.
```

### Riddle 2: Farmer's Animals Problem
**Question:**
_A farmer has chickens and cows. There are 20 heads and 56 legs in total. How many chickens and cows does the farmer have?_

**Answer:**
```
We will follow up with some questions to get the answer.
Follow up: How many heads are there in total?
Intermediate answer: 20 heads, each animal has one head.
Follow up: How many legs does a chicken and a cow have?
Intermediate answer: A chicken has 2 legs and a cow has 4 legs.
Follow up: How can we set up equations to solve this?
Intermediate answer: Let x be the number of chickens and y be the number of cows. We know: x + y = 20 (heads) and 2x + 4y = 56 (legs).
Follow up: Solve the equations.
Intermediate answer: From x + y = 20, we get y = 20 - x. Substitute into the second equation: 2x + 4(20 - x) = 56. Simplify: 2x + 80 - 4x = 56 → -2x = -24 → x = 12. So, y = 20 - 12 = 8.
Final Answer: The farmer has 12 chickens and 8 cows.
```

---

## Usage Instructions
1. **Set Up API Key**:
   Replace `groq_api_key` with your actual API key.

2. **Run the Script**:
   Execute the script in a Python environment with the necessary dependencies installed.

3. **Modify Riddles**:
   Add your own riddles by defining them as strings and formatting them into the prompt.

4. **Interpret Results**:
   The model’s response will be printed in the console for each riddle, including reasoning steps and the final answer.

---

## Customization
- Add more examples to the `examples` list to improve the model's performance on similar questions.
- Modify the `PromptTemplate` structure to adjust the reasoning steps or format.

---

## Troubleshooting
- **Invalid API Key**: Ensure you are using a valid Groq API key.
- **Dependencies Missing**: Verify that all required libraries are installed.
- **Incorrect Answers**: Fine-tune examples in the `FewShotPromptTemplate` to guide the model more effectively.

---

#### Acknowledgements
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [Groq LLM API Documentation](https://groq.com/docs)

---------------------
--------------------------------
# Iterative Prompt Development

## Overview
This notebook uses the `langchain_groq` library to generate a compelling product description for a smartwatch based on its technical specifications. The notebook interacts with the Groq API via the `ChatGroq` class and includes iterative refinements to enhance the output. Key features include creating concise, engaging descriptions and incorporating structured data like tables for improved clarity.

## Features
1. **Groq API Integration**: The notebook connects to the Groq API to generate responses using the `llama-3.1-8b-instant` model.
2. **Iterative Refinement**: Prompts are progressively refined to:
   - Adjust text length.
   - Emphasize specific features or details.
   - Add structured elements such as tables.
3. **Dynamic Input**: The technical specifications of the smartwatch are used as input for generating descriptions.


---

## Prerequisites

### Libraries and Tools
- Python 3.8+
- Required libraries:
  - `langchain_groq`
 
### API Key
- You need a valid Groq API key stored in the Colab `userdata` or passed directly to the function.

### Environment Setup
Install the required libraries (if not already installed):
```bash
pip install langchain_groq google-colab
```

---

## Code Structure

### Key Components
1. **Function: `get_completion`**
   - Generates a response from the Groq API based on the provided prompt and parameters.
   - Arguments:
     - `prompt`: The input text prompt.
     - `model`: Model name (default: `llama-3.1-8b-instant`).
     - `temperature`: Controls randomness in the output (default: `0.5`).
     - `groq_api_key`: API key for authentication (optional).
   - Returns: The content of the generated response.

2. **Smartwatch Specifications (`smartwatch_specs`)**
   - Provides the technical details of the smartwatch, including its features, dimensions, and connectivity options.

3. **Prompts**
   - Iterative prompts refine the description:
     - Initial prompt: A general description emphasizing features and usability.
     - Refinement 1: Limits text length to 50 words.
     - Refinement 2: Focuses on clarity, adds a table for dimensions, and highlights unique features.

4. **Response Handling**
   - Processes and prints the output for each refinement step.

---

## How to Run the Code

1. **Set Up the API Key**
   - Store the API key in Colab `userdata`:
     ```python
     from google.colab import userdata
     userdata['groq_api_key'] = 'your_api_key_here'
     ```

2. **Run the Script**
   - Execute the script in a Python environment (e.g., Google Colab).
   - The script will:
     1. Generate an initial description.
     2. Refine the description to meet specific requirements.
     3. Add structured elements like a dimensions table.

3. **Outputs**
   - The generated descriptions and any structured data (e.g., tables) will be printed to the console.

---

## Example Output

### Initial Description
> "A sleek and modern smartwatch with a high-resolution AMOLED display, built-in GPS, and over 20 sports modes. Perfect for fitness enthusiasts and professionals alike."

### Refinement 1: Text Length
> "This smartwatch combines sleek design with functionality, featuring a high-res AMOLED display, 20+ sports modes, and a 14-day battery life. Its heart rate monitor, GPS, and SpO2 sensor make it ideal for fitness enthusiasts. Water-resistant up to 50 meters, it’s perfect for any lifestyle."

### Refinement 2: Dimensions Table
| **Dimension**   | **Specification** |
|------------------|--------------------|
| Diameter         | 42mm              |
| Thickness        | 10.5mm            |
| Weight (without strap) | 45g         |

---

## Customization

### Adjusting the Model
- Change the model by modifying the `model` parameter in the `get_completion` function.

### Tuning the Output
- Adjust `temperature` for more creative (higher value) or deterministic (lower value) responses.

### Adding More Refinements
- Update the prompts to include additional requirements (e.g., focus on sustainability, add testimonials).

---

## Troubleshooting

### Common Issues
1. **API Key Missing**: Ensure the Groq API key is correctly set in the environment or passed to the function.
2. **Library Not Found**: Install missing libraries using `pip install`.
3. **Unexpected Output**: Refine the prompt or adjust model parameters for better results.

---



