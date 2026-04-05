# smart-context-memory
A smart memory manager for LLM conversations. Keeps your prompt within token limits automatically — summarizing old messages and retrieving only what's relevant.

## The problem
Every LLM has a context window limit. As conversations grow, you have to choose between:
- Sending the full history → token limits blow up, costs spike
- Truncating old messages → the model forgets important context
This library handles it for you automatically.

## How it works
Every time you add a message, it counts the tokens. If you're over the limit, it compresses the oldest messages into a rolling summary. When you ask for context, it retrieves only the messages most relevant to your current query.

<img width="539" height="157" alt="image" src="https://github.com/user-attachments/assets/18381d78-92af-450b-9a5a-d3368e721105" />


## Install

<img width="520" height="84" alt="image" src="https://github.com/user-attachments/assets/470dea37-2363-458b-8f95-25644c8bcef8" />

## Usage

<img width="672" height="418" alt="image" src="https://github.com/user-attachments/assets/4b17f6c4-2527-43bd-baf3-73d813df6adb" />

## Requirements

- Python 3.9+

- tiktoken

- scikit-learn

- numpy
