import gradio as gr
from huggingface_hub import InferenceClient
import os
import keras_hub
#import transformers
#from transformers import AutoModelForCausalLM,AutoTokenizer

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""

#from transformers import AutoTokenizer, AutoModelForCausalLM

import os

KAGGLE_KEY = os.getenv("KAGGLE_KEY")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
print(KAGGLE_USERNAME)



#tokenizer = AutoTokenizer.from_pretrained("komus/medquad-show")
#model = AutoModelForCausalLM.from_pretrained("komus/medquad-show")
path = "kaggle://smartsolutaris/gemma-2b-en-finetuned/keras/MedQuad"

class Chatbot:
    def __init__(self):
        self.model = keras_hub.models.GemmaCausalLM.from_preset(path)
        print(self.model)
        self.history = []
        self.isfirst = not bool(self.history)
        #self.tokenizer = transformers.GemmaTokenizer.from_pretrained("komus/medquad-show")
        if(self.isfirst):
            self.initiate_chat()

    def initiate_chat(self):
        greeting = "Hello there!!!, I answer a medical chatbot. Is there any question I can help with?"
        print(greeting)
        return greeting

    def format_message(self, message, previous):
        pass
        

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


def create_chatbot():
    chat = Chatbot()
    inf = gr.ChatInterface(
            respond,
            title = "MedQuad",
            description="ata",
            examples = ['As a healthcare fellow learning diagnosis, What is (are) Adhesions?',
            'As a healthcare fellow learning diagnosis, what research (or clinical trials) is being done for Miller Fisher Syndrome ?',
            'As a healthcare fellow learning diagnosis, What to do for Henoch-Schnlein Purpura '],
                theme = gr.themes.Soft()
            )
    return inf

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""


if __name__ == "__main__":
    chatbot = create_chatbot()
    chatbot.launch(share=True)
