# utils/api_interface.py

"""
API Interface for ScaffAliens.

This module routes prompt-based calls to GPT (OpenAI) or other providers
like Jamba. It is controlled by the YAML config file.

Author: Sab & Luca
Created: 2nd May
"""

import openai
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import os
import json
import datetime
import time
import threading
_thread_local = threading.local()
from openai import OpenAI
import random
from openai import RateLimitError
import requests

# (Optional) placeholder for future Jamba integration
# from jamba_sdk import call_jamba  # hypothetically

def get_client():
    """Retrieve or create an OpenAI client for the current thread."""
    if not hasattr(_thread_local, "client") or _thread_local.client is None or _thread_local.client.is_closed():
        _thread_local.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _thread_local.client

#def send_text_to_openai_safe(text, model_name, retries=15):
def call_gpt_api(messages, model_name, retries=15):
    """
    Thread-local client + exponential backoff + logging to messages.log
    and responses.log.
    """
    client = get_client()
    # initial jitter so we don't ALL retry in lockstep
    time.sleep(random.uniform(0, 1) + 0.1)

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
            )
            msg_content = completion.choices[0].message.content

            # —— Saving logs ——
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            try:
                with open('messages.log', 'a', encoding='utf-8') as msg_file:
                    msg_file.write(json.dumps({
                        "timestamp": timestamp,
                        "messages": messages
                    }, ensure_ascii=False) + "\n")
            except IOError as io_err:
                print(f"[LOGGING ERROR] Couldn’t write messages: {io_err}")

            try:
                with open('responses.log', 'a', encoding='utf-8') as resp_file:
                    resp_file.write(json.dumps({
                        "timestamp": timestamp,
                        "response": msg_content
                    }, ensure_ascii=False) + "\n")
            except IOError as io_err:
                print(f"[LOGGING ERROR] Couldn’t write response: {io_err}")
            # ——

            return msg_content

        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Waiting for {wait_time:.2f}s before retrying "
                  f"(attempt {attempt + 1}/{retries}).")
            time.sleep(wait_time)

        except Exception:
            # propagate anything else
            raise

    raise Exception("Exceeded maximum retry attempts due to rate limits.")

def call_gpt_api_new(messages, model_name, retries=15):
    """
    Thread-local client + exponential backoff + logging to messages.log
    and responses.log.
    """
    client = get_client()
    # initial jitter so we don't ALL retry in lockstep
    time.sleep(random.uniform(0, 1) + 0.1)

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
            )
            msg_content = completion.choices[0].message.content

            # —— Saving logs ——
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            try:
                with open('messages.log', 'a', encoding='utf-8') as msg_file:
                    msg_file.write(json.dumps({
                        "timestamp": timestamp,
                        "messages": messages
                    }, ensure_ascii=False) + "\n")
            except IOError as io_err:
                print(f"[LOGGING ERROR] Couldn’t write messages: {io_err}")

            try:
                with open('responses.log', 'a', encoding='utf-8') as resp_file:
                    resp_file.write(json.dumps({
                        "timestamp": timestamp,
                        "response": msg_content
                    }, ensure_ascii=False) + "\n")
            except IOError as io_err:
                print(f"[LOGGING ERROR] Couldn’t write response: {io_err}")
            # ——

            return msg_content

        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Waiting for {wait_time:.2f}s before retrying "
                  f"(attempt {attempt + 1}/{retries}).")
            time.sleep(wait_time)

        except Exception:
            # propagate anything else
            raise

    raise Exception("Exceeded maximum retry attempts due to rate limits.")

def call_gpt_api_old(messages, model_name, temperature=0.3, max_tokens=5000):
    """
    Wrapper for OpenAI's chat API using conversation history (v1.0+).
    """
    try:
        # NEW: use the v1.0+ chat completion endpoint
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Extract the assistant reply
        msg = response.choices[0].message

        # —— Saving logs ——
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        # Save the outgoing messages
        try:
            with open('messages.log', 'a', encoding='utf-8') as msg_file:
                msg_file.write(json.dumps({
                    "timestamp": timestamp,
                    "messages": messages
                }, ensure_ascii=False) + "\n")
        except IOError as io_err:
            print(f"[LOGGING ERROR] Couldn’t write messages: {io_err}")

        # Save the assistant’s response
        try:
            with open('responses.log', 'a', encoding='utf-8') as resp_file:
                resp_file.write(json.dumps({
                    "timestamp": timestamp,
                    "response": msg.content
                }, ensure_ascii=False) + "\n")
        except IOError as io_err:
            print(f"[LOGGING ERROR] Couldn’t write response: {io_err}")
        # ——

        return msg.content
        # return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"[GPT API ERROR] {e}")
        return "API_ERROR"

def call_model_api(messages_or_prompt, model_name):
    """
    Unified API interface used across all agents.

    Parameters:
    -----------
    messages_or_prompt : list or str
        Either a chat message history (for GPT) or a single prompt string (for Jamba)
    model_config : dict
        From YAML config: provider, name, temperature, max_tokens
    use_chat_format : bool
        True if using chat history (GPT); False for Jamba or other APIs

    Returns:
    --------
    str : model response
    """
    if model_name == "gpt-4o":
        return call_gpt_api(messages_or_prompt, model_name)

    elif model_name.startswith("llama") or model_name.startswith("deepseek") or model_name.startswith("gpt-oss"):
        return call_local_llm(messages_or_prompt, model=model_name)
    raise ValueError(f"Unknown provider: {model_name}")


def call_jamba_api(messages, model_name="jamba-1.6-mini", temperature=0.3, max_tokens=300):
    """
    Call the AI21 Jamba chat completion endpoint and return the response text.

    Parameters:
    -----------
    messages : list of dict
        Conversation history: [{'role': 'system'|'user'|'assistant', 'content': '...'}, ...]
    model_name : str
        Jamba model name (e.g. "jamba-1.6-mini").
    temperature : float
    max_tokens : int

    Returns:
    --------
    str : Generated response content.
    """
    # Ensure API key is set

    api_key = os.getenv("AI21_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the AI21_API_KEY environment variable to use Jamba.")

    # Initialise the AI21 client
    client = AI21Client(api_key=api_key)

    # Convert our dict messages into ChatMessage objects
    chat_msgs = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in messages
    ]

    # Streamed completion
    stream = client.chat.completions.create(
        model=model_name,
        messages=chat_msgs,
        temperature=temperature,
        maxTokens=max_tokens,
        stream=True
    )

    # Accumulate content from each chunk
    response_content = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        # Append any new text
        if hasattr(delta, "content") and delta.content:
            response_content += delta.content

    return response_content.strip()

def call_local_llm(message, model="llama3"):
    """
    Call a local LLM server to get chat completions.

    Parameters:
        model (str): Model name (e.g., "llama3", "deepseek").
        host (str): Base URL of the local LLM server.
        api_key (str): Bearer token for authentication (if required).

        Returns:
        str: The assistant's reply (content from the LLM).
    """
    url = f"{os.getenv('LOCAL_API_URL')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LOCAL_API_KEY')}"
    }

    payload = {
        "model": model,
        "messages": message
    }
    max_retries = 10
    delay = 2  # seconds
    for attempt in range(1, max_retries + 1):  # max 5 retries
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            completion = response.json()
            return completion["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retries failed.")
        except (KeyError, IndexError) as e:
            print(f"Unexpected response structure: {e}")
            print("Raw response:", response.text)
            break  # no point in retrying if the response structure is wrong
        
    return ""



