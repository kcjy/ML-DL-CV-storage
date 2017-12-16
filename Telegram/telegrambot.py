# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:36:53 2017

@author: Kenneth
"""
import json
import requests
import time
import urllib
import pandas_datareader as pdr
from datetime import datetime

TOKEN = "423546507:AAGIRusfcH1uIC9uXOI27o7Dgrmp7izLc-k"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates"
    if offset:
        url += "?offset={}".format(offset)
    js = get_json_from_url(url)
    return js

#Change this
def send_message(text, chat_id):
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)

def handle_updates(updates):
    for update in updates["result"]:            
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
    if text == "/start":
        send_message("Welcome! Send any ticker to me and I will give you its stock price!", chat)
    else:
        symbol = str(text)
        ticker = pdr.get_data_yahoo(symbols=symbol, start=datetime(2000, 1, 1), end=datetime(2016, 12, 31))
        data = ticker["Adj Close"].tail()
        send_message(data, chat)    

def main():
    updates = get_updates(None)
    if len(updates["result"]) > 0:
        last_update_id = get_last_update_id(updates) + 1
        handle_updates(updates)
    time.sleep(0.5)
        
if __name__ == '__main__':
    main()