import re
import sqlite3
from collections import Counter
from string import punctuation
from math import sqrt
import json
import requests
import time

TOKEN = "423546507:AAGIRusfcH1uIC9uXOI27o7Dgrmp7izLc-k"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

# initialize the connection to the database
connection = sqlite3.connect("C:\\Users\\Kenneth\\Documents\\chatbot.sqlite")
cursor = connection.cursor()
 
# create the tables needed by the program
create_table_request_list = [
    'CREATE TABLE words(word TEXT UNIQUE)',
    'CREATE TABLE sentences(sentence TEXT UNIQUE, used INT NOT NULL DEFAULT 0)',
    'CREATE TABLE associations (word_id INT NOT NULL, sentence_id INT NOT NULL, weight REAL NOT NULL)',
]
for create_table_request in create_table_request_list:
    try:
        cursor.execute(create_table_request)
    except:
        pass
        
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
    
def get_id(entityName, text):
    """Retrieve an entity's unique ID from the database, given its associated text.
    If the row is not already present, it is inserted.
    The entity can either be a sentence or a word."""
    tableName = entityName + 's'
    columnName = entityName
    cursor.execute('SELECT rowid FROM ' + tableName + ' WHERE ' + columnName + ' = ?', (text,))
    row = cursor.fetchone()
    if row:
        return row[0]
    else:
        cursor.execute('INSERT INTO ' + tableName + ' (' + columnName + ') VALUES (?)', (text,))
        return cursor.lastrowid
 
def get_words(text):
    """Retrieve the words present in a given string of text.
    The return value is a list of tuples where the first member is a lowercase word,
    and the second member the number of time it is present in the text."""
    wordsRegexpString = '(?:\w+|[' + re.escape(punctuation) + ']+)'
    wordsRegexp = re.compile(wordsRegexpString)
    wordsList = wordsRegexp.findall(text.lower())
    return Counter(wordsList).items()
    
def handle_updates(updates):
    for update in updates["result"]:            
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
    if text == "/start":
        send_message("Hello! I am Chat, a chatbot created to relieve you of your boredom.", chat)
    else:
        send_message(reply_mech(updates), chat)
        
def reply_mech(updates):    
    for update in updates["result"]:            
        text = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
   
    while True:
        H = text.split()
        B = "Hello!"
        # store the association between the bot's message words and the user's response
        words = get_words(B)
        words_length = sum([n * len(word) for word, n in words])
        sentence_id = get_id('sentence', H)
        for word, n in words:
            word_id = get_id('word', word)
            weight = sqrt(n / float(words_length))
            cursor.execute('INSERT INTO associations VALUES (?, ?, ?)', (word_id, sentence_id, weight))
        connection.commit()
    
        # retrieve the most likely answer from the database
        cursor.execute('CREATE TEMPORARY TABLE results(sentence_id INT, sentence TEXT, weight REAL)')
        words = get_words(H)
        words_length = sum([n * len(word) for word, n in words])
        for word, n in words:
            weight = sqrt(n / float(words_length))
            cursor.execute('INSERT INTO results SELECT associations.sentence_id, sentences.sentence, ?*associations.weight/(4+sentences.used) FROM words INNER JOIN associations ON associations.word_id=words.rowid INNER JOIN sentences ON sentences.rowid=associations.sentence_id WHERE words.word=?', (weight, word,))
        # if matches were found, give the best one
        cursor.execute('SELECT sentence_id, sentence, SUM(weight) AS sum_weight FROM results GROUP BY sentence_id ORDER BY sum_weight DESC LIMIT 1')
        row = cursor.fetchone()
        cursor.execute('DROP TABLE results')
        # otherwise, just randomly pick one of the least used sentences
        if row is None:
            cursor.execute('SELECT rowid, sentence FROM sentences WHERE used = (SELECT MIN(used) FROM sentences) ORDER BY RANDOM() LIMIT 1')
            row = cursor.fetchone()
        # tell the database the sentence has been used once more, and prepare the sentence
        B = row[1]
        cursor.execute('UPDATE sentences SET used=used+1 WHERE rowid=?', (row[0],))
        
    
def main():
    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            handle_updates(updates)
        time.sleep(0.5)
    
if __name__ == '__main__':
    main()
    
