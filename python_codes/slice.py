### read a json text
import json
with open('all.jsonl', 'r',encoding='utf-8') as f:
    for line in f:
        # encode as json
        data = json.loads(line)
        text = data['text']
        label = data['label']
        # regular expression to find the text in between index 7 to 10
        text = text[7:10]
        
        # print the text
        print(data['text'])
        