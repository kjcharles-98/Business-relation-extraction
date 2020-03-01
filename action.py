import json
import spacy


nlp = spacy.load("en_core_web_lg")

filenames = ["no_entity.json","less_entity.json","more_entity.json"]


no_entity = []
less_entity = []
more_entity = []

index = 0

for file in filenames:
    j = []
    with open(file,'r') as f:
        j = json.load(f)


    for sentence in j:
        print(index)
        verb = []
        doc = nlp(sentence['sent'])
        entList = sentence['entity']
        for token in doc:
            for ent in entList:
                if token.text == ent:
                    t = token
                    while(True):
                        if t.pos_ == 'VERB':
                            if t.text not in verb:
                                verb.append(t.text)

                        if t.dep_ == "ROOT":
                            break
                        t = t.head
                    break
        sentence.update({"action":verb})
        if file == "no_entity.json":
            no_entity.append(sentence)
        elif file == "less_entity.json":
            less_entity.append(sentence)
        elif file == "more_entity.json":
            more_entity.append(sentence)
        index = index+1


with open("no_entity_action.json",'w') as f:
    json.dump(no_entity,f)

with open("less_entity_action.json",'w') as f:
    json.dump(less_entity,f)

with open("more_entity_action.json",'w') as f:
    json.dump(more_entity,f)
