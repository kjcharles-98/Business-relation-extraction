import json
import spacy


j = []
with open("dataset.json",'r') as f:
    j = json.load(f)

nlp = spacy.load("en_core_web_lg")
limit = 1
no_entity = []
less_entity = []
more_entity = []
idx = 0
count = 0
for sample in j:
    print(idx)
    doc = nlp(sample['sent'])

    entityList = []
    for ent in doc.ents:
        entityList.append(ent.text)

    outputdict = {"sent":sample['sent'],"entity":entityList}

    if len(doc.ents) == 0 :
        no_entity.append(outputdict)
    elif len(doc.ents) <= limit:
        less_entity.append(outputdict)
    else:
        more_entity.append(outputdict)
    idx = idx+1


with open("no_entity.json",'w') as f:
    json.dump(no_entity,f)

with open("less_entity.json",'w') as f:
    json.dump(less_entity,f)

with open("more_entity.json",'w') as f:
    json.dump(more_entity,f)
