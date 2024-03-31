from model import Model
import json

with open('train.jsonl') as f:
        train_data = [json.loads(line) for line in f]

with open('test.jsonl') as f:
        test_data = [json.loads(line) for line in f]

    
def create_data(chunks, pos_tags):
    d = {}
    ones, zeros = 0, 0
    s = set()
    for i in range(len(chunks)):
        for j in range(1, len(chunks[i])):
            key = str(pos_tags[i][j-1]) + "_" + str(pos_tags[i][j]) + "="  + str(chunks[i][j])
            d[key] = d.get(key, 0)+1
            if str(chunks[i][j])=='1':
                 ones+=1
            else:
                 zeros+=1
            s.add(chunks[i][j])
    print(s)
    print(f"\nIn create_data zeros : {zeros} and ones are : {ones}\n")
    return d

original_chunks, original_pos = [], []
for i in test_data:
     original_chunks.append(i['chunk_tags'])
     original_pos.append(i["pos_tags"])
ori_d = create_data(original_chunks, original_pos)
# for k, v in ori_d.items():
#      print(k, v)

model = Model()
model.train(train_data, 2)
model.test(train_data)
print(model.weights)
model.test(test_data)
print(model.weights)
test_chunks, test_pos = [], []
for i in test_data:
    c, l = model.forward(i)
    test_chunks.append(c)
    test_pos.append(i['pos_tags'])

z, o = 0,0 
for i in test_chunks:
     for j in i[1:]:
          if j:o+=1
          else: z+=1
print(z, o)

print(test_chunks[:10])

print("\n\nTesting:")
test_d = create_data(test_chunks, test_pos)
# for k, v in test_d.items():
#      print(k, v)
for k, v in ori_d.items():
    print(f"{k} y:{v} pred:{test_d.get(k, 0)}      diff: {v-test_d.get(k, 0)}")