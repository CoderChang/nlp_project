import json

pdtb_data_file = 'pdtb-data.json'
pdtb_parses_file = 'pdtb-parses.json'

output_data_file = 'pdtb-data-dev.json'
output_parses_file = 'pdtb-parses-dev.json'

sample_doc_num = 70


print 'opening files ...'
with open(pdtb_data_file) as f1:
    data_json_list = [json.loads(line) for line in f1.readlines()]
with open(pdtb_parses_file) as f2:
    all_parse_dicts = json.loads(f2.read())

relations = []
parse_dicts = {}

print 'sampling ...'
cur_doc_num = 0
for tmp_DocID in all_parse_dicts :
    cur_doc_num += 1
    if cur_doc_num > sample_doc_num:
        break
    print 'DocID: ', tmp_DocID

    parse_dicts[tmp_DocID] = all_parse_dicts[tmp_DocID]
    for data_json in data_json_list :
        if data_json['DocID'] == tmp_DocID :
            relations.append(data_json)


print 'writing to files ...'
output = open(output_data_file, 'w')
for relation in relations :
    output.write('%s\n' % json.dumps(relation))
output.close()

output = open(output_parses_file, 'w')
json.dump(parse_dicts, output, indent=4)
output.close()
