import json

def filter_tokens(instance):
    filtered_tokens = []
    filtered_chunk_tags = []
    filtered_pos_tags = []
    for token, chunk_tag, pos_tag in zip(instance['tokens'], instance['chunk_tags'], instance['pos_tags']):
        if token.isalnum():
            filtered_tokens.append(token)
            filtered_chunk_tags.append(chunk_tag)
            filtered_pos_tags.append(pos_tag)
    return filtered_tokens, filtered_chunk_tags, filtered_pos_tags

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            instance = json.loads(line)
            filtered_tokens, filtered_chunk_tags, filtered_pos_tags = filter_tokens(instance)
            if filtered_tokens:  # If tokens are not empty after filtering
                instance['tokens'] = filtered_tokens
                instance['chunk_tags'] = filtered_chunk_tags
                instance['pos_tags'] = filtered_pos_tags
                json.dump(instance, f_out)
                f_out.write('\n')

input_file = "train.jsonl"
output_file = "filtered_output.jsonl"
process_jsonl_file(input_file, output_file)
