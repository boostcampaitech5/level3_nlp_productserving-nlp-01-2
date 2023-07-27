def summary(tokenizer, model, passage):
    input_ids = tokenizer.encode(passage, return_tensors='pt', add_special_tokens=True, truncation=True)
    #outputs = model.generate(input_ids=input_ids, num_beams=8, length_penalty=0.8, max_length=128)
    outputs = model.generate(input_ids=input_ids, num_beams=8, length_penalty=0.9, max_length=128)
    decoded_summary = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)

    print(f'summary:{decoded_summary}')
    return decoded_summary

