import torch
from model import InputEmbeddings

voca = {"I'm": 0, "fine": 1, "thankyou": 2, ".": 3}
vocab_size = len(voca)
d_model = 512 

tokenized_input = ["I'm", "fine", "thankyou", "."]
input_index = [voca[token] for token in tokenized_input]

input_tensor = torch.tensor([input_index])

embedding_layer = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)

output_embedding = embedding_layer(input_tensor)

print("토큰 입력:", tokenized_input)
print("토큰 인덱스:", input_index)
print("임베딩:\n", output_embedding)
print("출력 모양:", output_embedding.shape)  