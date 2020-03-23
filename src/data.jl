using JSON
using HDF5
using Flux

cd(@__DIR__)
json_file_data = JSON.parsefile("../data/quora_dataset/quora_data_prepro.json")
vocab = json_file_data["ix_to_word"]
vocab["0"] = "<UNK>"

ques_train_x = h5read("../data/quora_dataset/quora_data_prepro.h5", "ques_train")
ques_train_y = h5read("../data/quora_dataset/quora_data_prepro.h5", "ques1_train")
ques_test_x = h5read("../data/quora_dataset/quora_data_prepro.h5", "ques_test")
ques_test_y = h5read("../data/quora_dataset/quora_data_prepro.h5", "ques1_test")

seq_len = 10
n_words = length(vocab)

function process_data(data, data_len, batch_size, n_words)
    # [[Flux.onehotbatch(row, 0:(n_words-1)) for row in eachrow(ques_train_x[1:13,i:i+batch_size-1])] for i in 1:batch_size:(train_data_len-batch_size+1)]
    [[rand(0:1, n_words, batch_size) for row in 1:seq_len] for i in 1:10]
end

# train_data_len/batch_size ,seq_len, n_word, batch_size
