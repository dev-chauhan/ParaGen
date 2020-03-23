using Flux

include("data.jl")
include("encdec.jl")

# pgen = EncDec(n_words, 512, 512, seq_len)|>gpu
emb = Dense(n_words, 512)
enc = LSTM(512, 512)
dec = Chain(LSTM(512, 512),Dense(512, n_words), logsoftmax)

opt = Flux.ADAM()

function loss(x, y)
    # seq = pgen(x, y)
    y_ = [y[i] for i in 1:(length(y)-1)]
    seq = dec.([[enc.(emb.(x))[end]]; emb.(y_)])

    Flux.reset!(enc)
    enc_x = enc.(emb.(x))[end]
    
    Flux.reset!(enc)
    enc_y = enc.(emb.(y))[end]
    
    l1 = sum(Flux.crossentropy.(map((x)->exp.(x),seq), y))
    l2 = sum(clamp.(enc_x'enc_y .- sum(enc_x .* enc_y, dims=1)' .+ 1, 0.0, floatmax())) / (size(enc_x)[2] ^ 2)
    Flux.reset!(dec)
    Flux.reset!(enc)

    println("crossentropy + pair-wise loss = ", l1+l2)
    return l1+l2
end

# function eval(x, y)
#     seq = pgen(x)
#     Flux.reset!(pgen)

#     enc_x = pgen.enc.(pgen.emb.(x))[end]
#     enc_y = pgen.enc.(pgen.emb.(y))[end]

#     l1 = sum(Flux.crossentropy.(map((x)->exp.(x),(seq)), y))
#     l2 = sum(clamp.(enc_x'enc_y .- sum(enc_x .* enc_y, dims=1)' .+ 1, 0.0, floatmax())) / (size(enc_x)[2] ^ 2)
    
#     println("eval_loss ", l1+l2)
#     return l1+l2
# end

batch_size = 10
train_data_len = 20
val_data_len =  30

X_train = process_data(ques_train_x, train_data_len, batch_size, n_words)|>gpu
Y_train = process_data(ques_train_y, train_data_len, batch_size, n_words)|>gpu
X_val = process_data(ques_test_x, val_data_len, batch_size, n_words)|>gpu
Y_val = process_data(ques_test_y, val_data_len, batch_size, n_words)|>gpu

Flux.train!(Flux.params(enc, dec, emb), zip(X_train, Y_train), opt) do x, y
    # x = emb.(x)
    # for i in 1:(size(x)[1]-1)
    #     enc(x[i])
    # end
    # enc_x = enc(x[end])
    # y_ = [y[i] for i in 1:(length(y)-1)]
    # enc_x_y_ = [[enc_x]; emb.(y_)]
    # ŷ = [dec(token) for token in enc_x_y_]
    # l1 = sum(Flux.crossentropy.(map((x)->exp.(x),(ŷ)), y))
    # println("l1 ", l1)
    # Flux.reset!(enc)
    # Flux.reset!(dec)

    # l2 = sum(clamp.(enc_x'enc_y .- sum(enc_x .* enc_y, dims=1)' .+ 1, 0.0, floatmax())) / (size(enc_x)[2] ^ 2)
    return loss(x, y)
end
