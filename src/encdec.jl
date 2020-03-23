using Flux

struct EncDec
    seq_len::Int
    emb
    enc
    dec
end

EncDec(in::Int, emb_dim::Int, rnn_hid_dim::Int, seq_len::Int) = EncDec(seq_len, Dense(in, emb_dim), LSTM(emb_dim, emb_dim), Chain(LSTM(emb_dim, rnn_hid_dim),Dense(rnn_hid_dim, in), logsoftmax))

function (m::EncDec)(x, y)
    y_ = [y[i] for i in 2:length(y)]
    m.dec.([[m.enc.(m.emb.(x))[end]]; m.emb.(y_)])
end

function (m::EncDec)(x)
    seq = []
    push!(seq, m.dec(m.enc.(m.emb.(x))[end]))
    for _ in 2:m.seq_len
        push!(seq, m.dec(m.emb(map((x)->exp.(x), seq[end]))))
    end
    return seq
end

Flux.@functor EncDec