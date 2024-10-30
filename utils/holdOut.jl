using Random

function holdOut(N::Int, P::Real)
    @assert(N > 0)
    @assert(0 <= P <= 1)

    indices = randperm(N)
    split_index = Int(floor(N * P))
    train_idx, test_idx = indices[split_index+1:end], indices[1:split_index]
    return (train_idx, test_idx)
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert(N > 0)
    @assert(0 <= (Pval + Ptest) <= 1)

    # Separate N values into (1-Ptest, Ptest) elements.
    trainval_idx, test_idx = holdOut(N, Ptest)

    # Adjust Pval given trainval set is smaller than the whole set.
    Pval = (N / length(trainval_idx)) * Pval
    # Separate train set into propper training set and validation set.
    train_idx, val_idx = holdOut(length(trainval_idx), Pval)

    return (
        trainval_idx[train_idx],
        trainval_idx[val_idx],
        test_idx
    )
end