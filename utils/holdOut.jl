using Random

"""
Splits a dataset of `N` samples into a training and test set, with the test set
containing `P` proportion of the total samples.

Args:
    N (Int): Total number of samples in the dataset.
    P (Real): Proportion of samples to be included in the test set.

Returns:
    Tuple{Vector{Int}, Vector{Int}}: A tuple containing the indices of the training
    and test sets.
"""
function holdOut(N::Int, P::Real)
    @assert(N > 0, "Number of samples (N) must be greater than 0.")
    @assert(0 <= P <= 1, "Proportion (P) must be between 0 and 1.")

    # Generate a random permutation of the indices
    indices = randperm(N)
    # Calculate the split index based on the given proportion
    split_index = Int(floor(N * P))
    # Split the indices into training and test sets
    train_idx, test_idx = indices[split_index+1:end], indices[1:split_index]
    return (train_idx, test_idx)
end

"""
Splits a dataset of `N` samples into training, validation, and test sets, with the
test set containing `Ptest` proportion of the total samples, and the validation set
containing `Pval` proportion of the remaining samples.

Args:
    N (Int): Total number of samples in the dataset.
    Pval (Real): Proportion of the training set to be used as the validation set.
    Ptest (Real): Proportion of samples to be included in the test set.

Returns:
    Tuple{Vector{Int}, Vector{Int}, Vector{Int}}: A tuple containing the indices of the
    training, validation, and test sets.
"""
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert(N > 0, "Number of samples (N) must be greater than 0.")
    @assert(0 <= (Pval + Ptest) <= 1, "The sum of validation and test proportions must be between 0 and 1.")

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