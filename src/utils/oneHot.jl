"""
Perform one-hot encoding on a 1D array of data.

# Arguments
- `data::AbstractArray{<:Any,1}`: The input 1D array of data to be encoded.
- `classes::AbstractArray{<:Any,1}`: The array of unique classes or categories 
    present in the input data.

# Returns
- `::Matrix{Bool}`: A boolean matrix where each row represents an input data 
    point, and each column represents a unique class. The value at each position 
    is 1 if the data point belongs to the corresponding class, and 0 otherwise.
"""
function oneHotEncoding(data::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    encoded = zeros(Bool, length(data), length(classes))

    for (i, value) in enumerate(data)
        class_index = findfirst(==(value), classes)
        encoded[i, class_index] = 1
    end

    return encoded
end;