function oneHotEncoding(data::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    encoded = zeros(Bool, length(data), length(classes))

    for (i, value) in enumerate(data)
        class_index = findfirst(==(value), classes)
        encoded[i, class_index] = 1
    end

    return encoded
end;