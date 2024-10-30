function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs

    # Build hidden layers
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer]
        activation = transferFunctions[numHiddenLayer]
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, activation))
        numInputsLayer = numNeurons
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end

    return ann
end;