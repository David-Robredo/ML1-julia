using Flux;

"""
Builds a Flux-based feedforward neural network for classification tasks.

# Arguments
- `numInputs::Int`: The number of input features.
- `topology::AbstractArray{<:Int,1}`: An array specifying the number of neurons 
    in each hidden layer.
- `numOutputs::Int`: The number of output classes.
- `transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))`: An 
    array of activation functions for each hidden layer, defaults to 
    sigmoid (`σ`) for all layers.

# Returns
- `Flux.Chain`: A Flux neural network model.
"""
function buildClassANN(
    numInputs::Int,
    topology::AbstractArray{<:Int,1},
    numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))
)
    ann = Chain()
    numInputsLayer = numInputs

    # Build hidden layers
    for i in eachindex(topology)
        numNeurons = topology[i]
        activation = transferFunctions[i]
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, activation))
        numInputsLayer = numNeurons
    end

    # If using just a single output, apply sigmoid. Otherwise, apply softmax.
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end

    return ann
end;