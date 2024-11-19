using Printf;
using Flux;
using Statistics;
using Test;

"""
Compute metrics for a Flux model using confusion matrix and optional loss.

# Arguments
- `model::Flux.Chain`: Trained Flux model
- `inputs::AbstractArray{<:Real,2}`: Input features
- `targets::AbstractArray{Bool,2}`: One-hot encoded target labels
- `loss_fn::Union{Function, Nothing}=nothing`: Optional loss function to compute model loss

# Returns
- `Dict{Symbol, Any}`: Dictionary containing confusion matrix metrics and optional loss
    - Includes metrics such as `:accuracy`, `:errorRate`, `:recall`, etc.
    - Optionally includes `:loss` if loss function is provided
"""
function confusionMatrix(
    model::Flux.Chain, 
    inputs::AbstractArray{<:Real,2}, 
    targets::AbstractArray{Bool,2}, 
    loss_fn::Union{Function, Nothing}=nothing
)
    # Compute model predictions
    predictions = model(inputs')
    
    # Compute confusion matrix
    conf_matrix = confusionMatrix(
        predictions', 
        targets, 
        weighted=true
    )
    
    # Conditionally add loss if loss function is provided
    if loss_fn !== nothing
        model_loss = loss_fn(model, inputs', targets')
        conf_matrix[:loss] = model_loss
    end
    
    return conf_matrix
end

"""
Train a Multi-class Classification Artificial Neural Network with metrics tracking and early 
stopping criteria.

# Arguments
- `topology::AbstractArray{<:Int,1}`: Network layer topology defining neurons in each layer
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}`: Tuple of 
    training inputs and targets
- `validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional 
    validation dataset
- `testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional test 
    dataset
- `transferFunctions::AbstractArray{<:Function,1}=σ`: Transfer functions for each layer
- `maxEpochs::Int=1000`: Maximum number of training epochs
- `minLoss::Real=0.0`: Minimum loss threshold to stop training
- `learningRate::Real=0.01`: Learning rate for the optimizer
- `maxEpochsVal::Int=20`: Maximum epochs without improvement before early stopping
- `verbose::Bool=false`: Flag to print training progress
- `printFrequency::Int=10`: The number of iterations between prints.

# Returns
- `Tuple{Flux.Chain, Dict{Symbol, Vector{Dict{Symbol, Any}}}}`: 
    - Best trained neural network model
    - Dictionary of metrics for training, validation, and test datasets. Each entry contains a 
        vector of metrics for each epoch
"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    maxEpochsVal::Int=20, 
    verbose::Bool=false,
    printFrequency::Int=10
)
    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    # Ensure input sets and their corresponding targets have the same number of entries.
    @assert(size(train_inputs, 1) == size(train_targets, 1))
    @assert(size(val_inputs, 1) == size(val_targets, 1))
    @assert(size(test_inputs, 1) == size(test_targets, 1))

    # Ensure all targets have the same number of columns.
    @assert(size(train_targets, 2) == size(test_targets, 2))
    if (!isempty(val_inputs))
        @assert(size(train_targets, 2) == size(val_targets, 2))
    end

    # Define the loss function
    loss(model, x, y) = (size(y, 1) == 1) ? 
        Flux.Losses.binarycrossentropy(model(x), y) : 
        Flux.Losses.crossentropy(model(x), y)

    # We define the ANN
    ann = buildClassANN(
        size(train_inputs, 2), 
        topology, 
        size(train_targets, 2),
        transferFunctions=transferFunctions
    )
    
    # Metrics storage using a dictionary 
    metrics = Dict(
        :training => Vector{Dict{Symbol, Any}}(),
        :validation => Vector{Dict{Symbol, Any}}(),
        :test => Vector{Dict{Symbol, Any}}()
    )

    # Initialize the best ANN
    minValidationLoss = Inf
    bestAnn = deepcopy(ann)

    # Initialize counters
    numEpoch = 0
    numEpochsValidation = 0

    # Initial metrics computation
    push!(metrics[:training], confusionMatrix(ann, train_inputs, train_targets, loss))
    if !isempty(val_inputs)
        push!(metrics[:validation], confusionMatrix(ann, val_inputs, val_targets, loss))
    end
    if !isempty(test_inputs)
        push!(metrics[:test], confusionMatrix(ann, test_inputs, test_targets, loss))
    end

    # Optional initial output
    if verbose
        println("Epoch 0: Initial metrics computed")
    end

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Start the training until it reaches one of the stop criteria
    while (numEpoch < maxEpochs) && 
        (metrics[:training][end][:loss] > minLoss) && 
        (isempty(val_inputs) || (numEpochsValidation < maxEpochsVal))

        # Train a single iteration
        Flux.train!(loss, ann, [(train_inputs', train_targets')], opt_state)

        numEpoch += 1
        numEpochsValidation += 1

        # Compute and store training metrics
        push!(metrics[:training], confusionMatrix(ann, train_inputs, train_targets, loss))

        # Validation metrics and early stopping criteria with accuracy
        if (!isempty(val_inputs))
            val_metrics = confusionMatrix(ann, val_inputs, val_targets, loss)
            push!(metrics[:validation], val_metrics)

            # Early stopping and best model selection
            if val_metrics[:accuracy] > metrics[:validation][end-1][:accuracy]
                bestAnn = deepcopy(ann)
                minValidationLoss = val_metrics[:loss]
                numEpochsValidation = 0
            end
        end

        # Test metrics
        if (!isempty(test_inputs))
            push!(metrics[:test], confusionMatrix(ann, test_inputs, test_targets, loss))
        end

        # Optional verbose output
        if verbose && (numEpoch % printFrequency == 0 || numEpoch == 1)
            @printf("Epoch: %d\n", numEpoch)
            @printf("Training - Loss: %.4f, Accuracy: %.4f\n", 
                metrics[:training][end][:loss], 
                metrics[:training][end][:accuracy])
            
            if !isempty(val_inputs)
                @printf("Validation - Loss: %.4f, Accuracy: %.4f\n", 
                    metrics[:validation][end][:loss], 
                    metrics[:validation][end][:accuracy])
            end
            
            if !isempty(test_inputs)
                @printf("Test - Loss: %.4f, Accuracy: %.4f\n", 
                    metrics[:test][end][:loss], 
                    metrics[:test][end][:accuracy])
            end
        end
    end

    # Return results
    if !isempty(val_inputs)
        if verbose
            println("Returning best ANN from epoch ", numEpoch - maxEpochsVal)
        end
        return (bestAnn, metrics)
    else
        if verbose
            println("Returning last ANN")
        end
        return (ann, metrics)
    end
end;

"""
Train a Binary Classification Artificial Neural Network with metrics tracking and early 
stopping criteria.

# Arguments
- `topology::AbstractArray{<:Int,1}`: Network layer topology defining neurons in each layer
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}`: Tuple of 
    training inputs and targets
- `validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional 
    validation dataset
- `testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=<empty>`: Optional test 
    dataset
- `transferFunctions::AbstractArray{<:Function,1}=σ`: Transfer functions for each layer
- `maxEpochs::Int=1000`: Maximum number of training epochs
- `minLoss::Real=0.0`: Minimum loss threshold to stop training
- `learningRate::Real=0.01`: Learning rate for the optimizer
- `maxEpochsVal::Int=20`: Maximum epochs without improvement before early stopping
- `verbose::Bool=false`: Flag to print training progress
- `printFrequency::Int=10`: The number of iterations between prints.

# Returns
- `Tuple{Flux.Chain, Dict{Symbol, Vector{Dict{Symbol, Any}}}}`: 
    - Best trained neural network model
    - Dictionary of metrics for training, validation, and test datasets. Each entry contains a 
        vector of metrics for each epoch
"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    maxEpochsVal::Int=20, 
    verbose::Bool=false,
    printFrequency::Int=10
)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )
    val_targets = hcat(
        val_targets,   # First column for true values
        .!val_targets  # Second column for false values
    )
    test_targets = hcat(
        test_targets,   # First column for true values
        .!test_targets  # Second column for false values
    )

    return trainClassANN(
        topology,
        (train_inputs, train_targets);
        validationDataset=(val_inputs, val_targets),
        testDataset=(test_inputs, test_targets),
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal,
        verbose=verbose,
        printFrequency=printFrequency
    )
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1,
    validationRatio::Real=0.0, maxEpochsVal::Int=20)

    (train_inputs, train_targets) = trainingDataset

    # Metric results (train loss, validation loss, and test loss) for each of the k folds.
    k_folds = maximum(kFoldIndices)
    metric_results = Array{Float64,2}(undef, k_folds, 3)

    for k in 1:k_folds
        # Get the inputs and targets specific for this k-fold.
        k_test_mask = kFoldIndices .== k
        k_test_inputs = train_inputs[k_test_mask, :]
        k_test_targets = train_targets[k_test_mask, :]

        # Get validation and train sets for this k-fold.
        # TODO Adjust validationRation
        # TODO only do this if validationRatio
        train_indices, val_indices = holdOut(sum(.!k_test_mask), validationRatio)
        k_train_inputs = train_inputs[.!k_test_mask, :][train_indices, :]
        k_train_targets = train_targets[.!k_test_mask, :][train_indices, :]
        k_val_inputs = train_inputs[.!k_test_mask, :][val_indices, :]
        k_val_targets = train_targets[.!k_test_mask, :][val_indices, :]

        averageTrainLoss = 0
        averageValLoss = 0
        averageTestLoss = 0
        for rep in 1:repetitionsTraining
            ann, trainLosses, valLosses, testLosses = trainClassANN(topology,
                (k_train_inputs, k_train_targets),
                validationDataset=(k_val_inputs, k_val_targets),
                testDataset=(k_test_inputs, k_test_targets),
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs,
                minLoss=minLoss,
                learningRate=learningRate,
                maxEpochsVal=maxEpochsVal)

            @printf("Fold %d, repetition %d. \t train loss: %.4f, val loss: %.4f, test loss: %.4f\n",
                k, rep,
                trainLosses[end],
                valLosses[end],
                testLosses[end])

            averageTrainLoss += !isempty(trainLosses) ? trainLosses[end] : 0
            averageValLoss += !isempty(valLosses) ? valLosses[end] : 0
            averageTestLoss += !isempty(testLosses) ? testLosses[end] : 0
        end

        metric_results[k, 1] = averageTrainLoss / repetitionsTraining
        metric_results[k, 2] = averageValLoss / repetitionsTraining
        metric_results[k, 3] = averageTestLoss / repetitionsTraining
    end

    average_metrics = mean(metric_results, dims=1)
    std_dev_metrics = std(metric_results, dims=1)

    return average_metrics, std_dev_metrics
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}},
    kFoldIndices::Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1,
    validationRatio::Real=0.0, maxEpochsVal::Int=20)
    (train_inputs, train_targets) = trainingDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )

    return trainClassANN(
        topology,
        (train_inputs, train_targets),
        kFoldIndices,
        transferFunctions,
        maxEpochs,
        minLoss,
        learningRate,
        repetitionsTraining,
        validationRatio,
        maxEpochsVal
    )
end;
