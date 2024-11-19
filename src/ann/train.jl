using Printf;
using Flux;
using Statistics;
using Test;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    # Ensure input sets and their corresponding targets have the same number of entries.
    @assert(size(train_inputs, 1) == size(train_targets, 1))
    @assert(size(val_inputs, 1) == size(val_targets, 1))
    @assert(size(test_inputs, 1) == size(test_targets, 1))

    # Ensure all targets have the same number of columns.
    @assert(size(train_targets, 2) == size(test_targets, 1))
    if (!isempty(val_inputs))
        @assert(size(train_targets, 2) == size(val_targets, 1))
    end

    # We define the ANN
    ann = buildClassANN(
        size(train_inputs, 2), 
        topology, 
        size(train_targets, 2),
        transferFunctions
    )

    # Setting up the loss funtion to reduce the error
    loss(model, x, y) = (size(y, 1) == 1) ? Flux.Losses.binarycrossentropy(model(x), y) : Flux.Losses.crossentropy(model(x), y)

    # This vectos is going to contain the losses and precission on each training epoch
    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]
    minValidationLoss = Inf
    bestAnn = deepcopy(ann)

    # Inicialize the counter to 0
    numEpoch = 0
    numEpochsValidation = 0
    # Calcualte the loss without training
    trainingLoss = loss(ann, train_inputs', train_targets')
    #  Store this one for checking the evolution.
    push!(trainingLosses, trainingLoss)
    #  and give some feedback on the screen
    println("Epoch ", numEpoch, ": loss: ", trainingLoss)

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Start the training until it reaches one of the stop critteria
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)

        # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
        Flux.train!(loss, ann, [(train_inputs', train_targets')], opt_state)

        numEpoch += 1
        numEpochsValidation += 1

        # calculate the loss for this epoch
        trainingLoss = loss(ann, train_inputs', train_targets')
        # store it
        push!(trainingLosses, trainingLoss)

        if (!isempty(val_inputs))
            validationLoss = loss(ann, val_inputs', val_targets')
            push!(validationLosses, validationLoss)

            if validationLoss < minValidationLoss
                bestAnn = deepcopy(ann)
                minValidationLoss = validationLoss
                numEpochsValidation = 0
            end
        end

        testLoss = loss(ann, test_inputs', test_targets')
        push!(testLosses, testLoss)

        @printf("Epoch: %d \t train loss: %.4f, val loss: %.4f, test loss: %.4f\n",
            numEpoch,
            trainingLoss,
            validationLoss,
            testLoss)
    end

    if validationDataset != nothing
        println("Returning ANN from epoch ", numEpoch - maxEpochsVal)
        return (bestAnn, trainingLosses, validationLosses, testLosses)
    else
        println("Returning last ANN")
        return (ann, trainingLosses, validationLosses, testLosses)
    end
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

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
        (train_inputs, train_targets),
        (val_inputs, val_targets),
        (test_inputs, test_targets),
        transferFunctions,
        maxEpochs,
        minLoss,
        learningRate,
        maxEpochsVal,
        showText
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
