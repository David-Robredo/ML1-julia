using Test;
using ScikitLearn;
using PyCall;

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier

"""
Compute metrics for a ScikitLearn model using confusion matrix.

# Arguments
- `model::PyCall.PyObject`: A trained scikit-learn or compatible machine learning model
- `inputs::AbstractArray{<:Real,2}`: Input feature matrix with observations as rows
- `targets::AbstractArray{Bool,2}`: True target labels for the input data

# Returns
- `Dict{Symbol, Any}`: Dictionary containing confusion matrix metrics, including metrics such 
    as `:accuracy`, `:errorRate`, `:recall`, etc.
"""
function confusionMatrix(
    model::PyCall.PyObject, 
    inputs::AbstractArray{<:Real,2}, 
    targets::AbstractArray{Bool,2},
)
    # Compute model predictions
    predictions = predict(model, inputs)
    
    # Compute confusion matrix
    conf_matrix = confusionMatrix(
        predictions, 
        targets
    )
    
    return conf_matrix
end

"""
Create a scikit-learn machine learning model based on the specified model type and hyperparameters.

# Arguments
- `modelType::Symbol`: The type of machine learning model to create. 
  Supported types include:
  - `:MLP`: Multi-layer Perceptron (Neural Network Classifier)
  - `:SVM`: Support Vector Machine Classifier
  - `:DecisionTree`: Decision Tree Classifier
  - `:kNN`: K-Nearest Neighbors Classifier

- `modelHyperparameters::Dict`: A dictionary of hyperparameters specific to the chosen model type

# Returns
- `PyCall.PyObject`: A scikit-learn compatible machine learning model instance
"""
function createScikitLearnModel(
    modelType::Symbol,
    modelHyperparameters::Dict,
)
    if modelType == :MLP
        params = Dict{Symbol, Any}()
        params[:early_stopping] = true
        if haskey(modelHyperparameters, :topology)
            params[:hidden_layer_sizes] = modelHyperparameters[:topology]
        end
        if haskey(modelHyperparameters, :maxEpochs)
            params[:max_iter] = modelHyperparameters[:maxEpochs]
        end
        if haskey(modelHyperparameters, :learningRate)
            params[:learning_rate_init] = modelHyperparameters[:learningRate]
        end
        if haskey(modelHyperparameters, :maxEpochsVal)
            params[:n_iter_no_change] = modelHyperparameters[:maxEpochsVal]
        end
        if haskey(modelHyperparameters, :validationRatio)
            params[:validation_fraction] = modelHyperparameters[:validationRatio]
        end
        return MLPClassifier(; params...)
    elseif modelType == :SVM
        return  SVC(; modelHyperparameters...)
    elseif modelType == :DecisionTree
        return  DecisionTreeClassifier(; modelHyperparameters...)
    elseif modelType == :kNN
        return  KNeighborsClassifier(; modelHyperparameters...)
    else
        error("Unsupported model type: $modelType")
    end
end

"""
Perform Multi-Class Classification Cross-Validation for a machine learning model with 
multiple potential model-types.

# Arguments
- `modelType::Symbol`: Type of model to train (e.g., :MLP, :SVM, :DecisionTree, :kNN, :ANN)
- `modelHyperparameters::Dict`: Dictionary of hyperparameters for model configuration
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}`: Tuple containing 
    input features and corresponding target labels.
- `crossValidationIndices::Array{Int64,1}`: Indices defining the cross-validation folds
- `verbose::Bool=false`: If true, print detailed information during cross-validation
- `metric::Symbol=:f1Score`: Metric to use for model selection and evaluation

# Returns
- `Tuple{Union{Nothing, Any}, Union{Nothing, Dict}, Dict}`: 
    - Best trained neural network model across all folds
    - Best model's metrics 
    - Summary metrics across all folds (mean, max, min, std)
"""
function modelCrossValidation(
    modelType::Symbol,
    modelHyperparameters::Dict,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    crossValidationIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score
)

    (train_inputs, train_targets) = trainingDataset

    # Check if the model type is an ANN. If it is, call our own trainClassANN
    # directly instead of performing the k-fold cross-validation manually.
    if modelType == :ANN 
        # Train ANN model with our own training function with Flux
        return return trainClassANN(
            modelHyperparameters[:topology],
            (train_inputs, encoded_targets),
            crossValidationIndices;
            maxEpochs=modelHyperparameters[:maxEpochs],
            minLoss=modelHyperparameters[:minLoss],
            learningRate=modelHyperparameters[:learningRate],
            repetitionsTraining=modelHyperparameters[:repetitions],
            validationRatio=modelHyperparameters[:validation_ratio],
            maxEpochsVal=modelHyperparameters[:maxEpochsVal],
            verbose=verbose,
            metric=metric
        )
    end

    # Metric results (train loss, validation loss, and test loss) for each of the k folds.
    fold_results = Vector{Dict{Symbol, Any}}()
    
    # Overall best model across all folds.
    best_overall_model = nothing
    best_overall_metrics = nothing
    best_overall_score = -Inf

    k_folds = maximum(crossValidationIndices)
    for k in 1:k_folds
        # Get the inputs and targets specific for this k-fold.
        k_test_mask = crossValidationIndices .== k
        k_test_inputs = train_inputs[k_test_mask, :]
        k_test_targets = train_targets[k_test_mask, :]

        # Prepare train-set for this k-fold.
        k_train_inputs = train_inputs[.!k_test_mask, :]
        k_train_targets = train_targets[.!k_test_mask, :]

        # Create appropriate model based on modelType
        model = createScikitLearnModel(modelType, modelHyperparameters)

        # Train model on the train set for this fold
        fit!(model, k_train_inputs, k_train_targets)

        # Test the model on both sets for keeping of metrics.
        metrics = Dict()
        metrics[:training] = confusionMatrix(model, k_train_inputs, k_train_targets)
        metrics[:test] = confusionMatrix(model, k_test_inputs, k_test_targets)

        # Register metrics for this fold.
        push!(fold_results, metrics)

        # Report metrics if needed
        if verbose
            @printf("Fold %d - Train - Loss: %.4f, Metric: %.4f\n", 
                k, rep,
                metrics[:training][:loss], 
                metrics[:training][metric])
            
            if !isempty(k_test_inputs)
                @printf("Fold %d - Test - Loss: %.4f, Metric: %.4f\n", 
                    k, rep,
                    metrics[:test][:loss], 
                    metrics[:test][metric])
            end
        end

        # Update overall best model on the test set.
        if metrics[:test][metric] > best_overall_score
            best_overall_model = model
            best_overall_metrics = metrics
            best_overall_score = metrics[:test][metric]
        end
    end

    # Initialize resume_metrics of all folds
    resume_metrics = Dict()
    # Go through each subset (training, validation, test)
    for subset in [:training, :validation, :test]
        # Skip if no fold results for this subset
        subset_results = [fold[subset] for fold in fold_results if haskey(fold, subset)]
        if !isempty(subset_results)
            # Initialize metrics for this subset
            resume_metrics[subset] = Dict()
            
            # Compute mean, max, min for each metric
            for metric in keys(subset_results[1])
                # Collect all values for this metric across folds
                metric_values = [result[metric] for result in subset_results]
                # Compute mean, maximum, etc only if the metrics are numbers.
                if all(x -> isa(x, Real), metric_values)
                    resume_metrics[subset][metric] = Dict(
                        :mean => mean(metric_values),
                        :max => maximum(metric_values),
                        :min => minimum(metric_values),
                        :std => std(metric_values)
                    )
                end
            end
        end
    end

    return best_overall_model, best_overall_metrics, resume_metrics
end

"""
Perform Binary Classification Cross-Validation for a machine learning model with 
multiple potential model-types.

# Arguments
- `modelType::Symbol`: Type of model to train (e.g., :MLP, :SVM, :DecisionTree, :kNN, :ANN)
- `modelHyperparameters::Dict`: Dictionary of hyperparameters for model configuration
- `trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}`: Tuple containing 
    input features and corresponding target labels
- `crossValidationIndices::Array{Int64,1}`: Indices defining the cross-validation folds
- `verbose::Bool=false`: If true, print detailed information during cross-validation
- `metric::Symbol=:f1Score`: Metric to use for model selection and evaluation

# Returns
- `Tuple{Union{Nothing, Any}, Union{Nothing, Dict}, Dict}`: 
    - Best trained neural network model across all folds
    - Best model's metrics 
    - Summary metrics across all folds (mean, max, min, std)
"""
function modelCrossValidation(
    modelType::Symbol,
    modelHyperparameters::Dict,
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}},
    crossValidationIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score
)
    (train_inputs, train_targets) = trainingDataset

    train_targets = hcat(
        train_targets,   # First column for true values
        .!train_targets  # Second column for false values
    )

    return modelCrossValidation(
        modelType,
        modelHyperparameters,
        (train_inputs, train_targets),
        crossValidationIndices;
        verbose,
        metric
    )
end

@testset "modelCrossValidation" begin
    # Set seed for repetibility
    Random.seed!(42)
    
    n_samples = 200  # Total number of samples
    n_features = 4    # Number of input features

    # Generate dummy dataset. The target is based on 
    # whether the first column is greater than 0.5
    train_input  = rand(Float32, n_samples, n_features + 1)    
    train_output = vec(train_input[:, 1] .> 0.5)

    # Create indices for kfolds
    crossvalidation_indices = crossvalidation(train_output, 10)

    # Small test configuration
    knn_hyperparameters = Dict(
        :n_neighbors => 5,
        :weights => :uniform,
        :metric => :euclidean,
        :algorithm => :auto
    )

    # Train the small topology. The results should be
    # at least moderately good, or at least sligtly
    # better than random chance.
    bestAnn, bestAnnMetrics, metrics = modelCrossValidation(
        :kNN,
        knn_hyperparameters,
        (train_input, train_output),
        crossvalidation_indices;
    )

    @test metrics[:test][:accuracy][:mean] > 0.6
    @test metrics[:test][:errorRate][:mean] < 0.4
    @test metrics[:test][:recall][:mean] > 0.6
    @test metrics[:test][:specificity][:mean] > 0.6
    @test metrics[:test][:posPredValue][:mean] > 0.6
    @test metrics[:test][:negPredValue][:mean] > 0.6
    @test metrics[:test][:f1Score][:mean] > 0.6
    
end