using Test;
using ScikitLearn;

@sk_import ensemble: VotingClassifier
@sk_import ensemble: StackingClassifier

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier


function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:Dict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1};
    verbose::Bool=false,
    metric::Symbol=:f1Score
)

    (train_inputs, train_targets) = trainingDataset

end