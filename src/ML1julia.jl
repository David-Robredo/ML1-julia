module ML1julia
    export buildClassANN, trainClassANN, crossvalidation, holdOut, confusionMatrix, 
            printConfusionMatrix, accuracy, oneHotEncoding, normalize, compute_μσ,
            crossvalidation

    include("utils/holdOut.jl")
    include("utils/metrics.jl")
    include("utils/normalization.jl")
    include("utils/oneHot.jl")
    include("utils/crossvalidation.jl")
    include("ann/build.jl")
    include("ann/train.jl")
    include("scikit/train.jl")
    include("scikit/ensemble.jl")
end