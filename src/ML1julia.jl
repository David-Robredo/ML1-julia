module ML1julia
    export buildClassANN, trainClassANN, crossvalidation, holdOut, confusionMatrix, 
            printConfusionMatrix, accuracy, oneHotEncoding

    include("utils/build.jl")
    include("utils/holdOut.jl")
    include("utils/metrics.jl")
    include("utils/normalization.jl")
    include("utils/oneHot.jl")
    include("utils/train.jl")
end