using Statistics;

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert(size(outputs) == size(targets))

    # True positive: both the output and the target were True.
    tps = sum(outputs .& targets)
    # True negative: both the output and the target were False.
    tns = sum((.~outputs) .& (.~targets))
    # False negative: the output was False, but the target was True.
    fns = sum((.~outputs) .& targets)
    # False positive: the output was True, but the target was False.
    fps = sum(outputs .& (.~targets))

    accuracy = (tns + tps) / (tns + tps + fns + fps)
    errorRate = (fps + fns) / (tns + tps + fns + fps)

    # TODO better way to check this. Maybe check if, for example,
    # everything is a true positive or true negstive with
    # tns==length(targets)
    if fns + tps > 0
        recall = tps / (fns + tps)
    elseif fps == 0
        recall = 1.0
    else
        recall = 0.0
    end
    if fps + tns > 0
        specificity = tns / (fps + tns)
    elseif fns == 0
        specificity = 1.0
    else
        specificity = 0.0
    end
    if tps + fps > 0
        posPredValue = tps / (tps + fps)
    elseif fns == 0
        posPredValue = 1.0
    else
        posPredValue = 0.0
    end
    if tns + fns > 0
        negPredValue = tns / (tns + fns)
    elseif fps == 0
        negPredValue = 1.0
    else
        negPredValue = 0.0
    end
    if posPredValue != 0.0 && recall != 0.0
        fScore = (2 * posPredValue * recall) / (posPredValue + recall)
    else
        fScore = 0.0
    end

    confMatrix = [
        tns fps;
        fns tps
    ]

    return (
        accuracy,
        errorRate,
        recall,
        specificity,
        posPredValue,
        negPredValue,
        fScore,
        confMatrix
    )
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (
        accuracy,
        errorRate,
        recall,
        specificity,
        posPredValue,
        negPredValue,
        fScore,
        confMatrix
    ) = confusionMatrix(outputs, targets)

    TN = confMatrix[1, 1]
    FP = confMatrix[1, 2]
    FN = confMatrix[2, 1]
    TP = confMatrix[2, 2]

    println("             Predicted")
    println("            Pos     Neg")
    println("Actual Pos  $TP      $FN")
    println("       Neg  $FP      $TN")
    println("------------------------")
    println("Accuracy: $accuracy")
    println("Error rate: $errorRate")
    println("Recall: $recall")
    println("Specificity: $specificity")
    println("Precision: $posPredValue")
    println("Negative Predictive Value: $negPredValue")
    println("F1-Score: $fScore")
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .> threshold
    return printConfusionMatrix(outputs, targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if (size(targets, 2) == 1)
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        return mean(all(targets .== outputs, dims=2))
    end
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs, 2) == size(targets, 2),
        "The number of columns 'outputs' and 'targets' must match.")
    @assert(size(outputs, 2) != 2,
        "Two column matrices are invalid in multi-class classification.")

    # Initialize metrics
    numClass = size(outputs, 2)
    sensitivity = zeros(Float64, numClass)
    specificity = zeros(Float64, numClass)
    ppv = zeros(Float64, numClass)
    npv = zeros(Float64, numClass)
    f1 = zeros(Float64, numClass)

    # Compute metrics
    # TODO online iterate classes with instances.
    for i in 1:numClass
        if any(targets[:, i])
            metrics = confusionMatrix(outputs[:, i], targets[:, i])
            _, _, sensitivity[i], specificity[i], ppv[i], npv[i], f1[i], _ = metrics
        end
    end

    # Initialize and populate matrix
    confusion_mat = zeros(Int, numClass, numClass)

    for i in 1:numClass
        for j in 1:numClass
            confusion_mat[i, j] = sum(targets[:, i] .& outputs[:, j])
        end
    end

    # Strategy
    if weighted
        # TODO don't divide by target. Divide by vector of instances for each class.
        # numInstancesPerClass = vec(sum(targets, dims=1)). Also avoid division when
        # there are no instances for a class.
        class_weights = sum(targets, dims=1) ./ size(targets, 1)
        agg_sensitivity = sum(sensitivity .* class_weights) ./ size(targets, 2)
        agg_specificity = sum(specificity .* class_weights) ./ size(targets, 2)
        agg_ppv = sum(ppv .* class_weights) ./ size(targets, 2)
        agg_npv = sum(npv .* class_weights) ./ size(targets, 2)
        agg_f1 = sum(f1 .* class_weights) ./ size(targets, 2)
    else
        # TODO dont use the mean function to take into account possible
        # classes without instances. Also, the same as the above.
        agg_sensitivity = mean(sensitivity)
        agg_specificity = mean(specificity)
        agg_ppv = mean(ppv)
        agg_npv = mean(npv)
        agg_f1 = mean(f1)
    end

    # accuracy and error
    accuracy_value = accuracy(outputs, targets)
    error_rate = 1.0 - accuracy_value

    return agg_sensitivity, agg_specificity, agg_ppv, agg_npv, agg_f1, accuracy_value, error_rate, confusion_mat
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    boolean_outputs = classifyOutputs(outputs)
    return confusionMatrix(boolean_outputs, targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))

    classes_targets = unique(targets)

    encoded_outputs = oneHotEncoding(outputs, classes_targets)
    encoded_targets = oneHotEncoding(targets, classes_targets)

    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end;