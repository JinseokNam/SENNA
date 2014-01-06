function [embeddings, convolution_weights, convolution_bias, hidden_weights, hidden_bias, output_weights, trans_probs] = extractParameters(theta, netconfig)

numEmbeddings = numel(netconfig.embeddingSizes);
embeddings = cell(numEmbeddings,1);

totalEmbeddingSize = 0;
paramStartIndex = 0;
for e=1:numEmbeddings
    embeddedSize = netconfig.embeddingSizes{e}(1);
    embeddingInputSize = netconfig.embeddingSizes{e}(2);
    embeddings{e} = reshape(theta(paramStartIndex+1:paramStartIndex+embeddedSize*embeddingInputSize),embeddedSize, embeddingInputSize);
    paramStartIndex = paramStartIndex + (embeddedSize*embeddingInputSize);
    totalEmbeddingSize = totalEmbeddingSize + embeddedSize;
end

convolution_weights = reshape(theta(paramStartIndex+1:paramStartIndex+netconfig.convolutionSize*totalEmbeddingSize*netconfig.windowSize), ...
							netconfig.convolutionSize, totalEmbeddingSize*netconfig.windowSize);
paramStartIndex = paramStartIndex+ netconfig.convolutionSize*(totalEmbeddingSize*netconfig.windowSize);
convolution_bias = theta(paramStartIndex+1:paramStartIndex+netconfig.convolutionSize);
paramStartIndex = paramStartIndex + netconfig.convolutionSize;
hidden_weights = reshape(theta(paramStartIndex+1:paramStartIndex+netconfig.hiddenSize*netconfig.convolutionSize), ...
                    netconfig.hiddenSize, netconfig.convolutionSize);
paramStartIndex = paramStartIndex+ netconfig.hiddenSize * netconfig.convolutionSize;
hidden_bias = theta(paramStartIndex+1:paramStartIndex+netconfig.hiddenSize);
paramStartIndex = paramStartIndex + netconfig.hiddenSize;

output_weights = reshape(theta(paramStartIndex+1:paramStartIndex+(netconfig.outputSize*netconfig.hiddenSize)), netconfig.outputSize, netconfig.hiddenSize);
paramStartIndex = paramStartIndex + (netconfig.outputSize*netconfig.hiddenSize);
trans_probs = reshape(theta(paramStartIndex+1:paramStartIndex+(netconfig.outputSize*(netconfig.outputSize+1))), netconfig.outputSize, netconfig.outputSize+1);
end
