PRECISION = 'double';

wordVocabSize = 13;	% two padding + one empty
wordEmbeddingSize = 5;
capsVocabSize = 3;
capsEmbeddingSize = 2;
relTagPosVocabSize = 3;
relTagPosEmbeddingSize = 2;
convolutionSize = 10;
hiddenSize = 30;
outputSize = 5;
windowSize = 5;
%{
wordVocabSize = 100003;	% two padding + one empty
wordEmbeddingSize = 50;
capsVocabSize = 3;
capsEmbeddingSize = 2;
relTagPosVocabSize = 3;
relTagPosEmbeddingSize = 2;
convolutionSize = 100;
hiddenSize = 300;
outputSize = 50;
windowSize = 5;
%}
wordEmbedding = 0.01*randn(wordEmbeddingSize, wordVocabSize, PRECISION);
capsEmbedding = 0.01*randn(capsEmbeddingSize, capsVocabSize, PRECISION);
relTagPosEmbedding = 0.01*randn(relTagPosEmbeddingSize, relTagPosVocabSize, PRECISION);
%{
wordEmbedding = 0.01*randn(wordEmbeddingSize, wordVocabSize, PRECISION);
capsEmbedding = 0.01*randn(capsEmbeddingSize, capsVocabSize, PRECISION);
relTagPosEmbedding = 0.01*randn(relTagPosEmbeddingSize, relTagPosVocabSize, PRECISION);
%}
%embedding(:,[1 12 13]) = 0;
totalEmbeddingSize = wordEmbeddingSize+capsEmbeddingSize+relTagPosEmbeddingSize;

convolution_weights = 1/sqrt(totalEmbeddingSize*windowSize)*randn(convolutionSize, totalEmbeddingSize*windowSize, PRECISION);
convolution_bias = zeros(convolutionSize,1, PRECISION);
convolution_to_hidden_weights = 1/sqrt(convolutionSize)*randn(hiddenSize, convolutionSize, PRECISION);
convolution_to_hidden_bias = zeros(hiddenSize,1, PRECISION);
output_weights = 1/sqrt(hiddenSize)*randn(outputSize,hiddenSize, PRECISION);
start_trans_probs = 1/outputSize*ones(outputSize,1, PRECISION);
trans_probs = 0.01*rand(outputSize,outputSize, PRECISION);
trans_probs = [start_trans_probs trans_probs];

theta = [embCell2params({wordEmbedding,capsEmbedding,relTagPosEmbedding}) ; convolution_weights(:) ; convolution_bias(:) ; convolution_to_hidden_weights(:) ; convolution_to_hidden_bias(:) ; output_weights(:) ; trans_probs(:)];
modelOptions.netconfig.embeddingSizes = {[wordEmbeddingSize, wordVocabSize], ...
										[capsEmbeddingSize, capsVocabSize], ...
										[relTagPosEmbeddingSize, relTagPosVocabSize] ...
										};
modelOptions.netconfig.windowSize = windowSize;
modelOptions.netconfig.convolutionSize = convolutionSize;
modelOptions.netconfig.hiddenSize = hiddenSize;
modelOptions.netconfig.outputSize = outputSize;
modelOptions.sll = true;       % an option for log-likelihood
modelOptions.useMex = false;

% read sentence 
sentence_length = 10;
word_index = randi(wordVocabSize, [sentence_length,1]);
caps_index = randi(capsVocabSize, [sentence_length,1]);

% add padding
start_padding = 1;
end_padding = wordVocabSize;
word_index = [repmat(start_padding, (windowSize-1)/2,1); word_index; repmat(end_padding,(windowSize-1)/2,1)];

padding = 1;
caps_index = [repmat(padding,(windowSize-1)/2,1); caps_index; repmat(padding,(windowSize-1)/2,1)];

rel_tag_index = randi(3, [length(word_index), sentence_length]);
num_to_tag = size(rel_tag_index,2);
word_index = repmat(word_index, 1, num_to_tag);
caps_index = repmat(caps_index, 1, num_to_tag);

data = {word_index,caps_index,rel_tag_index};
target = randi(outputSize, [sentence_length,1]);

assert(numel(target)==num_to_tag);

costFunc = @(p) sennaCost(p, data, target, modelOptions);
[~,grad] = costFunc(theta);
%{
for i=1:1000
	[cost,grad] = costFunc(theta);
	if rem(i,100)==0
		fprintf('[%5d] %s: %f\n', i, getTime, cost);
	end	
	theta = theta - 0.1.*grad;
end
%}


% Use your code to numerically compute the gradient of simpleQuadraticFunction at x.
numgrad = computeNumericalGradient(costFunc, theta);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([(1:length(numgrad))'./1000 numgrad grad numgrad./grad]);
%disp([(1:length(numgrad(1:38)))'./1000 numgrad(1:38) grad(1:38) numgrad(1:38)./grad(1:38)]);
fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be 2.1452e-12 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
