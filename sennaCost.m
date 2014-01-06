function [cost, grad] = sennaCost(theta, X, Y, modelOptions)

PRECISION = 'double';

[embeddings, convolution_weights, convolution_bias, convolution_to_hidden_weights, convolution_to_hidden_bias, output_weights, trans_probs] = ...
				extractParameters(theta, modelOptions.netconfig);
            
netconfig = modelOptions.netconfig;         % configuration of the network architecture

M = size(X{1},2);   % the number of instances
sLen = size(X{1},1);		% sentence length including padding terms

% From discrete values to the distributed representations

embedded = cell(numel(embeddings),1);
totalEmbeddingSize = 0;
for k=1:numel(embeddings)	% We have K discrete feature types
    embedded{k} = embeddings{k}(:,X{k}(:));
	totalEmbeddingSize = totalEmbeddingSize + size(embedded{k},1);
end
embedded = cell2mat(embedded);	% concatenate K distributed representations for a single word

%%
A = (netconfig.windowSize-1)/2+1:sLen-(netconfig.windowSize-1)/2;
B = -(netconfig.windowSize-1)/2:1:(netconfig.windowSize-1)/2;
[p,q] = meshgrid(A, B);
index_set = sum([p(:) q(:)],2); clear A B p q
index_set = repmat(index_set,1,M);
index_set = bsxfun(@plus, index_set, (0:M-1)*sLen);
extEmbedded = reshape(embedded(:,index_set(:)),netconfig.windowSize*totalEmbeddingSize,(sLen-(netconfig.windowSize-1))*M);

convolved = bsxfun(@plus, convolution_weights*extEmbedded, convolution_bias);
convolved = reshape(convolved, size(convolved,1), sLen-(netconfig.windowSize-1), M);

[global_features, max_indices] = max(convolved,[],2);
global_features = squeeze(global_features);

% copute non-linear activation of global features 
[hiddenOutput, deriv_hiddenOutput] = hiddenActFunc(bsxfun(@plus, convolution_to_hidden_weights*global_features, convolution_to_hidden_bias));

predictive_scores = output_weights*hiddenOutput;

if modelOptions.sll
    % sentence-level log-likelihood
	[cost, trans_probGrad, loss_func_deriv] = sentence_log_lik(trans_probs, predictive_scores, Y);
else
    % word-level log-likelihood
    [cost, loss_func_deriv] = word_log_lik(predictive_scores, Y);
    trans_probGrad = zeros(size(trans_probs), PRECISION);
end

if nargout > 1
    % Start to compute the gradients
	delta = loss_func_deriv;

	% compute gradient of softmax weights
	output_weightsGrad = delta*hiddenOutput';

	delta = (output_weights'*delta).*deriv_hiddenOutput;

	% compute gradient of hidden weights and bias
	convolution_to_hidden_weightsGrad = delta*global_features';
	convolution_to_hidden_biasGrad = sum(delta,2);

	delta = convolution_to_hidden_weights'*delta;

	% compute gradient of the convolution weights and bias
	errors = zeros(netconfig.convolutionSize, sLen-((netconfig.windowSize)-1), M);
	[i,j,v] = find(max_indices);
	errors(sub2ind(size(errors), i,v,j)) = 1;

	delta = bsxfun(@times, errors, reshape(delta, netconfig.convolutionSize,1,M));
	delta = reshape(delta, size(delta,1),[]);

	convolution_weightsGrad = delta * extEmbedded';
	convolution_biasGrad = sum(delta,2);

	embeddingsGrad = cell(numel(embeddings),1);
    prevEmbeddingSize = 0;
    for k=1:numel(embeddings)
		embeddingsGrad{k} = zeros(size(embeddings{k}), PRECISION);

        embeddingSize = netconfig.embeddingSizes{k}(1);
        expansion_matrix = speye(netconfig.embeddingSizes{k}(2));
		A = reshape((X{k}(index_set(:))),netconfig.windowSize*(sLen-(netconfig.windowSize-1)),[]);
        for w=1:netconfig.windowSize
            embeddingsGrad{k} = embeddingsGrad{k} + ...
                (convolution_weights(:,prevEmbeddingSize+(w-1)*(totalEmbeddingSize)+1:prevEmbeddingSize+(w-1)*totalEmbeddingSize+embeddingSize)'*delta)* expansion_matrix(reshape(A(w:netconfig.windowSize:size(A,1),:),1,[]),:);
        end
        prevEmbeddingSize = prevEmbeddingSize+embeddingSize;
    end

	grad = [ embCell2params(embeddingsGrad) ; convolution_weightsGrad(:) ; convolution_biasGrad(:) ; convolution_to_hidden_weightsGrad(:) ; convolution_to_hidden_biasGrad(:) ; output_weightsGrad(:) ; trans_probGrad(:) ];
end


end

function [fVal, dfVal] = hiddenActFunc(X)
%%
%   HardTanh
%
	PRECISION = 'double';

    fVal = X;
    fVal(X<-1) = -1;
    fVal(X>1) = 1;

	if nargout > 1
        dfVal = ones(size(X), PRECISION);
        dfVal(X<-1 | X>1) = 0;
	end
end

function [loglik, dF] = word_log_lik(S,Y)

	[L,T] = size(S);
    
    S = exp(bsxfun(@minus, S, max(S,[],1)));	% To prevent overflow, substract the maximum value of inputs to exponent from each input
    S = bsxfun(@rdivide, S, sum(S,1));
    
	groundTruth = full(sparse(Y, 1:T, 1, L, T));
	loglik = 1/T*(-groundTruth(:)'*log(S(:)));

    if nargout > 1
        dF = 1/T*(S-groundTruth);
    end
end

function [loglik, dA, dF] = sentence_log_lik(A, S, Y)
	PRECISION = 'double';

    A0 = A(:,1);        % seperate starting states from transition scores matrix
    A(:,1) = [];
    
	T = numel(Y);       % time, or the length of the sentence
	N = size(A,1);      % cardinality of transition scores matrix
	delta = zeros(N,T, PRECISION);
    
	% init
    delta(:,1) = S(:,1) + A0;
    score_tag = A0(Y(1)) + S(Y(1),1);
    
    % forward recursion
    for t=2:T        
        score_tag = score_tag + A(Y(t-1), Y(t)) + S(Y(t),t);
		delta(:,t) = S(:,t) + logsumexp(bsxfun(@plus, A, delta(:,t-1)))';
    end
	clear t
    
    % Maximizing Equation (13) in [1] is equivalent to
    % minimizing negation of the Equation.
    % We would like to minimize the negation of the Equation (13) as if an error
    
	%loglik = logsumexp(delta(:,T)) - score_tag;
	loglik = 1/T*(logsumexp(delta(:,T)) - score_tag);
    
    if nargout > 1
        % Now, we will compute gradients of negative log-likelihood (loglik) obtained above with respect to inputs F and
        % transition scores matrix A

        % initialize gradients to zero
        % These will be updated over time
        dA = zeros(size(A), PRECISION);		% gradients with respect to transition matrix
        dA0 = zeros(size(A0), PRECISION);
        dF = zeros(size(S), PRECISION); 

        % compute initial partial derivative at time T
        deriv_Clogadd = exp(delta(:,T)-max(delta(:,T)))/sum(exp(delta(:,T)-max(delta(:,T))),1);

        % backward recursion
        for t=T:-1:2
            % update gradients of second part in Equation (13)
            dA(Y(t-1),Y(t)) = dA(Y(t-1),Y(t)) - 1;
            dF(Y(t),t) = dF(Y(t),t) - 1;

			% compute gradients wrt inputs
			dF(:,t) = dF(:,t) + deriv_Clogadd;

            % compute and add partial derivatives wrt transition scores
			ii = bsxfun(@plus, A, delta(:,t-1));
			path_transition_probs = exp(bsxfun(@minus, ii, max(ii,[],1))); clear ii % subtract maximum to avoid overflow
			path_transition_probs = bsxfun(@rdivide, path_transition_probs, sum(path_transition_probs,1));

			dAt=bsxfun(@times, path_transition_probs, deriv_Clogadd');
			dA = dA + dAt;
			
			% update derivatives
			deriv_Clogadd = sum(dAt,2);	clear dAt
        end

        % update gradients at time 1
        t = 1;
        dA0(Y(t)) = dA0(Y(t)) - 1;
        dF(Y(t),t) = dF(Y(t),t) - 1;
        dA0 = dA0 + deriv_Clogadd;
        dF(:,t) = dF(:,t) + deriv_Clogadd;      clear deriv_Clogadd t
        dA = [dA0 dA];                          clear dA0

		dA = 1/T*dA;
		dF = 1/T*dF;
    end

end

function s = logsumexp(x,dim)
	if nargin < 2
		dim = 1;
	end
	y = max(x,[],dim);	
	x = bsxfun(@minus, x, y);
	s = y + log(sum(exp(x),dim));
	i = find(~isfinite(y));
	if ~isempty(i)
		s(i) = y(i);
	end
end

