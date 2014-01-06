function params = embCell2params(embCell,embSizes)
    params = [];    
    for e=1:numel(embCell)
        params = [params; embCell{e}(:)];
    end
end
