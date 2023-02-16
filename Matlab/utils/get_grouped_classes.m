function groupedClasses = get_grouped_classes(negGroup, posGroup, params)

    if nargin < 3
        negClassValue = -1;
        posClassValue = 1;
    else
        negClassValue = params.negClassValue;
        posClassValue = params.posClassValue;
    end
    
    groupedClasses = nan(1,size(negGroup,2));

    if size(negGroup,1) > 1
        negIdx = any(negGroup == 1);
    else
        negIdx = negGroup == 1;
    end

    if size(posGroup,1) > 1
        posIdx = any(posGroup == 1);
    else
        posIdx = posGroup == 1;
    end

    groupedClasses(negIdx) = negClassValue;
    groupedClasses(posIdx) = posClassValue;

    

end