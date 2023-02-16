function init_project(params)
 
if ~isfolder(params.outFolder)
    mkdir(params.outFolder)
    save(fullfile(params.outFolder,"params.mat"),"params");

else
    inp = input("Warning you maybe overwritting previous analysis!\n" + ...
                "Do you want to continue [y/n]: " ,"s");
    if ~strcmp(inp,"y")
        return
    else
        save(fullfile(params.outFolder,"params.mat"),"params");
    end
end

end