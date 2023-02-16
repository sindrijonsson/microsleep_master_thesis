function out = parse_json(file)
fid = fopen(file); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
out = jsondecode(str);
end