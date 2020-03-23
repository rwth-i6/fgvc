local utils = {}

function utils.get_bb(out,sizes,scale_size,threshold)
    local heat_map = utils.minmaxnorm(out):ge(threshold)
    local heat_map_hull = utils.convex_hull(heat_map)
    local f_x = (scale_size / heat_map:size(2) ) * (sizes[3] / scale_size)
    local f_y = (scale_size / heat_map:size(1) ) * (sizes[2] / scale_size)
    local x1 = (heat_map_hull[3] - 1) * f_x
    local y1 = (heat_map_hull[1] - 1) * f_y
    local x2 = (heat_map_hull[4]) * f_x
    local y2 = (heat_map_hull[2]) * f_y
    if x2-x1 < 10 or y2-y1 < 10 then
          x1 = 1
          y1 = 1
          x2 = sizes[3]
          y2 = sizes[2]
          print('ERROR, new: ',x1,y1,x2,y2)
    end
    return {x1,y1,x2-x1,y2-y1}
end

function utils.convex_hull(matrix)
    local min_j = 1
    local min_i = 1
    local max_j = matrix:size(1)
    local max_i = matrix:size(2)

    local matrix_max1,_ = matrix:max(1)
    local matrix_max2,_ = matrix:max(2)

    for j=1,matrix:size(1) do
        if (matrix_max2[j][1] > 0) then
            min_j = j
            break
        end
    end

    for j=matrix:size(1),1,-1 do
        if (matrix_max2[j][1] > 0) then
            max_j = j
            break
        end
    end

    for i=1,matrix:size(2) do
        if (matrix_max1[1][i] > 0) then
            min_i = i
            break
        end
    end

    for i=matrix:size(2),1,-1 do
        if (matrix_max1[1][i] > 0) then
            max_i = i
            break
        end
    end

    return {min_j,max_j,min_i,max_i}
end

function utils.minmaxnorm(input)
    input = input - input:min()
    input = input / input:max()
    return input
end

function utils.read_lines(file_name,flags)
    local lines = {}
    for line in io.lines(file_name) do
        lines[#lines + 1] = line
    end

    return lines
end

function utils.is_file(file_name)
    local file = io.open(file_name, "r")
    if file ~= nil then
        io.close(file)
        return true
    else
        return false
    end
end

function utils.string_split(string,words)
    local words = {}
    for word in string.gmatch(string,"%S+") do
        words[#words + 1] = word
    end
    return words
end

return utils

