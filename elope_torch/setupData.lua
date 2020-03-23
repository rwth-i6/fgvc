-- Creates database
--
-- Expects 'train_images.txt' and 'test_images.txt' to be present in opt.data_path
-- Both text files should have two columns, first class number (int) and then a
-- string to the image path (can be relative)
--
-- Output t7 files are written to opt.data_path with prefix specified by opt.name


require 'lfs'
local ffi = require 'ffi'
local lapp = require 'pl.lapp'
local utils = require 'utils'
local opt = lapp [[
--data_path      (default '')
--val_size       (default 0.0)
--val_mode       (default 0)
--num_classes    (default 200)
--name           (default '')
]]


local seed = 1
local max_string_length = 128
torch.manualSeed(seed)
math.randomseed(seed)

function get_lists(image_list_name)
    local image_list_path = path.join(opt.data_path,image_list_name)
    if not utils.is_file(image_list_path) then
        print('File not found: ' .. image_list_path)
        os.exit()
    end

    local image_list_lines = utils.read_lines(image_list_path)
    local total_images = #image_list_lines

    local image_list = torch.CharTensor(total_images,max_string_length):zero()
    local label_list = torch.LongTensor(total_images)

    local images_by_label = {}
    for l=1,opt.num_classes do
        images_by_label[l] = {}
    end

    for i,line in ipairs(image_list_lines) do
        local words = utils.string_split(line)
        local image_name = words[2]
        local image_class = tonumber(words[1])
        ffi.copy(image_list[i]:data(), image_name)
        label_list[i] = image_class

        table.insert(images_by_label[image_class],i)
    end

    return image_list, label_list, images_by_label
end


-- get train lists
local train_image_list, train_label_list, train_images_by_label = get_lists('train_images.txt')

--split up in train/val
if opt.val_size > 0 and opt.val_mode == 1 then
    --split up in train/val
    local N = math.ceil((train_image_list:size(1) / opt.num_classes) * opt.val_size)
    local total_images_val = opt.num_classes*N 
    local total_images_train = train_image_list:size(1) - total_images_val
    local rand_indices = torch.randperm(train_image_list:size(1))

    local val_indices = torch.LongTensor(total_images_val):fill(-1)
    local train_indices = torch.LongTensor(total_images_train):fill(-1)

    local index_val = 1
    local index_train = 1

    for l=1,opt.num_classes do
        local indices_l = torch.Tensor(train_images_by_label[l])
        local rand = torch.randperm(indices_l:size(1))
   
        assert(indices_l:size(1) > N)
 
        for i=1,N do
            val_indices[index_val] = indices_l[rand[i]]
            index_val = index_val + 1
        end
 
        for i=N+1,indices_l:size(1) do
            train_indices[index_train] = indices_l[rand[i]]
            index_train = index_train + 1
        end 

    end

    --create val dataset
    val_dataset = {}
    val_dataset.data = train_image_list:index(1,val_indices)
    val_dataset.label = train_label_list:index(1,val_indices)
    torch.save(path.join(opt.data_path,opt.name .. '_val.t7'),val_dataset)
    print('Val images: ', val_dataset.data:size(1))

    --create train dataset
    train_dataset = {}
    train_dataset.data = train_image_list:index(1,train_indices)
    train_dataset.label = train_label_list:index(1,train_indices)
    torch.save(path.join(opt.data_path,opt.name .. '_train.t7'),train_dataset)
    print('Train images: ', train_dataset.data:size(1))
else
    --create train dataset
    train_dataset = {}
    train_dataset.data = train_image_list
    train_dataset.label = train_label_list
    torch.save(path.join(opt.data_path,opt.name .. '_train.t7'),train_dataset)
    print('Train images: ', train_dataset.data:size(1))
end

if opt.val_mode == 2 then
    --get val lists
    local val_image_list, val_label_list, val_images_by_label = get_lists('val_images.txt')
    --create val dataset
    val_dataset = {}
    val_dataset.data = val_image_list
    val_dataset.label = val_label_list
    torch.save(path.join(opt.data_path,opt.name .. '_val.t7'),val_dataset)
    print('Val images: ', val_dataset.data:size(1))
end

-- get test list
local test_image_list, test_label_list, _ = get_lists('test_images.txt')
test_dataset = {}
test_dataset.data = test_image_list
test_dataset.label = test_label_list
torch.save(path.join(opt.data_path,opt.name .. '_test.t7'),test_dataset)
print('Test images: ', test_dataset.data:size(1))

