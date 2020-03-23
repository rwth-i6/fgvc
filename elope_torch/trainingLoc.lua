require 'nn'
require 'optim'
require 'cudnn'
require 'cunn'
require 'torchx'
local ffi = require 'ffi'
local image = require 'image'
local preprocess = require 'external/preprocess'
local utils = require 'utils'


function random_hflip(input, p)
    if torch.uniform() < p then
        input = image.hflip(input)
    end
    return input
end

function rotation_augmentation(input, p, max_angle)
    if torch.uniform() < p then
        local theta = math.rad(torch.random(-max_angle,max_angle))
        input = image.rotate(input, theta, 'bilinear')
    end
    return input
end


function training(dataset, opt, model, criterion, optim_state, epoch)
    model.network:training()
    model.loc:training()
    print('TRAIN: Current optim_state:')
    print(optim_state)
    local N = dataset.data:size(1)
    local B = opt.batch_size
    local scale_size = opt.scale_size
    local crop_size = opt.crop_size
    -- generate random training batches
    local batches = torch.randperm(N):long():split(B)
    if N % B ~= 0 then
        batches[#batches] = nil
    end
   
    local inputs_char = torch.CharTensor(B, 128)
    local inputs = torch.zeros(B, 3, crop_size, crop_size)
    local prepT = preprocess.getPreprocessTransformer(scale_size,crop_size,opt.jitter_b,opt.jitter_c,opt.jitter_s,opt.lighting)    

    local function feval()
        return loss, model.gradParams
    end

    local total_iterations = #batches
    local total_loss = 0
    local total_timer = torch.Timer()
    for t,batch in ipairs(batches) do
        timer = torch.Timer()
        inputs_char:copy(dataset.data:index(1,batch))

        for i=1, inputs_char:size(1) do 
            local img_name = ffi.string(inputs_char[i]:data())
            local img = image.load(path.join(dataset.image_path,img_name), 3)
            img_size = img:size()

            if opt.rotation_aug_p > 0 then
                img = rotation_augmentation(img, opt.rotation_aug_p, opt.rotation_aug_deg)
            end
                
            if opt.hflip then
                img = random_hflip(img,0.5)
            end
 
            img = prepT(img)
            inputs[i]:copy(img)
        end
        inputs = inputs:cuda()

        timePrep = timer:time().real
        --forward the input
        local network_output = model.network:forward(inputs)
        local loc_output = model.loc:forward(inputs)
        --network_output = network_output:float()
        --loc_output = loc_output:float()

        model.loc:zeroGradParameters()

        local loc_gt = model.network.modules[8].output:mean(2):view(B,opt.loc_w*opt.loc_h)
        local loss = criterion.smoothl1:forward(loc_output, loc_gt)
        local dl_do = criterion.smoothl1:backward(loc_output, loc_gt)
        

        local iteration_string = 'Iteration [' .. t .. '/' .. total_iterations .. '] ' .. os.date("%x %X", sys.clock()) .. ': '
        print(iteration_string .. ' Loss = ' .. loss)
	total_loss = total_loss + loss
 
        model.loc:backward(inputs,dl_do:cuda())

        optim.sgd(feval,model.params,optim_state)
        timePrepFwd = timer:time().real
        print(iteration_string .. ' Time prep: ' .. timePrep)
        print(iteration_string .. ' Time total: ' .. timePrepFwd)

        local garbage_before = collectgarbage("count")*1024
        collectgarbage()
        local garbage_after = collectgarbage("count")*1024
        print(iteration_string .. ' Garbage removal: ' .. garbage_before .. ' -> ' .. garbage_after)

    end

    local total_time = total_timer:time().real / 3600
    local mean_loss = total_loss / #batches

    return mean_loss, total_time, model, optim_state
end

