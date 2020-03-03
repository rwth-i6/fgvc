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



function center_update(centers, alpha, data, labels)
    -- list indices within batch by class 
    local indices_by_class = {}
    for n=1,labels:size(1) do
        local class_n = labels[n]
        if indices_by_class[class_n] == nil then
            indices_by_class[class_n] = {n}
        else
            table.insert(indices_by_class[class_n], n)
        end
    end

    -- compute delta for each class in batch    
    local delta_c = torch.zeros(centers:size())
    for k, indices_k in pairs(indices_by_class) do
        local data_c = data:index(1, torch.LongTensor(indices_k))
        local N = data_c:size(1)
        delta_c[k]:copy(centers[k]:repeatTensor(N, 1):add(-data_c):sum(1))
    end
    centers:add(-alpha * delta_c)

    --renormalize centers
    centers = centers:cdiv(centers:norm(2,2):expandAs(centers))

    return centers
end


function training(dataset, opt, model, criterion, optim_state, epoch)
    model.network:training()
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
    local labels = torch.zeros(B)
    local prepT = preprocess.getPreprocessTransformer(scale_size,crop_size,opt.jitter_b,opt.jitter_c,opt.jitter_s,opt.lighting)    
    local prepTLoc = preprocess.getPreprocessTransformerLoc(scale_size,opt.jitter_b,opt.jitter_c,opt.jitter_s,opt.lighting)

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
            local img_path = ffi.string(inputs_char[i]:data())
            local img = image.load(path.join(dataset.image_path,img_path), 3)
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
        labels:copy(dataset.label:index(1,batch))
        --forward the input
        local network_output = model.network:forward(inputs)
        local normnn_output = model.normnn:forward(network_output)
        labels = labels:float()
        network_output = network_output:float()
        normnn_output = normnn_output:float()

        model.full_network:zeroGradParameters()
        

        --evaluate and update softmax loss
        local fc_output = model.fc:forward(network_output:cuda())
        local softmax_loss = 0.0
        local dl_do_softmax = 0
        if criterion.softmax then
            softmax_loss = criterion.softmax:forward(fc_output:cuda(), labels:cuda())
            local dl_fcOutput = criterion.softmax:backward(fc_output, labels:cuda())
            dl_do_softmax = model.fc:backward(network_output:cuda(), dl_fcOutput)
        end
  
        --center update
        model.centers = center_update(model.centers, opt.alpha, normnn_output, labels)
        criterion.intra_loss.centers:copy(model.centers)
        criterion.inter_loss.centers:copy(model.centers)  

        --evaluate and update intra-class loss 
        local intra_loss = 0
        local intra_log = {0,0,0,0}
        local dl_do_intra = torch.zeros(network_output:size())
        if criterion.intra_loss then
	    intra_loss,intra_log = criterion.intra_loss:forward(normnn_output:float(), labels)
            local dl_normnnOutput = criterion.intra_loss:backward(normnn_output:float(), labels)
            dl_do_intra = model.normnn:backward(network_output:cuda(),  dl_normnnOutput:cuda()):clone()
        end

        --evaluate and update inter-class loss
        local inter_loss = 0
        local inter_log = {0,0,0,0,0}
        local dl_do_inter = torch.zeros(network_output:size())
        if criterion.inter_loss then
            inter_loss,inter_log = criterion.inter_loss:forward(normnn_output:float(), labels)
            local dl_normnnOutput = criterion.inter_loss:backward(normnn_output:float(), labels)
            dl_do_inter = model.normnn:backward(network_output:cuda(),  dl_normnnOutput:cuda())
        end
         

	local loss = softmax_loss + opt.lambda * (intra_loss + opt.gamma * inter_loss)
        local iteration_string = 'Iteration [' .. t .. '/' .. total_iterations .. '] ' .. os.date("%x %X", sys.clock()) .. ': '
        print(iteration_string .. ' Intra log: ' .. intra_log[1] .. ' ' .. intra_log[2] .. ' ' .. intra_log[3])
        print(iteration_string .. ' Inter log: ' .. inter_log[1] .. ' ' .. inter_log[2] .. ' ' .. inter_log[3] .. ' ' .. inter_log[4] .. ' ' .. inter_log[5])
        print(iteration_string .. ' Loss softmax = ' .. softmax_loss)
        print(iteration_string .. ' Loss intra-class = ' .. intra_loss)
        print(iteration_string .. ' Loss inter-class = ' .. inter_loss)
        print(iteration_string .. ' Loss = ' .. loss)
	total_loss = total_loss + loss
 
        local dl_do = dl_do_softmax + opt.lambda * (dl_do_intra:cuda() + opt.gamma * dl_do_inter:cuda()) 
        local dl_do_network = model.network:backward(inputs,dl_do:cuda())

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


function testing(dataset, model, criterion, opt)
    model.network:evaluate()
    local N = dataset.data:size(1)
    local B = opt.batch_size
    local scale_size = opt.scale_size
    local crop_size = opt.crop_size
     
    -- generate batches
    local batches = torch.LongTensor(N)
    for i=1,N do
        batches[i] = i
    end
    batches = batches:split(B)

    local inputs_char = torch.CharTensor(B,128) --TODO
    local inputs = torch.zeros(B, 1, crop_size, crop_size)
    local labels = torch.zeros(B)
    local prepT = preprocess.getPreprocessTransformerTest(scale_size,crop_size)
    local prepTLoc = preprocess.getPreprocessTransformerTestLoc(scale_size)

    local total_loss = 0
    local total_correct = 0
    local total_iterations = #batches
    for t,batch in ipairs(batches) do
        B = batch:size(1)
        labels:resize(B)
        labels:copy(dataset.label:index(1,batch))
        inputs:resize(B, 3, crop_size, crop_size)
        inputs_char:resize(B,128) --TODO
        inputs_char:copy(dataset.data:index(1,batch))
        for i=1, inputs_char:size(1) do
            local img_path = ffi.string(inputs_char[i]:data())
            local img = image.load(path.join(dataset.image_path,img_path),3)
            local img_size = img:size()

            img = prepT(img)

            inputs[i]:copy(img)
        end

        local network_output = model.network:forward(inputs:cuda())
        local output = model.fc(network_output)

        labels = labels:cuda()
        total_loss = total_loss + criterion.softmax:forward(output, labels)
        local _, preds = torch.sort(output, output:size():size(), true)
        local predsk = preds:narrow(2,1,1):float()
        local predskmax,_ = predsk:eq(labels:float():expandAs(predsk)):max(2)
        local local_correct = predskmax:sum()
        total_correct = total_correct + local_correct

        print('Iteration [' .. t .. '/' .. total_iterations .. '] ' .. os.date("%x %X", sys.clock()) .. ': ' .. local_correct / B * 100)

        collectgarbage()
    end

    local accuracy = total_correct / N * 100
    local mean_loss = total_loss / #batches

    return mean_loss, accuracy 
end

