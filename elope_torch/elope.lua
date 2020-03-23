require 'nn'
require 'cudnn'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntraClassLoss'
require 'InterClassLoss'
require 'KMaxPooling'
require 'training'
local lapp = require 'pl.lapp'
local utils = require 'utils'
local checkpoint = require 'external/checkpoints'
local modelInit = require 'external/init'
local opt = lapp [[
--save_path             (default '') 
--epochs                (default 90)
--save_epoch            (default 30)
--num_classes           (default 200)
--lambda                (default 2.0)
--num_samples           (default 256)
--margin                (default 0.75)
--alpha                 (default 0.5)
--gamma                 (default 16.0)
--seed                  (default 1)
--batch_size            (default 14)
--epoch_step            (default 30)
--learning_rate_decay   (default 0.1)
--learning_rate         (default 0.003)
--weight_decay          (default 0.001)
--momentum              (default 0.9)
--name                  (default '')
--backbone              (default '')
--centers               (default '')
--finetune              (default '')
--softmax               (default 1)
--intra                 (default 1)
--inter                 (default 1)
--tau                   (default 0.3)
--scale_size            (default 448)
--crop_size             (default 448)
--lastLayerSize         (default 512)
--F                     (default 512)
--K                     (default 4)
--image_path            (default '')
--dataset_path          (default '')
--dataset_name          (default '')
--num_threads           (default 4)
--mode                  (default 'train')
--hflip                 (default true)
--rotation_aug_p        (default 1)
--rotation_aug_deg      (default 20)
--jitter_b              (default 0.4)
--jitter_c              (default 0.4)
--jitter_s              (default 0.4)
--lighting              (default 0.1)
--localization_module   (default '')
]]

print('Options: ', opt)
if opt.seed > 0 then
    print('Using random seed ' .. opt.seed)
    torch.manualSeed(opt.seed)
    cutorch.manualSeed(opt.seed)
    math.randomseed(opt.seed)
end
torch.setnumthreads(opt.num_threads)

local train_dataset, test_dataset, val_dataset
train_dataset = torch.load(path.join(opt.dataset_path, opt.dataset_name .. '_train.t7'))
train_dataset.image_path = opt.image_path
train_dataset.name = 'train'
test_dataset = torch.load(path.join(opt.dataset_path, opt.dataset_name .. '_test.t7'))
test_dataset.image_path = opt.image_path
test_dataset.name = 'test'
if utils.is_file(path.join(opt.dataset_path, opt.dataset_name .. '_val.t7')) then
    val_dataset = torch.load(path.join(opt.dataset_path, opt.dataset_name .. '_val.t7'))
    val_dataset.image_path = opt.image_path
    val_dataset.name = 'val'
end

local model = {}
local criterion = {}
local optim_state = nil


local centers = torch.randn(opt.num_classes, opt.F) 
if not(opt.centers == '') then
    centers = torch.load(opt.centers)
end
model.centers = centers

if not(opt.finetune == '') then
    print('Finetuning from: ' .. opt.finetune)
    model.network = torch.load(opt.finetune)
    local finetune_fc = string.gsub(opt.finetune, 'network', 'fc')
    model.fc = torch.load(finetune_fc)
else
    print('Building new model from backbone')
    --backbone, e.g. resnet
    local backbone  = torch.load(opt.backbone)
    --feature dimension of last conv layer
    local nFeatures = 2048
    --remove classification layer and global average pooling
    backbone:remove(#backbone)
    backbone:remove(#backbone)
    backbone:remove(#backbone)
    --add global k-max pooling
    backbone:add(nn.KMaxPooling(opt.K))
    backbone:add(nn.View(nFeatures):setNumInputDims(3))
    --add embedding layer if defined
    local lastLayerSize = opt.lastLayerSize
    if lastLayerSize > -1 then
       backbone:add(nn.Linear(nFeatures, lastLayerSize))
       backbone:add(nn.BatchNormalization(lastLayerSize))
       backbone:add(nn.PReLU())
    else
       lastLayerSize = nFeatures
    end

    --create new fully connected classification layer
    backbone:add(nn.Linear(lastLayerSize, opt.num_classes))
    backbone = backbone:cuda()

    --optimize memory
    opt.tensorType = 'torch.CudaTensor'
    modelInit.shareGradInput(backbone,opt)
    backbone.gradInput = nil

    model.fc = backbone:get(#backbone)
    backbone:remove(#backbone)
    model.network = backbone
end

model.normnn = nn.Sequential()
model.normnn:add(nn.Normalize(2))
model.centers = model.centers:cdiv(model.centers:norm(2,2):expandAs(model.centers))

if opt.localization_module ~= '' then
    model.localization_module = torch.load(opt.localization_module)
    model.localization_module:cuda()
    model.localization_module:evaluate()
end

criterion.softmax = nil
if (opt.softmax == 1) then
    criterion.softmax = nn.CrossEntropyCriterion()
    criterion.softmax = criterion.softmax:cuda()
end

criterion.intra_loss = nil
if (opt.intra == 1) then
    criterion.intra_loss = nn.IntraClassLoss(opt, model.centers)
    criterion.intra_loss = criterion.intra_loss:cuda()
end

criterion.inter_loss = nil
if (opt.inter == 1) then
    criterion.inter_loss = nn.InterClassLoss(opt, model.centers)
    criterion.inter_loss = criterion.inter_loss:cuda()
end

model.network = model.network:cuda()
model.fc = model.fc:cuda()
model.normnn:cuda()
model.network:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
model.full_network = nn.Sequential()
model.full_network:add(model.network)
model.full_network:add(model.fc)

-- initiaization
optim_state = {
   learningRate = opt.learning_rate,
   momentum = opt.momentum,
   learningRateDecay = 0.0,
   weightDecay = opt.weight_decay,
   dampening = 0.0,
   nesterov = false,
}

model.params, model.gradParams = model.full_network:getParameters()
print('Network has', model.params:numel(), 'parameters')
print('Network has', #model.network:findModules'cudnn.SpatialConvolution', 'convolutions')

modelFilePrefix = opt.name  
modelFilePrefix = modelFilePrefix .. '_s'..  opt.seed
modelFilePrefix = modelFilePrefix .. '_is' .. opt.scale_size
modelFilePrefix = modelFilePrefix .. '_lls' .. opt.lastLayerSize
modelFilePrefix = modelFilePrefix .. '_lr' .. opt.learning_rate
modelFilePrefix = modelFilePrefix .. '_bs' .. opt.batch_size
modelFilePrefix = modelFilePrefix .. '_es' .. opt.epoch_step
modelFilePrefix = modelFilePrefix .. '_aug' .. opt.rotation_aug_p .. '-' .. opt.rotation_aug_deg .. '-' .. opt.jitter_b .. '-' .. opt.jitter_c .. '-' .. opt.jitter_s .. '-' .. opt.lighting
modelFilePrefix = modelFilePrefix .. '_g' .. opt.gamma
modelFilePrefix = modelFilePrefix .. '_l' .. opt.lambda
modelFilePrefix = modelFilePrefix .. '_a' .. opt.alpha
modelFilePrefix = modelFilePrefix .. '_m' .. opt.margin


if opt.mode == 'test' then
    if (val_dataset ~= nil) then
        local val_mean_loss, val_accuracy = testing(val_dataset, model, criterion, opt)
        print('VAL:   Accuracy epoch = ' .. val_accuracy .. '%')
        print('VAL:  Mean loss epoch = ' .. val_mean_loss .. '%')
    end

    if (test_dataset ~= nil) then
        local test_mean_loss, test_accuracy = testing(test_dataset, model, criterion, opt)
        print('TEST:  Accuracy epoch = ' .. test_accuracy .. '%')
        print('TEST: Mean loss epoch = ' .. test_mean_loss .. '%')
    end
end


local epoch = 1
local best_val_accuracy = 0
while (opt.epochs == -1 or opt.epochs >= epoch) and (opt.mode == 'train') do
    print('TRAIN: Epoch ' .. epoch)
    mean_loss, total_time, model, optim_state = training(train_dataset, opt, model, criterion, optim_state, epoch)
    print('TRAIN: Total time = ' .. total_time .. ' hours')
    print('TRAIN:  Mean loss = ' .. mean_loss)
   
    local val_mean_loss = 0
    local val_accuracy = 0
    local test_mean_loss = 0 
    local test_accuracy = 0

    if (val_dataset ~= nil) then
        val_mean_loss, val_accuracy = testing(val_dataset, model, criterion, opt)
        print('VAL:   Accuracy epoch ' .. epoch .. ' = ' .. val_accuracy .. '%')
        print('VAL:  Mean loss epoch ' .. epoch .. ' = ' .. val_mean_loss .. '%')
    end

    if (test_dataset ~= nil) then
        test_mean_loss, test_accuracy = testing(test_dataset, model, criterion, opt)
        print('TEST:  Accuracy epoch ' .. epoch .. ' = ' .. test_accuracy .. '%')
        print('TEST: Mean loss epoch ' .. epoch .. ' = ' .. test_mean_loss .. '%')
    end

    if (epoch % opt.save_epoch == 0) then
        checkpoint.save(epoch, model.network, model.fc, model.optim_state, torch.getRNGState(), cutorch.getRNGState(), model.centers, modelFilePrefix, opt)
    end 

    if (val_accuracy >= best_val_accuracy) then
        checkpoint.save('best', model.network, model.fc, optim_state, torch.getRNGState(), cutorch.getRNGState(), model.centers, modelFilePrefix, opt)
        best_val_accuracy = val_accuracy
    end


    --update the learning rate
    if (epoch % opt.epoch_step == 0) then
        optim_state.learningRate = optim_state.learningRate * opt.learning_rate_decay
    end 
    epoch = epoch + 1
end

