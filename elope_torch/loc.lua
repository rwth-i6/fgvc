require 'torch'
require 'cudnn'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

require 'paths'
require 'KMaxPooling'
local lapp = require 'pl.lapp'
local utils = require 'utils'
local optnet =  require 'optnet'
local checkpoint = require 'external/checkpoints'
local modelInit = require 'external/init'
local opt = lapp [[
--save_path            (default '') 
--backbone             (default '')
--epochs               (default 45)
--save_epoch           (default 15)
--num_classes          (default 200)
--hflip                (default true)
--seed                 (default 1)
--batch_size           (default 16)
--epoch_step           (default 15)
--learning_rate_decay  (default 0.1)
--learning_rate        (default 0.1)
--weight_decay         (default 0.001)
--momentum             (default 0.9)
--name                 (default '')
--rotation_aug_p       (default 1)
--rotation_aug_deg     (default 30)
--scale_size           (default 448)
--crop_size            (default 448)
--loc_w                (default 14)
--loc_h                (default 14)
--image_path           (default '')
--dataset_path         (default '')
--dataset_name         (default '')
--num_threads          (default 4)
--jitter_b             (default 0.4)
--jitter_c             (default 0.4)
--jitter_s             (default 0.4)
--lighting             (default 0.1)
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

require 'trainingLoc'
local model = {}
local criterion = {}
local optim_state = nil

model.network = torch.load(opt.backbone)
model.network = model.network:cuda()
model.network:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
criterion.smoothl1 = nn.SmoothL1Criterion()
criterion.smoothl1 = criterion.smoothl1:cuda()

--define localization modile, e.g. for ResNet-50:
model.loc = nn.Sequential()
model.loc:add(nn.SpatialUpSamplingBilinear({oheight=64, owidth=64}))
model.loc:add(model.network.modules[1]:clone())
model.loc:add(model.network.modules[2]:clone())
model.loc:add(model.network.modules[3]:clone())
model.loc:add(model.network.modules[4]:clone())
model.loc:add(model.network.modules[5]:clone())
model.loc:add(nn.SpatialConvolution(256, 1, 3, 3, 1, 1, 0, 0))
model.loc:add(nn.View(-1):setNumInputDims(3))
model.loc = model.loc:cuda()
model.loc:get(2).gradInput = torch.CudaTensor(4,3,64,64)

-- initiaization
optim_state = {
   learningRate = opt.learning_rate,
   momentum = opt.momentum,
   learningRateDecay = 0.0,
   weightDecay = opt.weight_decay,
   dampening = 0.0,
   nesterov = false,
}

model.params, model.gradParams = model.loc:getParameters()
print('Network has', model.params:numel(), 'parameters')
print('Network has', #model.network:findModules'cudnn.SpatialConvolution', 'convolutions')

modelFilePrefix = opt.name  

local epoch = 1
while (opt.epochs == -1 or opt.epochs >= epoch) do
    print('TRAIN: Epoch ' .. epoch)
    mean_loss, total_time, model, optim_state = training(train_dataset, opt, model, criterion, optim_state, epoch)
    print('TRAIN: Total time = ' .. total_time .. ' hours')
    print('TRAIN: Mean loss = ' .. mean_loss)
    local val_accuracy = 0
    local val_mean_loss =0

    if (epoch % opt.save_epoch == 0) then
        checkpoint.save(epoch, model.loc, nil, optim_state, torch.getRNGState(), cutorch.getRNGState(), nil, modelFilePrefix, opt)
    end

    --update the learning rate
    if (epoch % opt.epoch_step == 0) then
        optim_state.learningRate = optim_state.learningRate * opt.learning_rate_decay
    end 
    epoch = epoch + 1
end

