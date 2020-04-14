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
local setupModel = require 'setupModel'
local opt = lapp [[
--save_path                 (default '')        Path where to save the models
--backbone                  (default '')        t7-file of backbone network
--epochs                    (default 45)        Number of epochs to run training algorithm
--save_epoch                (default 15)        Save model after this many epochs
--hflip                     (default true)      Horizontal flipping of input images with probability 0.
--seed                      (default 1)         Random seed
--batch_size                (default 16)        Batch size for training algorithm
--learning_rate             (default 0.003)     Learning rate for training algorithm
--learning_rate_decay       (default 0.1)       Learning rate decay for training algorithm
--learning_rate_decay_step  (default 0.1)       Reduce learning rate after this many epochs
--weight_decay              (default 0.001)     Weight decay for training algorithm
--momentum                  (default 0.9)       Momentum for training algorithm
--name                      (default '')        Name for the model to train
--rotation_aug_p            (default 1)         Probability of using rotation augmentation
--rotation_aug_deg          (default 30)        Degree of rotation for augmentation
--scale_size                (default 448)       Scale for input images
--crop_size                 (default 448)       Size of image patch to crop from image
--loc_w                     (default 14)        Width of heat map used for localization
--loc_h                     (default 14)        Height of heat map used for localization
--image_path                (default '')        Path to the images
--dataset_path              (default '')        Path where to find the dataset t7-files
--dataset_name              (default '')        Base name of the dataset
--num_threads               (default 4)         Number of threads to use
--jitter_b                  (default 0.4)       Brightness for color jitter augmentation
--jitter_c                  (default 0.4)       Contrast for color jitter augmentation
--jitter_s                  (default 0.4)       Saturation for color jitter augmentation
--lighting                  (default 0.1)       Lighting augmentation
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

model.network, model.loc = setupModel.loc(opt)
model.network:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
criterion.smoothl1 = nn.SmoothL1Criterion()
criterion.smoothl1 = criterion.smoothl1:cuda()

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
    if (epoch % opt.learning_rate_decay_step == 0) then
        optim_state.learningRate = optim_state.learningRate * opt.learning_rate_decay
    end 
    epoch = epoch + 1
end

