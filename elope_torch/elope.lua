require 'nn'
require 'cunn'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntraClassLoss'
require 'InterClassLoss'
require 'KMaxPooling'
require 'training'
local lapp = require 'pl.lapp'
local utils = require 'utils'
local checkpoint = require 'external/checkpoints'
local setupModel = require 'setupModel'
local opt = lapp [[
--save_path                 (default '')        Path where to save the models
--epochs                    (default 90)        Number of epochs to run training algorithm
--save_epoch                (default 30)        Save model after this many epochs (additionally to best validation model)
--num_classes               (default 200)       Number of classes in dataset
--lambda                    (default 2.0)       Weight for intra-class loss
--num_samples               (default 256)       Number of samples used in inter-class loss
--margin                    (default 0.75)      Margin for inter-class loss
--alpha                     (default 0.5)       Learning rate for centers
--gamma                     (default 16.0)      Weight for inter-class loss
--seed                      (default 1)         Random seed
--batch_size                (default 14)        Batch size for training algorithm
--learning_rate             (default 0.003)     Learning rate for training algorithm
--learning_rate_decay       (default 0.1)       Learning rate decay for training algorithm
--learning_rate_decay_step  (default 30)        Reduce learning rate after this many epochs
--weight_decay              (default 0.001)     Weight decay for training algorithm
--momentum                  (default 0.9)       Momentum for training algorithm
--name                      (default '')        Name for the model to train
--backbone                  (default '')        t7-file of backbone network
--backboneF                 (default 2048)      Feature dimension of backbone network before classification layer
--centers                   (default '')        Read in centers
--finetune                  (default '')        Finetune from this model
--localization_module       (default '')        t7-file containing the localiztion module
--softmax                   (default 1)         Indicator to use softmax-loss
--intra                     (default 1)         Indicator to use intra-class loss
--inter                     (default 1)         Indicater to use inter-class loss
--tau                       (default 0.3)       Threshold for localization module
--scale_size                (default 448)       Scale for input images
--crop_size                 (default 448)       Size of image patch to crop from image
--lastLayerSize             (default 512)       Size of the last fc-layer (-1 => no fc-layer is inserted)
--F                         (default 512)       Feature dimension for the embedding
--K                         (default 4)         K in K-max pooling
--image_path                (default '')        Path to the images
--dataset_path              (default '')        Path where to find the dataset t7-files
--dataset_name              (default '')        Base name of the dataset
--num_threads               (default 4)         Number of threads to use
--mode                      (default 'train')   Mode can be either test or train
--rotation_aug_p            (default 1)         Probability of using rotation augmentation
--rotation_aug_deg          (default 20)        Degree of rotation for augmentation
--jitter_b                  (default 0.4)       Brightness for color jitter augmentation
--jitter_c                  (default 0.4)       Contrast for color jitter augmentation
--jitter_s                  (default 0.4)       Saturation for color jitter augmentation
--lighting                  (default 0.1)       Lighting augmentation
--hflip                     (default true)      Horizontal flipping of input images with probability 0.5
]]

print('Options: ', opt)

-- set random seed
if opt.seed > 0 then
    print('Using random seed ' .. opt.seed)
    torch.manualSeed(opt.seed)
    cutorch.manualSeed(opt.seed)
    math.randomseed(opt.seed)
end

torch.setnumthreads(opt.num_threads)

-- load datasets
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

-- setup model and critetion
local model = {}
local criterion = {}
local optim_state = nil

model.centers = setupModel.centers(opt)
model.normnn = setupModel.normnn(opt)

if not(opt.finetune == '') then
    print('Finetuning from: ' .. opt.finetune)
    model.network, model.fc = setupModel.finetune(opt)
else
    print('Building new model from backbone')
    model.network, model.fc = setupModel.resnetFb(opt)
end

-- load localization module if defined
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

model.network:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
model.full_network = nn.Sequential()
model.full_network:add(model.network)
model.full_network:add(model.fc)

-- initiaization of training algorithm
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

local modelFilePrefix = setupModel.modelFilePrefix(opt)

-- test only
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


-- train
local epoch = 1
local best_val_accuracy = 0
while (opt.epochs == -1 or opt.epochs >= epoch) and (opt.mode == 'train') do
    -- train for one epoch
    print('TRAIN: Epoch ' .. epoch)
    mean_loss, total_time, model, optim_state = training(train_dataset, opt, model, criterion, optim_state, epoch)
    print('TRAIN: Total time = ' .. total_time .. ' hours')
    print('TRAIN:  Mean loss = ' .. mean_loss)
   
    -- test
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
    if (epoch % opt.learning_rate_decay_step == 0) then
        optim_state.learningRate = optim_state.learningRate * opt.learning_rate_decay
    end 
    epoch = epoch + 1
end

