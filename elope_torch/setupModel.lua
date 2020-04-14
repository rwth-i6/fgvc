local modelInit = require 'external/init'

local setupModel = {}

function setupModel.centers(opt)
    -- initialize centers either random or from file
    local centers = torch.randn(opt.num_classes, opt.F)
    if not(opt.centers == '') then
        centers = torch.load(opt.centers)
    end
    centers = centers:cdiv(centers:norm(2,2):expandAs(centers))

    return centers
end

function setupModel.finetune(opt)
    local network = torch.load(opt.finetune)
    local finetune_fc = string.gsub(opt.finetune, 'network', 'fc')
    local fc = torch.load(finetune_fc)

    network = network:cuda()
    fc = fc:cuda()

    -- optimize memory
    opt.tensorType = 'torch.CudaTensor'
    modelInit.shareGradInput(network,opt)
    network.gradInput = nil

    return network, fc
end

function setupModel.resnetFb(opt)
    -- backbone, e.g. resnet
    local backbone  = torch.load(opt.backbone)
    -- feature dimension of last conv layer
    local backboneF = opt.backboneF
    -- remove classification layer and global average pooling
    backbone:remove(#backbone)
    backbone:remove(#backbone)
    backbone:remove(#backbone)
    backbone:add(nn.KMaxPooling(opt.K))
    backbone:add(nn.View(backboneF):setNumInputDims(3))
    -- add embedding layer if defined
    local lastLayerSize = opt.lastLayerSize
    if lastLayerSize > -1 then
       backbone:add(nn.Linear(backboneF, lastLayerSize))
       backbone:add(nn.BatchNormalization(lastLayerSize))
       backbone:add(nn.PReLU())
    else
       lastLayerSize = backboneF
    end

    -- create new fully connected classification layer
    backbone:add(nn.Linear(lastLayerSize, opt.num_classes))
    backbone = backbone:cuda()

    -- optimize memory
    opt.tensorType = 'torch.CudaTensor'
    modelInit.shareGradInput(backbone,opt)
    backbone.gradInput = nil

    local fc = backbone:get(#backbone)
    backbone:remove(#backbone)

    return backbone, fc
end

function setupModel.normnn()
    local normnn = nn.Sequential()
    normnn:add(nn.Normalize(2))
    normnn = normnn:cuda()

    return normnn
end

function setupModel.loc(opt)
    --define localization modile, e.g. from ResNet-50:
    opt.finetune = opt.backbone
    local network, fc = setupModel.finetune(opt)

    local loc = nn.Sequential()
    loc:add(nn.SpatialUpSamplingBilinear({oheight=64, owidth=64}))
    loc:add(network.modules[1]:clone())
    loc:add(network.modules[2]:clone())
    loc:add(network.modules[3]:clone())
    loc:add(network.modules[4]:clone())
    loc:add(network.modules[5]:clone())
    loc:add(nn.SpatialConvolution(256, 1, 3, 3, 1, 1, 0, 0))
    loc:add(nn.View(-1):setNumInputDims(3))
    loc = loc:cuda()
    loc:get(2).gradInput = torch.CudaTensor(4,3,64,64)

    return network, loc
end

function setupModel.modelFilePrefix(opt)
    local modelFilePrefix = opt.name
    modelFilePrefix = modelFilePrefix .. '_s'..  opt.seed
    modelFilePrefix = modelFilePrefix .. '_is' .. opt.scale_size
    modelFilePrefix = modelFilePrefix .. '_lls' .. opt.lastLayerSize
    modelFilePrefix = modelFilePrefix .. '_lr' .. opt.learning_rate
    modelFilePrefix = modelFilePrefix .. '_bs' .. opt.batch_size
    modelFilePrefix = modelFilePrefix .. '_lrds' .. opt.learning_rate_decay_step
    modelFilePrefix = modelFilePrefix .. '_aug' .. opt.rotation_aug_p 
                      .. '-' .. opt.rotation_aug_deg .. '-' 
                      .. opt.jitter_b .. '-' .. opt.jitter_c 
                      .. '-' .. opt.jitter_s .. '-' .. opt.lighting
    modelFilePrefix = modelFilePrefix .. '_g' .. opt.gamma
    modelFilePrefix = modelFilePrefix .. '_l' .. opt.lambda
    modelFilePrefix = modelFilePrefix .. '_a' .. opt.alpha
    modelFilePrefix = modelFilePrefix .. '_m' .. opt.margin

    return modelFilePrefix
end

return setupModel
