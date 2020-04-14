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

function setupModel.getKMaxWeight(opt)
    local KMaxWeight = nn.Sequential()
    KMaxWeight:add(nn.View(1,opt.backboneF*opt.K,1):setNumInputDims(2))
    local conv = nn.SpatialConvolution(1,1,1,opt.K,1,opt.K,0,0)
    conv.bias = nil
    conv.weight:fill(1/opt.K)
    KMaxWeight:add(conv)
    KMaxWeight:add(nn.View(-1):setNumInputDims(3))

    return KMaxWeight
end


function setupModel.finetune(opt)
    local network = torch.load(opt.finetune)
    local finetune_fc = string.gsub(opt.finetune, 'network', 'fc')
    local fc = torch.load(finetune_fc)

    -- if weights for KMaxPooling are enabled, check if model to 
    -- finetune has such weights, otherwise insert them 
    for n,m in pairs(network.modules) do
        if torch.typename(m) == 'nn.KMaxPooling' then
            -- for compatibility with older implemention of nn.KMaxPooling
            if m.global ==  nil then
                m.global = true
            end
            if opt.KMaxWeight then
                if m.global then
                    network:remove(n)
                    local KMaxWeight = setupModel.getKMaxWeight(opt)
                    network:insert(nn.KMaxPooling(opt.K, false), n)
                    network:insert(KMaxWeight, n+1)
   
                    -- remove old view
                    network:remove(n+2) 
                end
            end
        end
    end

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
    -- add global (weighted) k-max pooling
    if opt.KMaxWeight then
        local KMaxWeight = setupModel.getKMaxWeight(opt)
        backbone:add(nn.KMaxPooling(opt.K, false))
        backbone:add(KMaxWeight)
    else
        backbone:add(nn.KMaxPooling(opt.K, true))
        backbone:add(nn.View(backboneF):setNumInputDims(3))
    end
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
