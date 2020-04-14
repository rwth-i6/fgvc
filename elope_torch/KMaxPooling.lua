local KMaxPooling, parent = torch.class('nn.KMaxPooling', 'nn.Module')

function KMaxPooling:__init(k, global)
    parent.__init(self)
    self.k = k
    self.global = global
    self.indices = torch.Tensor()
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
end

function KMaxPooling:__tostring__()
    local string = torch.type(self) .. ' (K = ' .. self.k .. ', ' 
                   .. 'global = ' .. tostring(self.global) .. ')'
    
    return string
end

function KMaxPooling:updateOutput(input)
    local B = input:size(1)
    local D = input:size(2)
    local H = input:size(3)
    local W = input:size(4)

    self.output:typeAs(input):resize(B, D, 1, 1)
    local x = input:view(B, D, H*W)
    local topKActivations
    topKActivations, self.indices = x:topk(self.k, x:size():size(), true)

    if self.global then
        self.output:typeAs(input):resize(B, D, 1, 1)
        torch.sum(self.output,topKActivations,3)
        self.output:div(self.k)
    else
        self.output:typeAs(input):resize(B, D, self.k)
        self.output:copy(topKActivations)
    end

    return self.output
end

function KMaxPooling:updateGradInput(input, gradOutput)
    local B = input:size(1)
    local D = input:size(2)
    local H = input:size(3)
    local W = input:size(4)

    local gradOutputK
    if self.global then
        gradOutputK = torch.expand(gradOutput:clone():view(B, D, 1), B, D, self.k):div(self.k)
    else
        gradOutputK = gradOutput:clone():view(B, D, self.k)
    end

    self.gradInput = torch.zeros(B,D,H*W):typeAs(input)
    self.gradInput:scatter(3, self.indices, gradOutputK)

    self.gradInput = self.gradInput:view(B,D,H,W)
    return self.gradInput
end

