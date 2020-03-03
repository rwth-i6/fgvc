local KMaxPooling, parent = torch.class('nn.KMaxPooling', 'nn.Module')

function KMaxPooling:__init(k)
    parent.__init(self)
    self.k = k
    self.indices = torch.Tensor()
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
end

function KMaxPooling:updateOutput(input)
    local B = input:size(1)
    local D = input:size(2)
    local H = input:size(3)
    local W = input:size(4)

    self.output:typeAs(input):resize(B, D, 1, 1)
    local x = input:view(B, D, H*W)
    local scoreSorted
    scoreSorted,self.indices = x:topk(self.k, x:size():size(), true)

    torch.sum(self.output,scoreSorted,3)
    self.output:div(self.k)

    return self.output
end

function KMaxPooling:updateGradInput(input, gradOutput)
    local B = input:size(1)
    local D = input:size(2)
    local H = input:size(3)
    local W = input:size(4)

    local tmp = torch.expand(gradOutput:clone():view(B, D, 1), B, D, self.k)
    self.gradInput = torch.zeros(B,D,H*W):typeAs(input)
    self.gradInput:scatter(3, self.indices, tmp):div(self.k)

    self.gradInput = self.gradInput:view(B,D,H,W)
    return self.gradInput
end

