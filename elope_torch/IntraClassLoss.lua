local IntraClassLoss, parent = torch.class('nn.IntraClassLoss', 'nn.Criterion')

function IntraClassLoss:__init(opt,centers) 
    parent.__init(self)
    self.dists = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.alpha = opt.alpha
    self.mode = 2 --opt.intra_mode --mode 1 = center loss, mode 2 = modified center loss
    self.centers = centers
end

function IntraClassLoss:updateOutput(input, labels)	
    local B = input:size(1)
    local diff = torch.Tensor(B, input:size(2))
    self.centers = self.centers:float() --TODO
    for i=1, B do
        diff[i]:copy(input[i] - self.centers[labels[i]])
    end	
    self.dists = diff:norm(2,2):pow(2)
    self.output = (self.dists:sum()  / B ) / 2
    return self.output,{self.dists:mean(),self.dists:min(),self.dists:max()}
end


function IntraClassLoss:updateGradInput(input,  labels)
    self.gradInput = self.gradInput:resize(input:size()):typeAs(input):fill(0)
    local B = input:size(1)

    if (self.mode == 1) then
        for i=1, B do
            self.gradInput[i]:copy((input[i] - self.centers[labels[i]]))  
        end
    elseif(self.mode == 2) then
        for i=1, B do
            local indices_ti = torch.find(labels, labels[i])
            local count = 1 + #indices_ti
            self.gradInput[i]:copy((input[i] - self.centers[labels[i]]) * (1 - self.alpha / count))
        end
    end
   

    self.gradInput = self.gradInput / B
	
    return self.gradInput
end












