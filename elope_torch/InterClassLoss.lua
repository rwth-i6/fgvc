local InterClassLoss, parent = torch.class('nn.InterClassLoss', 'nn.Criterion')


function InterClassLoss:__init(opt,centers) 
    parent.__init(self)
    self.L_2 = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.alpha = opt.alpha
    self.ind = torch.Tensor()
    self.num_samples = math.min(opt.num_samples,opt.batch_size*(opt.batch_size-1))
    self.margin = opt.margin
    self.mode = 2 --opt.inter_mode --mode 1 = basic, mode 2 = squared
    self.sel_mode = 2 --opt.inter_sel_mode --selection mode 1 = random, 2 = random from pairs with dist < margin
    self.sel_pairs = {}
    self.alpha_vector = torch.Tensor()
    self.centers = centers
end


function InterClassLoss:updateOutput(input,  labels)
    self.sel_pairs = {}
    self.output = 0
    local all_pairs = {}
    local B = input:size(1)
    local diff = torch.Tensor(1):fill(0)
    local total_cdiff = 0

    local ccounter = 0
    for i=1, B-1 do
        for j = i+1, B do
            if (labels[i] ~= labels[j]) then
                cdiff = self.centers[labels[i]] - self.centers[labels[j]]
                cdiff = cdiff:norm(2,1):pow(2)
                total_cdiff = total_cdiff + cdiff
                ccounter = ccounter + 1
                if (cdiff:sum() < self.margin or self.sel_mode == 1) then
                    table.insert(all_pairs, {labels[i],labels[j]})
                end
	    end
        end
    end

    total_cdiff = total_cdiff / ccounter

    if (#all_pairs > 0) then

        local randperm = torch.randperm(#all_pairs):long()
        self.ind = randperm:narrow(1,1,math.min(#all_pairs,self.num_samples))
        for i=1, self.ind:size(1) do
	    table.insert(self.sel_pairs, all_pairs[self.ind[i]])
        end

	
        diff = torch.Tensor(#self.sel_pairs, self.centers:size(2))
        for i=1, #self.sel_pairs do
            diff[i]:copy(self.centers[self.sel_pairs[i][1]] - self.centers[self.sel_pairs[i][2]])
	end
	diff = diff:norm(2,2):pow(2)
	diff = diff:resize(diff:size(1))

        local m = torch.Tensor(#self.sel_pairs):fill(self.margin):typeAs(diff)
        local zero = torch.Tensor(#self.sel_pairs):zero():typeAs(diff)
        local tmp = torch.cat(zero, m - diff, 2)

        if (self.mode == 1) then
            self.L_2 = torch.max(tmp, 2)
        elseif (self.mode == 2) then
            self.L_2 = (torch.max(tmp, 2)):pow(2)
        end

	self.output = (self.L_2:sum() / self.num_samples) / (self.mode * 2)
    end

    if type(total_cdiff) ~= 'number' then
        total_cdiff = total_cdiff:sum() --turn into a number
    end

    return self.output,{total_cdiff,diff:mean(),diff:min(),#all_pairs,#self.sel_pairs}
end


function InterClassLoss:updateGradInput(input,  labels)
    local B = input:size(1)

    self.gradInput = self.gradInput:resize(input:size()):typeAs(input):fill(0)

    if (self.alpha > 0) then
        if (#self.sel_pairs > 0) then
            for i=1, B do
		local find_y = torch.find(labels, labels[i])
		for j=1, #self.sel_pairs do
		    local dist = (self.centers[self.sel_pairs[j][1]]-self.centers[self.sel_pairs[j][2]]):norm(2)
                    if (self.mode == 2) then
                        dist = dist * dist
                    end
		    if (dist < self.margin) then
		   	if (labels[i] == self.sel_pairs[j][1] or labels[i] == self.sel_pairs[j][2]) then
		    	    local tmp_grad =  self.alpha * (self.centers[self.sel_pairs[j][1]]-self.centers[self.sel_pairs[j][2]]) / (1 + #find_y)
                            if (self.mode == 2) then
                                tmp_grad = tmp_grad * (self.margin - dist)
                            end
                            tmp_grad = tmp_grad:typeAs(self.gradInput)
			    if (self.sel_pairs[j][1] == labels[i]) then
			        self.gradInput[i] = self.gradInput[i]:add(-tmp_grad)
			    else		
				self.gradInput[i] = self.gradInput[i]:add(tmp_grad)
			    end
			end
		    end	
		end
	    end
	
            self.gradInput = self.gradInput / self.num_samples 
        end
    end

    return self.gradInput
end

