--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local optnet = require 'optnet'

local checkpoint = {}

-- this creates a copy of a network with new modules and the same tensors
local function deepCopy(tbl)
   if type(tbl) == "table" then
      local copy = { }
      for k, v in pairs(tbl) do
         if type(v) == "table" then
            copy[k] = deepCopy(v)
         else
            copy[k] = v
         end
      end
      if torch.typename(tbl) then
         torch.setmetatable(copy, torch.typename(tbl))
      end
      return copy
   else
      return tbl
   end
end

-- this will return a float network leaving the original cuda network untouched
local function floatCopy(model)
   return deepCopy(model):float()
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

function checkpoint.restore(opt)
    modelFilePrefix = opt.resume
    modelFileSuffix = '_epoch_' .. opt.resume_epoch .. '.t7'
    print('Resuming training from: ' .. modelFilePrefix .. '*' .. modelFileSuffix)
    network = torch.load(path.join(opt.save_path, modelFilePrefix .. '_network' .. modelFileSuffix))
    fc = torch.load(path.join(opt.save_path, modelFilePrefix .. '_fc' .. modelFileSuffix))
    centers = torch.load(path.join(opt.save_path, modelFilePrefix .. '_centers' .. modelFileSuffix))
    meta = torch.load(path.join(opt.save_path, modelFilePrefix .. '_meta' .. modelFileSuffix))
    optimState = meta[1]
    torchRNGState = meta[2]
    cutorchRNGState = meta[3]
   
   local sample_input=torch.randn(8,3,224,224):cuda()
   local sample_input_fc=torch.randn(8,512):cuda()
   return network, fc, centers, optimState, torchRNGState, cutorchRNGState
end


function checkpoint.save(epoch, network, fc, optimState, torchRNGState, cutorchRNGState, centers, modelFilePrefix, opt)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(network) == 'nn.DataParallelTable' then
      network = network:get(1)
   end
   -- create a clean copy on the CPU without modifying the original network
   network_f = floatCopy(network) --:clearState()
   optnet.removeOptimization(network_f)
 
   local modelFileSuffix = '_epoch_' .. epoch .. '.t7'
   local networkFile = modelFilePrefix .. '_network' .. modelFileSuffix
   local fcFile = modelFilePrefix .. '_fc' .. modelFileSuffix
   local metaFile = modelFilePrefix .. '_meta' .. modelFileSuffix
   local centerFile = modelFilePrefix .. '_centers' .. modelFileSuffix

   torch.save(paths.concat(opt.save_path, networkFile), network_f:clearState())
   torch.save(paths.concat(opt.save_path, metaFile), {optimState,torchRNGState,cutorchRNGState})
   if centers ~= nil then
       torch.save(paths.concat(opt.save_path, centerFile), centers)
   end

   if fc ~= nil then
       if torch.type(fc) == 'nn.DataParallelTable' then
          fc = fc:get(1)
       end

       fc_f = floatCopy(fc) --:clearState()
       optnet.removeOptimization(fc_f)
       torch.save(paths.concat(opt.save_path, fcFile), fc_f:clearState())
   end

end


return checkpoint
