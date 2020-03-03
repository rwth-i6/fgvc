local t = require 'external/transforms'

local M = {}

M.meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
M.pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}


function M.getPreprocessTransformerTest(scale_size, crop_size)
   return t.Compose{
         t.ScaleSquare(scale_size),
         t.ColorNormalize(M.meanstd)
   }
end

function M.getPreprocessTransformerTestLoc(scale_size)
   return t.Compose{
         t.ColorNormalize(M.meanstd)
   }
end

function M.getPreprocessTransformer(scale_size,crop_size,jitter_b,jitter_c,jitter_s,lighting,gray)
   return t.Compose{
         t.ScaleSquare(scale_size),
         t.ColorJitter({
            brightness = jitter_b,
            contrast = jitter_c,
            saturation = jitter_s,
         }),
         t.Lighting(lighting, M.pca.eigval, M.pca.eigvec),
         t.ColorNormalize(M.meanstd)
   }
end

function M.getPreprocessTransformerLoc(scale_size,jitter_b,jitter_c,jitter_s,lighting)
   return t.Compose{
         t.ColorJitter({
            brightness = jitter_b,
            contrast = jitter_c,
            saturation = jitter_s,
         }),
         t.Lighting(lighting, M.pca.eigval, M.pca.eigvec),
         t.ColorNormalize(M.meanstd)
   }
end

return M
