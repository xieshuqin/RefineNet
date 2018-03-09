# Detectron Code Struct
    1. RPN & FPN generates a set of proposals. 

        rpn_rois_fpn

    2. CollectAndDistribute Op

        rois_fpn2, rois_fpn3 ...
        rois_idx_restore

    3. Fast/Mask R-CNN RoIalign

        for lvl in range(k_min, k_max):
            rois = 'rois_fpn'.format(lvl)
            fpn_feat = blob_in[lvl]
            roi_feat = RoIAlign([fpn_feat, rois])
            blob_out_list = blob_out_list.append(roi_feat)

        blob_out = Permute(Concat(blob_out_list))
        **'_[mask]_roi_feat' = blob_out**

    4. Mask Head
        'mask_fcn_logits' = add_mask_head('_[mask]_roi_feat')


# Refine Mask Net Code Struct
###  Prepare inputs 

The struct is similar to Mask R-CNN, defined after adding the mask-rcnn head. 
A function packages all the ops. Named as `add_refined_mask_input()`

`add_refined_mask_input(model, blobs_in, spatial_scales)`:

    Do the Rescale and Dumplicate operator, generate global indicator, concatenate to be the `refine_net_input`. 

    blobs_in: FPN or Resnet feature
    blob_out: `refine_net_input`

    Pseudocode: 
        features = RescaleAndDumplicateFeatures()
        indicators = GenerateMaskIndicators()
        concat = Concat([features, indicators], axis=1)
        refined_mask_net_input = concat


### Refine Mask Net definition:
Implemented in `refine_net_head.py`. The net is defined after adding mask heads as well as the input op for refine net. We use hourglass as the fcn. 

The `_add_refine_mask_head` are called after `_add_mask_head` on the `model_builder.py`. The code for training is good, but it's wrong at the inference stage. The way to extract the `model.refine_mask_net` is **wrong** becuase the `model.net.Proto()` only contains the `bbox_net` but not the `mask_net`, so brute-force delete the `bbox_net` and `mask_net` from the new net will be wrong. 

### Refine Mask Net label:

Implemented with an `add_refine_mask_blobs()` function which is called after the `add_mask_rcnn_blobs()` with a trigger `cfg.MODEL.REFINE_MASK_ON`. 

The implement of `add_refine_mask_blobs()`. 


### Refine Net Heads
`refine_net_heads.py`

Makes it general for Mask/Keypoint indicators and Mask/Keypoint refined output. Abstract the indicator type out from the `add_refine_net_inputs()` func and the refined-output type from the `add_refine_net_outputs()` as well as the loss functions. 

Currently, the **mask** part has finished. But the **keypoint** part is **empty**

##### IMPORTANT!!!
To make the code compatable with test time code, we have to construct a 
`refine_mask_net` and a `refine_keypoints_net`. Therefore, the model definition must avoid the same name for different nets. The `refine_head` and 
`refine_output` part is done by adding a 'prefix'. but the `refine_input` 
**HAVEN'T** yet. Must correct this issue.

### Inference Phase
test.py

Only implement the `mask` related test function. The `keypoint` function haven't been written yet. 

