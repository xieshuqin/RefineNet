from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from caffe2.python import core

import modeling.ResNet as ResNet
#from utils.c2 import const_fill
#from utils.c2 import gauss_fill

# ---------------------------------------------------------------------------- #
# Hourglass model
# ---------------------------------------------------------------------------- #
# The blobs are terribly named. So is the code struct.
# Re-implement it in the future.

def add_hourglass_head(model, blob_in, blob_out, dim_in, prefix, n):
    """ add stacked-hourglass head to feature map"""
    """ Only allow 1 stacked hourglass"""
    assert cfg.HG.NUM_STACKS == 1, 'Only 1 stacked hourglass is allowed'

    nStacks = cfg.HG.NUM_STACKS
    dim_feats = cfg.HG.DIM_FEATS

    if dim_in != dim_feats: # conv1 to change the input channel
        blob_in = add_linear_layer(
            model, blob_in, prefix+'_hg_resort_dim', dim_in, dim_feats
        )

    i = 0
    prefix = prefix + '_stack{}'.format(i)
    hg = add_hourglass_unit(model, blob_in, prefix, n)

    # Residual layers at output resolution
    ll = add_residual_block(
        model, hg, prefix+'_hg_conv2', dim_feats, dim_feats
    )
    # linear layer to generate blob_out
    blob_out = add_linear_layer(
        model, ll, blob_out, dim_feats, dim_feats
    )

    return blob_out, dim_feats


def add_hourglass_unit(model, blob_in, prefix, n):

    dim_in = cfg.HG.DIM_FEATS
    dim_out = cfg.HG.DIM_FEATS

    # Upper branch
    up1 = add_residual_block(
        model, blob_in, prefix+'_hg{}_up1'.format(n), dim_in, dim_out
    )
    # Lower branch
    low1 = model.MaxPool(
        blob_in, prefix+'_hg{}_p1'.format(n), kernel=2, stride=2
    )
    low1 = add_residual_block(
        model, low1, prefix+'_hg{}_low1'.format(n), dim_in, dim_out
    )

    if n > 1:
        low2 = add_hourglass_unit(model, low1, prefix, n-1)
    else:
        low2 = add_residual_block(
            model, low1, prefix+'_hg{}_low2'.format(n), dim_in, dim_out
        )

    low3 = add_residual_block(
            model, low2, prefix+'_hg{}_low3'.format(n), dim_in, dim_out
        )
    up2 = model.net.UpsampleNearest(
        low3, prefix+'_hg{}_up2'.format(n), scale=2
    )

    blob_out = model.net.Sum([up1, up2], prefix+'_hg{}_sum'.format(n))

    return blob_out


def add_residual_block(model, blob_in, prefix, dim_in, dim_out):
    is_test = (not model.train)
    # transform
    tr = add_conv_block(model, blob_in, prefix, dim_in, dim_out, is_test)
    # shortcut
    sc = add_shortcut(model, blob_in, prefix, dim_in, dim_out, is_test)
    # Sum -> Relu
    s = model.net.Sum([tr, sc], prefix+'_sum')

    return model.Relu(s,s)


def add_linear_layer(model, blob_in, blob_out, dim_in, dim_out):

    is_test = (not model.train)
    blob_conv = model.Conv(
        blob_in, blob_out+'_conv', dim_in, dim_out,
        kernel=1, stride=1, pad=0
    )
    # blob_bn = model.SpatialBN(
    #     blob_conv, blob_out+'_bn', dim_out, is_test=is_test
    # )
    # # A little bit surgery to get the running mean and variance
    # # at test time
    # model.params.append(core.ScopedBlobReference(blob_out+'_bn_rm'))
    # model.params.append(core.ScopedBlobReference(blob_out+'_bn_riv'))
    blob_bn = blob_conv

    blob_out = model.Relu(blob_bn, blob_out)

    return blob_out


def add_conv_block(model, blob_in, prefix, dim_in, dim_out, is_test):
    """ fixed the dim_inner to be dim_out/2 as in implemented in Torch"""

    dim_inner = dim_out // 2
    # conv 1x1 -> BN -> Relu
    blob_conv_1 = model.Conv(
        blob_in, prefix+'_branch2a_conv', dim_in, dim_inner,
        kernel=1, stride=1, pad=0
    )
    # blob_bn_1 = model.SpatialBN(
    #     blob_conv_1, prefix+'_branch2a_bn', dim_inner, is_test=is_test
    # )
    # model.params.append(core.ScopedBlobReference(prefix+'_branch2a_bn_rm'))
    # model.params.append(core.ScopedBlobReference(prefix+'_branch2a_bn_riv'))
    blob_bn_1 = blob_conv_1

    blob_relu_1 = model.Relu(blob_bn_1, blob_bn_1)

    # conv 3x3 -> BN -> Relu
    blob_conv_2 = model.Conv(
        blob_relu_1, prefix+'_branch2b_conv', dim_inner, dim_inner,
        kernel=3, stride=1, pad=1
    )
    # blob_bn_2 = model.SpatialBN(
    #     blob_conv_2, prefix+'_branch2b_bn', dim_inner, is_test=is_test
    # )
    # model.params.append(core.ScopedBlobReference(prefix+'_branch2b_bn_rm'))
    # model.params.append(core.ScopedBlobReference(prefix+'_branch2b_bn_riv'))
    blob_bn_2 = blob_conv_2

    blob_relu_2 = model.Relu(blob_bn_2, blob_bn_2)

    # conv 1x1 -> BN
    blob_conv_3 = model.Conv(
        blob_relu_2, prefix+'_branch2c_conv', dim_inner, dim_out,
        kernel=1, stride=1, pad=0
    )
    # blob_bn_3 = model.SpatialBN(
    #     blob_conv_3, prefix+'_branch2c_bn', dim_out, is_test=is_test
    # )
    # model.params.append(core.ScopedBlobReference(prefix+'_branch2c_bn_rm'))
    # model.params.append(core.ScopedBlobReference(prefix+'_branch2c_bn_riv'))
    blob_bn_3 = blob_conv_3

    return blob_bn_3


def add_shortcut(model, blob_in, prefix, dim_in, dim_out, is_test):
    if dim_in == dim_out:
        return blob_in

    blob_conv = model.Conv(
        blob_in, prefix+'_branch1_conv', dim_in, dim_out,
        kernel=1, stride=1, pad=0
    )
    # blob_bn = model.SpatialBN(
    #     blob_conv, prefix+'_branch1_bn', dim_out, is_test=is_test
    # )
    # model.params.append(core.ScopedBlobReference(prefix+'_branch1_bn_rm'))
    # model.params.append(core.ScopedBlobReference(prefix+'_branch1_bn_riv'))
    blob_bn = blob_conv

    return blob_bn




