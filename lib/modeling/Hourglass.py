from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
import modeling.ResNet as ResNet
from caffe2.python import brew
#from utils.c2 import const_fill
#from utils.c2 import gauss_fill
#import modeling.ResNet as ResNet
#import utils.blob as blob_utils

# ---------------------------------------------------------------------------- #
# Hourglass model
# ---------------------------------------------------------------------------- #
# The blobs are terribly named. So is the code struct.
# Re-implement it in the future.

def add_hourglass_head(model, blob_in, blob_out, dim_in, prefix):
    """ add stacked-hourglass head to feature map"""
    """ Only allow 1 stacked hourglass"""
    assert cfg.HG.NUM_STACKS == 1, 'Only 1 stacked hourglass is allowed'

    nStacks = cfg.HG.NUM_STACKS
    dim_feats = cfg.HG.DIM_FEATS

    if dim_in != dim_feats: # conv1 to change the input channel
        blob_in = _add_linear_layer(
            model, blob_in, prefix+'_resort_dim', dim_in, dim_feats
        )

    i = 0
    prefix = prefix + '_stack{}'.format(i)
    hg = _add_hourglass_unit(model, blob_in, prefix, 4)

    # Residual layers at output resolution
    ll = _add_hourglass_residual_block(
        model, hg, prefix+'_hg_conv2', dim_feats, dim_feats
    )
    # linear layer to generate blob_out
    blob_out = _add_linear_layer(
        model, ll, blob_out, dim_feats, dim_feats
    )

    return blob_out, dim_feats


def _add_hourglass_unit(model, blob_in, prefix, n):

    dim_in = cfg.HG.DIM_FEATS
    dim_out = cfg.HG.DIM_FEATS

    # Upper branch
    up1 = _add_hourglass_residual_block(
        model, blob_in, prefix+'_hg{}_up1'.format(n), dim_in, dim_out
    )

    # Lower branch
    low1 = model.MaxPool(
        blob_in, prefix+'_hg{}_p1'.format(n), kernel=2, stride=2
    )
    low1 = _add_hourglass_residual_block(
        model, low1, prefix+'_hg{}_low1'.format(n), dim_in, dim_out
    )

    if n > 1:
        low2 = _add_hourglass_unit(model, low1, prefix, n-1)
    else:
        low2 = _add_hourglass_residual_block(
            model, low1, prefix+'_hg{}_low2'.format(n), dim_in, dim_out
        )

    low3 = _add_hourglass_residual_block(
            model, low2, prefix+'_hg{}_low3'.format(n), dim_in, dim_out
        )
    up2 = model.net.UpsampleNearest(
        low3, prefix+'_hg{}_up2'.format(n), scale=2
    )

    blob_out = model.net.Sum([up1, up2], prefix+'_hg{}_sum'.format(n))

    return blob_out

def _add_linear_layer(model, blob_in, blob_out, dim_in, dim_out):
    blob_conv = model.Conv(
        blob_in, blob_out+'_conv', dim_in, dim_out,
        kernel=1, stride=1, pad=0
    )
    #blob_bn = model.AffineChannel(blob_conv, blob_out+'_bn', inplace=False)
    blob_bn = model.SpatialBN(blob_conv, blob_out+'_bn', dim_out, is_test=False)
    print('blob_bn', blob_bn)
    blob_out = model.Relu(blob_bn, blob_out)
    return blob_out

    #conv_bn = model.ConvAffine(
    #    blob_in, prefix, dim_in, dim_out, kernel=1, stride=1, pad=0,
    #    inplace=True
    #)
    ##blob_out = model.Relu(conv_bn, blob_out)
    #conv_bn = model.Relu(conv_bn, conv_bn)
    #return blob_out

def _add_hourglass_residual_block(model, blob_in, prefix, dim_in, dim_out):
    """ fixed the dim_inner to be dim_out/2 as in implemented in Torch"""
    dim_inner = int(dim_out/2)

    blob_out = ResNet.add_residual_block(
        model,
        prefix,
        blob_in,
        dim_in,
        dim_out,
        dim_inner,
        1,
        inplace_sum=True
    )

    return blob_out



