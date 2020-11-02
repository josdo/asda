from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from datasets import VOCSegmentation, Cityscapes, Potsdam
from utils import ext_transforms as et
from metrics import StreamSegMetrics

# for model loading
import torch
from torch.utils import data, model_zoo
import copy
import sys
sys.path.append(sys.path[0] + "/..")
from model.deeplab_multi import DeeplabMulti

import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import imgaug as ia
import imgaug.augmenters as iaa

from utils import display

DROPRATE = 0.1
NORM_STYLE = 'bn' # or in
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='../data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['potsdam', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
#     parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
#                         choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
#                                  'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
#                                  'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
#     parser.add_argument("--separable_conv", action='store_true', default=False,
#                         help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--unfreeze_to", type=str, default='decoder',
                        choices=['head', 'aspp_main', 'last_conv_aspp_both', 'all'], 
                        help='unfreeze to which model segment of DeepLabV3+')
    parser.add_argument("--use-se", action='store_true', help="use se block.")
    parser.add_argument("--train_bn", action='store_true', help="train batch normalization.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--droprate", type=float, default=DROPRATE,
                        help="DropRate.")
    parser.add_argument("--norm-style", type=str, default=NORM_STYLE,
                        help="Norm Style in the final classifier.")  
#     num_classes = opts.num_classes,
# #       use_se = use_se, #
# #       train_bn = train_bn, #
#       norm_style = norm_style, # gn
# #       droprate = droprate, # .1
# #       restore_from = restore_from) # default RESTORE_FROM
    
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_percent", type=int, default=0.3)
    parser.add_argument("--no_aug", action='store_true', default=False,
                        help='no data augmentation (default: False)')
    
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--focal_alpha", type=float, default=1,
                        help="alpha value for focal loss (default: 2)")
    parser.add_argument("--focal_gamma", type=float, default=2,
                        help="gamma value for focal loss (default: 1)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    
    # Checkpoint Options
    parser.add_argument("--goal_name", default="no_goal", type=str,
                        help="concise goal of this and related experiments")
    parser.add_argument("--exp_name", default="000", type=str,
                        help="experiment number and 0-3 optional words")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    
    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'potsdam':
        # Augmentation (normalizing happens in the dataset class)
        train_transform = iaa.Sequential([
            # Random crop
            iaa.Crop(percent = (0, opts.crop_percent)),
            # Color jitter
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            # Flips
            iaa.Fliplr(0.5),
            iaa.Rotate([0, 90, 180, 270]),
        ])
        
        val_transform = iaa.Sequential([
            # Flips
            iaa.Fliplr(0.5),
            iaa.Rotate([0, 90, 180, 270]),
        ])
        
        # Turn off augmentation (to reduce regularization for initial training)
        if opts.no_aug:
            train_transform = None
            val_transform = None
        
        # Partitions
        partition = np.load(os.path.join(opts.data_root, 'partition.npy'), allow_pickle=True).item()
        train_IDs = partition['train']
        val_IDs = partition['val']
        
        train_dst = Potsdam(root=opts.data_root,
                               list_IDs=train_IDs, transform=train_transform)
        val_dst = Potsdam(root=opts.data_root,
                             list_IDs=val_IDs, transform=val_transform)
    return train_dst, val_dst
        
def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            _, outputs = model(images) # aux, final classifier outputs
            # upsample to original image size
            outputs = nn.Upsample(size=list(labels.shape)[1:], mode='bilinear', align_corners=True)(outputs)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def restore_model(model, opts, device, verbose=False, from_pretrained=False):
    if opts.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(opts.restore_from)
    elif from_pretrained: # when loading weights from MemReg pre-trained
        saved_state_dict = torch.load(opts.restore_from, map_location=torch.device(device))
    else: # when
        saved_state_dict = torch.load(opts.restore_from, map_location=torch.device(device))["model_state"]
    
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if opts.restore_from[:4] == 'http' :
            if i_parts[1] !='fc' and i_parts[1] !='layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        else:
            if i_parts[0] =='module': # when model parallelized
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
#                 print('%s is loaded from pre-trained weight.\n'%i_parts[1:])
            else:
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
#                 print('%s is loaded from pre-trained weight.\n'%i_parts[0:])
        if verbose:
            print('%s is loaded from pre-trained weight.\n'%i_parts[1:])
    return new_params

def main():
    opts = get_argparser().parse_args()

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True) # num_workers=2
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True) # num_workers=2
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Load model
    model = DeeplabMulti(num_classes = opts.num_classes, # TODO change to opts.num_classes
                              use_se = opts.use_se, 
                              train_bn = opts.train_bn, 
                              norm_style = opts.norm_style,
                              droprate = opts.droprate, 
                        )
    # Load weights
    new_params = restore_model(model, opts, device)
    model.load_state_dict(new_params)
    
    def freeze_model(model, unfreeze_to, bn_unfreeze_affine=False, bn_use_batch_stats=True, verbose=True):
        """Freeze model weights.
        
        The bn arguments change batchnorm behavior in the freezed part of the model.
            bn_unfreeze_affine: `False` freezes gamma and beta to be unlearnable
            bn_use_batch_stats: `True` uses batch statistics to normalize rather than 
                the historic running mean and std. Using only historic stats may be 
                useful for very small batch sizes.
        """
        if unfreeze_to == "head":
            segments_to_unfreeze = np.array(["layer5.head.1", "layer6.head.1"])
        elif unfreeze_to == "aspp_main":
            segments_to_unfreeze = np.array(["layer6"])
        elif unfreeze_to == "last_conv_aspp_both":
            segments_to_unfreeze = np.array(["layer4", "layer5", "layer6"])
        else: # unfreeze everything
            segments_to_unfreeze = np.array(["layer"+str(i) for i in range(1,7)])
        
        # Apply unfreeze
        for name, m in model.named_modules():
            to_unfreeze = True if np.any([name.startswith(prefix) for prefix in segments_to_unfreeze]) \
                               else False
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not to_unfreeze:
                # For a batchnorm layer
                do_grad = bn_unfreeze_affine # decides if new gamma, beta learned
                do_batch_stats = bn_use_batch_stats # decides if normalizing with batch or historic running stats
                for p in m.parameters(recurse=False):
                    p.requires_grad = do_grad
                m.train(do_batch_stats)
            else:
                # For a general layer
                for p in m.parameters(recurse=False):
                    p.requires_grad = to_unfreeze # True if we want to unfreeze
                m.train(to_unfreeze)
                
        # Print unfreeze outcomes
        if verbose:
            print("Unfreezing down to", unfreeze_to)
            param_ct = 0
            for name, m in model.named_modules():
                to_unfreeze = False
                for p in m.parameters(recurse=False):
                    if p.requires_grad:
                        param_ct += np.prod(p.size())
                        to_unfreeze = True
                if to_unfreeze:
                    print("    ", name)
            print("Numbers of parameters: ", param_ct)

    freeze_model(model, unfreeze_to=opts.unfreeze_to, verbose=False)

    # Set up optimizer with unfrozen layers
    optimizer = torch.optim.SGD(params=model.optim_parameters(opts), lr=opts.learning_rate, momentum=0.9, weight_decay=opts.weight_decay)
    
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=0, alpha = opts.focal_alpha, gamma = opts.focal_gamma, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    def save_ckpt(path):
        """ save current model
        """
#         torch.save({
#             "cur_itrs": cur_itrs,
#             "model_state": model.module.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "scheduler_state": scheduler.state_dict(),
#             "best_score": best_score,
#         }, path, _use_new_zipfile_serialization=False)
        torch.save(model.state_dict(), path)
        print("Model saved as %s" % path)
    
#     # Restore training checkpoint
    utils.mkdir('checkpoints/%s_%s' % (opts.goal_name, opts.exp_name))
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
#     if opts.ckpt is not None and os.path.isfile(opts.ckpt) and opts.continue_training:
#         checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
#         scheduler.load_state_dict(checkpoint["scheduler_state"])
#         cur_itrs = checkpoint["cur_itrs"]
#         best_score = checkpoint['best_score']
#         print("Training state restored from %s" % opts.ckpt)
#         del checkpoint  # free memory

    # Parallelize model for training
#     model = nn.DataParallel(model)
    model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
#         simple_eval(model=model, device=device, metrics=metrics)
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            _, outputs = model(images) # aux, final classifier outputs
#             _, outputs = torch.utils.checkpoint.checkpoint(model, (images)) # aux, final classifier outputs
            # upsample to original image size
            outputs = nn.Upsample(size=list(labels.shape)[1:], mode='bilinear', align_corners=True)(outputs)
#             print(aux_outputs.shape, outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/%s_%s/latest_%.5f_%d.pth' %
                    (opts.goal_name, opts.exp_name, opts.learning_rate, opts.gamma))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/%s_%s/latest_%.5f_%d.pth' %
                        (opts.goal_name, opts.exp_name, opts.learning_rate, opts.gamma))
#                     save_ckpt('checkpoints/%s_%s/best_%d_%d.pth' %
#                         (opts.goal_name, opts.exp_name, opts.learning_rate, opts.batch_size))
                
                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()  

            if cur_itrs >=  opts.total_itrs:
                return
        
if __name__ == '__main__':
    main()

    
def simple_eval(model, device, metrics):
    """Hacky prediction on a single image"""
    # load data
    def preprocess(arr):
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr
    
#     images = preprocess(np.load('../data/imgs/2_10_0_0.npy'))
#     labels = preprocess(np.load('../data/masks/2_10_0_0.npy'))
    images = np.array(Image.open('../data/staticmap_ex.png'))[:, :, :3] # alpha channel removed
    images = preprocess(images)
    
    # run thru model
    with torch.no_grad():    
        images_t = torch.from_numpy(images).to(device, dtype=torch.float32)
#         labels_t = torch.from_numpy(labels).to(device, dtype=torch.long)

        outputs = model(images_t)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
#         targets = labels_t.cpu().numpy()
        
#         print(preds.shape, targets.shape) # 19 bands (classes) vs 3 bands (RGB)
    
    # plotting
#     for image, pred, label in zip(images, preds, labels):    
    for image, pred in zip(images, preds):    
        image = image.transpose(1, 2, 0)
        pred = Cityscapes.decode_target(pred).astype(np.uint8)
#         label = label.transpose(1, 2, 0)
#         display([image, label, pred], save=True)
        display([image, pred], save=True)

