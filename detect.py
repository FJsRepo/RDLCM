import os
import sys
import random
import logging
import argparse
from time import time
import numpy as np
import torch
from tqdm import tqdm
from lib.config import Config
from utils.target_detection import target_detection

def detecter(model, SSL_target_loader, exp_root, cfg, threshold, cellWidth, epoch, max_batches=None, verbose=True):
    if verbose:
        logging.info("Starting detect.")
    if epoch > 0:
        model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])
        print("load trained parameter!")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_parameters = cfg.get_test_parameters()
    time_consume = 0
    totalDetectedTargets = 0
    totalGtTargets = 0
    totalTD = 0
    totalFD = 0
    totalTU = 0
    with torch.no_grad():
        loop = tqdm(enumerate(SSL_target_loader), total=len(SSL_target_loader), ncols=100)
        for idx, (images, labels, img_idxs) in loop:
            if max_batches is not None and idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            t0 = time()
            outputs = model(images)
            outputs = model.decode(outputs, labels, **test_parameters)
            t = time() - t0

            TD, FD, TU, td_diff, detectedNum, gtNum, time_consume_one_frame = target_detection(t, outputs, SSL_target_loader, idx, threshold, cellWidth)

            time_consume = time_consume + time_consume_one_frame
            totalDetectedTargets = totalDetectedTargets + detectedNum
            totalGtTargets = totalGtTargets + gtNum
            totalTD = totalTD + TD
            totalFD = totalFD + FD
            totalTU = totalTU + TU

    return totalDetectedTargets, totalGtTargets, totalTD, totalFD, totalTU, time_consume

def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to test the model on")
    parser.add_argument("--batch_size", type=int, help="Number of images per batch")
    parser.add_argument("--view", action="store_true", help="Show predictions")

    return parser.parse_args()

def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"] if args.batch_size is None else args.batch_size

    model = cfg.get_model().to(device)
    total_epoch = cfg["epochs"]
    model_save_interval = cfg["model_save_interval"]

    SSL_target_dataset = cfg.get_dataset("SSL_target")

    SSL_target_loader = torch.utils.data.DataLoader(dataset=SSL_target_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=8)
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

##################################################################################################################
    # thresholds = [0.2, 0.4, 1.0, 1.2, 1.5, 1.9, 2.2, 2.6]
    thresholds = [2.0]

    cellWidth = 3
    TRr = []
    FARr = []
    time_consumes = []
    total_pixels = 288 * 384
    for threshold in thresholds:
        totalDetectedTargets, totalGtTargets, totalTD, totalFD, totalTU, time_consume = detecter(model, SSL_target_loader,
                                                                                                 exp_root, cfg,
                                                                                                 threshold=threshold,
                                                                                                 cellWidth=cellWidth,
                                                                                                 epoch=400)
        logging.info("threshold: {:.3f}".format(threshold))
        logging.info("Time Consume: {:.4f}".format(time_consume))
        logging.info("Total Detected Targets: {:.0f}".format(totalDetectedTargets))
        logging.info("Total GT Targets: {:.0f}".format(totalGtTargets))
        logging.info("Total True Detected Targets: {:.00f}".format(totalTD))
        logging.info("Total False Detected Targets: {:.0f}".format(totalFD))
        logging.info("Total Undetected True Targets: {:.0f}".format(totalTU))
        logging.info("TR rate: {:.4f}".format(totalTD / totalGtTargets))
        logging.info("FAR rate: {:.5f}".format(totalFD / total_pixels))
        TRr.append(round(totalTD / totalGtTargets, 4))
        FARr.append(round(totalFD / total_pixels, 5))
        time_consumes.append(round(time_consume, 2))

    Average_time_consume = np.mean(time_consumes)
    print("Threshold:")
    print(thresholds)
    print("TR rate:")
    print(TRr)
    print("FAR rate:")
    print(FARr)
    print("Time Consume:")
    print(time_consumes)
    print("Average Time Consume:")
    print(Average_time_consume)
    print("FPS:")
    print(round(195 / Average_time_consume, 2))