import os
import sys

sys.path.append(os.path.abspath("./"))
from model import stage_4
from utils.opt import Options
from utils import util
from utils import log

import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import tqdm

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

datapath_save_out = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
config = {
    "item_step": 2,
    "window_step": 2,
    # "input_n": 50,
    "input_n": 10,
    "output_n": 25,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}


# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print(">>> create models")
    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)

    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now
    )
    print(
        ">>> total params: {:.2f}M".format(
            sum(p.numel() for p in net_pred.parameters()) / 1000000.0
        )
    )

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = "./{}/ckpt_best.pth.tar".format(opt.ckpt)
        else:
            model_path_len = "./{}/ckpt_last.pth.tar".format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt["epoch"] + 1
        err_best = ckpt["err"]
        lr_now = ckpt["lr"]
        net_pred.load_state_dict(ckpt["state_dict"])
        print(
            ">>> ckpt len loaded (epoch: {} | err: {})".format(
                ckpt["epoch"], ckpt["err"]
            )
        )

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = utils_pipeline.load_dataset(
        datapath_save_out, "train", config
    )
    esplit = "test" if "mocap" in datapath_save_out else "eval"

    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        datapath_save_out, esplit, config
    )
    dataset_test, dlen_test = utils_pipeline.load_dataset(
        datapath_save_out, "test", config
    )

    # evaluation
    if opt.is_eval:
        # Load preprocessed datasets
        label_gen_test = utils_pipeline.create_labels_generator(
            dataset_test["sequences"], config
        )

        stime = time.time()
        ret_test = run_model(
            net_pred, is_train=3, data_loader=label_gen_test, opt=opt, dlen=dlen_test
        )
        ftime = time.time()
        print("Testing took {} seconds".format(int(ftime - stime)))

        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name="test_walking")

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print(">>> training epoch: {:d}".format(epo))

            label_gen_train = utils_pipeline.create_labels_generator(
                dataset_train["sequences"], config
            )
            label_gen_eval = utils_pipeline.create_labels_generator(
                dataset_eval["sequences"], config
            )
            label_gen_test = utils_pipeline.create_labels_generator(
                dataset_test["sequences"], config
            )

            ret_train = run_model(
                net_pred,
                optimizer,
                is_train=0,
                data_loader=label_gen_train,
                epo=epo,
                opt=opt,
                dlen=dlen_train,
            )
            print("train error: {:.3f}".format(ret_train["m_p3d_h36"]))
            ret_valid = run_model(
                net_pred,
                is_train=1,
                data_loader=label_gen_eval,
                opt=opt,
                epo=epo,
                dlen=dlen_eval,
            )
            print("validation error: {:.3f}".format(ret_valid["m_p3d_h36"]))
            ret_test = run_model(
                net_pred,
                is_train=3,
                data_loader=label_gen_test,
                opt=opt,
                epo=epo,
                dlen=dlen_test,
            )
            print("testing error: {:.3f}".format(ret_test["#40ms"]))

            ret_log = np.array([epo, lr_now])
            head = np.array(["epoch", "lr"])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ["valid_" + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ["test_" + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid["m_p3d_h36"] < err_best:
                err_best = ret_valid["m_p3d_h36"]
                is_best = True
            log.save_ckpt(
                {
                    "epoch": epo,
                    "lr": lr_now,
                    "err": ret_valid["m_p3d_h36"],
                    "state_dict": net_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                opt=opt,
            )
            torch.cuda.empty_cache()


def eval(opt):
    print(">>> create models")
    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # load model
    model_path_len = "./{}/ckpt_best.pth.tar".format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt["state_dict"])

    print(
        ">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt["epoch"], ckpt["err"])
    )

    # Load preprocessed dataset
    dataset_test, dlen_test = utils_pipeline.load_dataset(
        datapath_save_out, "test", config
    )
    label_gen_test = utils_pipeline.create_labels_generator(
        dataset_test["sequences"], config
    )

    stime = time.time()
    ret_test = run_model(
        net_pred, is_train=3, data_loader=label_gen_test, opt=opt, dlen=dlen_test
    )
    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))

    ret_log = np.array([])
    head = np.array([])
    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, [k])
    log.save_csv_log(opt, head, ret_log, is_create=True, file_name="test_all")


def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size : i + 1], dim=1)
    return smooth_data


def run_model(
    net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dlen=0
):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1) * 40
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size

    itera = 1

    if is_train == 0:
        nbatch = opt.batch_size
    else:
        nbatch = opt.test_batch_size

    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
        total=int(dlen / nbatch),
    ):
        batch_size = nbatch
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size

        sequences_train = prepare_sequences(batch, nbatch, "input", device)
        sequences_gt = prepare_sequences(batch, nbatch, "target", device)
        p3d_h36 = torch.cat([sequences_train, sequences_gt], dim=1)

        smooth1 = smooth(
            p3d_h36[:, :, :],
            sample_len=opt.kernel_size + opt.output_n,
            kernel_size=opt.kernel_size,
        ).clone()

        smooth2 = smooth(
            smooth1,
            sample_len=opt.kernel_size + opt.output_n,
            kernel_size=opt.kernel_size,
        ).clone()

        smooth3 = smooth(
            smooth2,
            sample_len=opt.kernel_size + opt.output_n,
            kernel_size=opt.kernel_size,
        ).clone()

        input = p3d_h36[:, :, :].clone()

        p3d_sup_4 = p3d_h36.clone()[:, :, :][:, -out_n - seq_in :].reshape(
            [-1, seq_in + out_n, opt.in_features // 3, 3]
        )
        p3d_sup_3 = smooth1.clone()[:, -out_n - seq_in :].reshape(
            [-1, seq_in + out_n, opt.in_features // 3, 3]
        )
        p3d_sup_2 = smooth2.clone()[:, -out_n - seq_in :].reshape(
            [-1, seq_in + out_n, opt.in_features // 3, 3]
        )
        p3d_sup_1 = smooth3.clone()[:, -out_n - seq_in :].reshape(
            [-1, seq_in + out_n, opt.in_features // 3, 3]
        )

        p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = net_pred(
            input, input_n=in_n, output_n=out_n, itera=itera
        )

        p3d_out_4 = p3d_h36.clone()[:, in_n : in_n + out_n]
        p3d_out_4[:, :, :] = p3d_out_all_4[:, seq_in:]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, opt.in_features // 3, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, opt.in_features // 3, 3])

        p3d_out_all_4 = p3d_out_all_4.reshape(
            [batch_size, seq_in + out_n, opt.in_features // 3, 3]
        )
        p3d_out_all_3 = p3d_out_all_3.reshape(
            [batch_size, seq_in + out_n, opt.in_features // 3, 3]
        )
        p3d_out_all_2 = p3d_out_all_2.reshape(
            [batch_size, seq_in + out_n, opt.in_features // 3, 3]
        )
        p3d_out_all_1 = p3d_out_all_1.reshape(
            [batch_size, seq_in + out_n, opt.in_features // 3, 3]
        )

        # 2d joint loss:
        if is_train == 0:
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup_4, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3 - p3d_sup_3, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2 - p3d_sup_2, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup_1, dim=3))

            loss_all = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1) / 4
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size

        if (
            is_train <= 1
        ):  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(
                torch.norm(p3d_h36[:, in_n : in_n + out_n] - p3d_out_4, dim=3)
            )
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(
                torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3), dim=2),
                dim=0,
            )
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == "__main__":
    option = Options().parse()

    if option.is_eval == False:
        stime = time.time()
        main(opt=option)
        ftime = time.time()
        print("Training took {} seconds".format(int(ftime - stime)))
    else:
        eval(option)
