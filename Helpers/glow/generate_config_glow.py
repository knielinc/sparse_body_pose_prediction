from Helpers.glow import config

def generate_cfg(): # hidden_channels, N, num_layers, framerate, seq_len):
    return_dict = {
        "Dir": {
            "data_root": "data/locomotion",
            "log_root": "results/locomotion"
        },
        "Glow" : {
            "hidden_channels": 512,
            "K": 8,
            "actnorm_scale": 1.0,
            "flow_permutation": "invconv",
            "flow_coupling": "affine",
            "network_model": "LSTM",
            "num_layers": 3,
            "LU_decomposed": True,
            "distribution": "normal"
        },
        "Data" : {
            "framerate": 20,
            "seqlen": 40,
            "n_lookahead":0,
            "dropout":0.95,
            "mirror":True,
            "reverse_time":True
        },
        "Optim": {
            "name": "adam",
            "args": {
                "lr": 1e-2,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            },
            "Schedule": {
                "name": "noam_learning_rate_decay",
                "args": {
                    "warmup_steps": 1000,
                    "minimum": 1e-4
                }
            }
        },
        "Device": {
            "glow": ["cuda:0"],
            "data": "cuda:0"
        },
        "Train": {
            "batch_size": 100,
            "num_batches": 160000,
            "max_grad_clip": 5,
            "max_grad_norm": 100,
            "max_checkpoints": 20,
            "checkpoints_gap": 40000,
            "num_plot_samples": 1,
            "scalar_log_gap": 50,
            "validation_log_gap": 200,
            "plot_gap": 20000,
            "warm_start": ""
        },
        "Infer": {
            "pre_trained": ""
        }
    }
    return config.JsonConfig(return_dict)
