def create_deepspeed_config(args):
    ds_config = {
        "steps_per_print": 1000,
        "train_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # "train_micro_batch_size_per_gpu": args.batch_size, # determined by (train_batch_size, gradient_accumulation_steps) 
        "optimizer": {
            "type": "Adam",
            "adam_w_mode": True,
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "bias_correction": True,
                "betas": [
                    args.beta1,
                    args.beta2
                ],
            }
        },
        "fp16": {
            "enabled": args.mixed_precision == 'fp16',
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": args.mixed_precision == 'bf16',
        },
        # "flops_profiler": {
        #     "enabled": True,
        #     "profile_step": -1,
        #     "module_depth": -1,
        #     "top_modules": 1,
        #     "detailed": True,
        # },
        "zero_allow_untested_optimizer": True
    }

    if args.clip_grad is not None:
        ds_config.update({'gradient_clipping': args.clip_grad})

    if args.zero_stage == 0:
        ds_config.update({"zero_optimization": 
        {
            "stage": args.zero_stage, 
            "contiguous_gradients": True,
            "overlap_comm": True,
        }
    })
    elif args.zero_stage == 1:
        ds_config.update({"zero_optimization": 
        {
            "stage": args.zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
        }
    })
    elif args.zero_stage == 2:
        ds_config.update({"zero_optimization": 
        {
            "stage": args.zero_stage, 
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
    })
    elif args.zero_stage == 3:
        ds_config.update({"zero_optimization": 
        {
            "stage": args.zero_stage, 
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        }
    })

    return ds_config
