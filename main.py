from training.trainer_factory import get_trainer
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_profile', action='store_true')

    # training args
    parser.add_argument('--train_mode', type=str, choices=['reinforce'], default='reinforce')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, choices=['reinforce_loss', 'reinforce_loss_entropy'], default='reinforce_loss')
    parser.add_argument('--reinforce_loss_entropy_alpha', type=float, default=0.6)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--lr_scheduler', type=str, choices=['transformer', 'linear'], default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--lr_scheduler_max_steps', type=int, default=10000)
    parser.add_argument('--lr_min', type=float, default=None)
    parser.add_argument('--lr_linear_scheduler_min_lr', type=float, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_epochs', type=int, default=5)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--metrics', nargs='*', type=str, default=None)
    parser.add_argument('--tb_comment', type=str, default='')
    parser.add_argument('--reinforce_baseline', type=str, choices=['baseline'], default='baseline')
    parser.add_argument('--train_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--eval_shuffle_data', type=bool, default=True)
    parser.add_argument('--n_nodes', type=int, default=50)
    parser.add_argument('--max_n_nodes', type=int, default=None)
    parser.add_argument('--metric_for_best_checkpoint', type=str, choices=['len_to_ref_len_ratio', 'avg_tour_len', 'avg_tour_len_ils'], default=None)
    parser.add_argument('--best_is_highest', action='store_true')

    # fine-tuning
    parser.add_argument('--override_lr', action='store_true')
    parser.add_argument('--override_optim', action='store_true')
    parser.add_argument('--override_lr_scheduler', action='store_true')
    
    # model args
    parser.add_argument('--model', type=str, choices=['custom', 'baseline', 'networkx'], default='custom')
    parser.add_argument('--in_features', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout_p', type=float, default=0.)
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu')
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--norm_eps', type=float, default=1e-5)
    parser.add_argument('--norm_first', type=bool, default=False)
    parser.add_argument('--num_hidden_encoder_layers', type=int, default=2)
    parser.add_argument('--sinkhorn_tau', type=float, default=5e-2)
    parser.add_argument('--sinkhorn_i', type=int, default=20)
    parser.add_argument('--add_cross_attn', type=bool, default=True)
    parser.add_argument('--use_q_proj_ca',  action='store_true')
    parser.add_argument('--use_feedforward_block_sa', action='store_true')
    parser.add_argument('--use_feedforward_block_ca', action='store_true')
    parser.add_argument('--positional_encoding', type=str, choices=['sin', 'custom_sin', 'custom'], default='sin')
    # baseline
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_hidden_decoder_layers', type=int, default=2)
    parser.add_argument('--clip_logit_c', type=int, default=None)

    # ILS
    parser.add_argument('--ils_n_restarts', type=int, default=5)
    parser.add_argument('--ils_n_iterations', type=int, default=10)
    parser.add_argument('--ils_n_permutations', type=int, default=15)
    parser.add_argument('--ils_n_permutations_hillclimbing', type=int, default=7)
    parser.add_argument('--ils_k', type=int, default=0)
    parser.add_argument('--ils_max_perturbs', type=int, default=None)

    # profiling
    parser.add_argument('--filename', type=str, default='')


    args = parser.parse_args()

    trainer = get_trainer(args)
    if args.do_train:
        train_result = trainer.do_train()
    elif args.do_eval or args.do_test:
        eval_result = trainer.do_eval()
    elif args.do_profile:
        profile_result = trainer.do_eval()