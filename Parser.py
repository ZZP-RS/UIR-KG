import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument("--seed", type=int, default=2023, help="random seed.")

    # nargs=?，如果没有在命令行中出现对应的项，则给对应的项赋值为default。
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        help="choose a dataset from {last-fm,amazon-book,yelp2018}")
    parser.add_argument("--dataset_dir", nargs="?", default="datasets", help="datasets path")

    parser.add_argument("--use_pretrain", type=int, default=1,
                        help="0:No pretrain, "
                             "1:Pretrain with users and items embeddings, "
                             "2:Pretrain with stored model.")
    parser.add_argument("--pretrain_embeddings_dir", nargs="?", default="datasets/pretrain/",
                        help="path of pretrain embeddings")
    parser.add_argument("--pretrain_model_path", nargs="?", default="train_model/model.pth",
                        help="path of stored model")

    parser.add_argument("--cf_batch_size", type=int, default=1024, help="CF batch size")
    parser.add_argument("--kg_batch_size", type=int, default=2048, help="KG batch size")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Test batch size")

    parser.add_argument("--embedding_dim", type=int, default=64, help="user or entity embedding size")
    parser.add_argument("--relation_dim", type=int, default=64, help="relation embedding size")

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    # parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
    #                     help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument("--conv_dim_list", nargs="?", default="[64,32,16]",
                        help="output size of every aggregation layer.")
    parser.add_argument("--mess_dropout", nargs="?", default="[0.1,0.1,0.1]",
                        help="message dropout probability for each layer")

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=1000, help="Maximum of epoch")
    parser.add_argument("--stopping_steps", type=int, default=10, help="number of epoch for early stopping")

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    save_dir = 'trained_model/UIR-KG/{}/embed-dim{}_relation-dim{}_{}_{}_lr{}_pretrain{}/'.format(
        args.dataset, args.embedding_dim, args.relation_dim, args.laplacian_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args