# parser.add_argument("--max_len", type=int, default=70)
# parser.add_argument("--model_name", type=str, default='resnet50')
# parser.add_argument("--epochs", type=int, default=25)
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--margin", type=float, default=0.5)
# parser.add_argument("--s", type=float, default=30)
# parser.add_argument("--pool", type=str, default="gem")
# parser.add_argument("--dropout", type=float, default=0.5)
# parser.add_argument("--last_hidden_states", type=int, default=3)
# parser.add_argument("--fc_dim", type=int, default=512)
# parser.add_argument("--lr", type=float, default=0.00001)
# parser.add_argument("--weight_decay", type=float, default=1e-5)
# parser.add_argument("--l2_wd", type=float, default=1e-5)
# parser.add_argument("--metric", type=str, default="adacos")
# parser.add_argument("--input_path", type=str)
# parser.add_argument("--smooth_ce", type=float, default=0.0)
# parser.add_argument("--warmup_epoch", type=int, default=10)
# parser.add_argument("--verbose", type=int, default=0)

class Config(object):
    max_len = 70
    model_name = None
    epochs = 25
    batch_size = 32
    margin = 0.5
    s = 30
    pool = "gem"
    dropout = 0.5
    last_hidden_states = 3
    fc_dim = 512
