# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from simvp.api import NonDistExperiment
from simvp.utils import create_parser, load_config, update_config

import json
import os
import pandas as pd
import time
import traceback


try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':

    try:
        start_time = time.time()
        args = create_parser().parse_args()
        config = args.__dict__

        if has_nni:
            tuner_params = nni.get_next_parameter()
            config.update(tuner_params)

        cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
            if args.config_file is None else args.config_file
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method', 'batch_size', 'val_batch_size', 'sched'])

        log_file = f"./{args.dataname}_summary.txt"
        exp_data_file = f"./{args.dataname}_exp_data.csv"
        summary_for_log_file = ""
        exp_records_df = None
        if osp.exists(exp_data_file):
            exp_records_df = pd.read_csv(exp_data_file)
            exp_records_df = exp_records_df.append(config, ignore_index=True)
        else:
            exp_records_df = pd.DataFrame.from_dict(config, orient='index').T

        exp_records_df.loc[len(exp_records_df)-1, "user"] = str(os.getlogin())
        print(f"Running {args.method} for {args.dataname}")
        summary_for_log_file += f"---{args.method}---\n"
        summary_for_log_file += f"Config: {osp.basename(args.config_file)}\n"
        exp = NonDistExperiment(args)
        print('>'*35 + ' training ' + '<'*35)
        exp.train()

        train_time = time.time() - start_time
        exp_records_df.loc[len(exp_records_df)-1, "train_time"] = train_time
        summary_for_log_file += f"Train time (s): {train_time}\n"

        start_time_test = time.time()
        print('>'*35 + ' testing  ' + '<'*35)
        test_time = time.time() - start_time_test
        exp_records_df.loc[len(exp_records_df)-1, "test_time"] = test_time
        summary_for_log_file += f"Test time (s): {start_time_test}\n"

        mse = exp.test()
        exp_records_df.loc[len(exp_records_df)-1, "mse"] = mse
        summary_for_log_file += f"MSE: {mse}\n"
        if has_nni:
            nni.report_final_result(mse)
    except KeyboardInterrupt:
        summary_for_log_file += f"Killed manually\n"
        # If we kill the run, don't record it since it's probably not useful
        exp_records_df = None

    except Exception as e:
        f = open(f"./{args.ex_name}_err.txt", "w")
        f.write(str(e))
        f.write("\n")
        f.write(traceback.format_exc())
        f.close()

        summary_for_log_file += f"ERROR: See {args.ex_name}_err.txt for details\n"

        exp_records_df.loc[len(exp_records_df)-1, "error"] = str(e)
        traceback.print_exc()
    finally:
        f = open(log_file, "a")
        f.write(summary_for_log_file)
        f.close()
        if exp_records_df is not None:
            exp_records_df.to_csv(exp_data_file, index=False)
