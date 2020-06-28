import ctypes
import json
import os
import sys
import pandas as pd
pd.set_option('display.max_columns', None)

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR+"/rl_game/game")

# import game.so
os.chdir(ROOT_DIR+"/rl_game/game")
soFile = "./game.so"
expso = ctypes.cdll.LoadLibrary(soFile)

info_names = [
    "Done",
    "LastPrice",
    "BidPrice1",
    "BidVolume1",
    "AskPrice1",
    "AskVolume1",
    "BidPrice2",
    "BidVolume2",
    "AskPrice2",
    "AskVolume2",
    "BidPrice3",
    "BidVolume3",
    "AskPrice3",
    "AskVolume3",
    "BidPrice4",
    "BidVolume4",
    "AskPrice4",
    "AskVolume4",
    "BidPrice5",
    "BidVolume5",
    "AskPrice5",
    "AskVolume5",
    "Volume",
    "HighestPrice",
    "LowestPrice",
    "TradingDay",
    "Target_Num",
    "Actual_Num",
    "AliveBidPrice1",
    "AliveBidVolume1",
    "AliveBidPrice2",
    "AliveBidVolume2",
    "AliveBidPrice3",
    "AliveBidVolume3",
    "AliveBidPriceNUM",
    "AliveBidVolumeNUM",
    "AliveAskPrice1",
    "AliveAskVolume1",
    "AliveAskPrice2",
    "AliveAskVolume2",
    "AliveAskPrice3",
    "AliveAskVolume3",
    "AliveAskPriceNUM",
    "AliveAskVolumeNUM",
    "score",
    "profit",
    "total_profit",
    "baseline_profit",
    "close_profit",
    "action",
]

arr_len = 100
arr1 = ctypes.c_int * arr_len
arr = ctypes.c_int * 1

actions = arr1()
action_len = arr()
infos = arr1()
infos_len = arr()
rewards = arr1()
rewards_len = arr()

all_data = []

for start_day in range(1, 91):

    # day_data = []

    start_info = {"date_index": f"{start_day} - {start_day}", "skip_steps": 0}
    ctx = expso.CreateContext(json.dumps(start_info).encode())

    expso.GetInfo(ctx, infos, infos_len)
    expso.GetReward(ctx, rewards, rewards_len)

    step = 1
    action = 0
    while True:

        expso.GetInfo(ctx, infos, infos_len)
        expso.GetReward(ctx, rewards, rewards_len)

        info_dict = {}
        for i in range(44):
            info_dict[info_names[i]] = infos[i]
        for i in range(5):
            info_dict[info_names[i + 44]] = rewards[i]
        info_dict[info_names[48]] = action
        # print(info_dict)
        # day_data.append(info_dict)
        all_data.append(info_dict)

        done = infos[0]
        if done == 1:
            print("Day", infos[25], "data_len:", step)
            # day_data_df = pd.DataFrame(day_data)
            # day_data_df.to_csv(ROOT_DIR + "/r18-day" + str(start_day) + "-baseline_policy.csv")
            # print("day" + str(start_day) + "data saved in " + ROOT_DIR + "/r18-day" + str(
            #     start_day) + "-baseline_policy.csv")
            expso.ReleaseContext(ctx)
            break

        target_num = infos[26]
        actual_num = infos[27]

        action = 0

        #         if abs(actual_num - target_num) > 5:
        #             if target_num > actual_num:
        #                 action = 6
        #             else:
        #                 action = 9

        expso.Action(ctx, action)
        expso.Step(ctx)
        step += 1
all_data_df = pd.DataFrame(all_data)
print(all_data_df.tail())
print(all_data_df.describe())
all_data_df.to_csv(ROOT_DIR + "/r18-all_data.csv")
print("all_data saved in " + ROOT_DIR + "/r18-90days_data.csv")
