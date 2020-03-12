import numpy as np
import os
import pandas as pd
import json
import ctypes
import matplotlib.pyplot as plt

os.chdir("/home/shuai/trading-game/rl_game/game")
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
    "AliveAskPrice1",
    "AliveAskVolume1",
    "AliveAskPrice2",
    "AliveAskVolume2",
    "AliveAskPrice3",
    "AliveAskVolume3",

    "score",
    "profit",
    "total_profit",
    "action",
    "reward"
]
count = [
    "AliveBidPriceNUM",
    "AliveBidVolumeNUM",
    "AliveAskPriceNUM",
    "AliveAskVolumeNUM",
]

all_data = []

for start_day in range(1, 12):

    arr_len = 100
    arr1 = ctypes.c_int * arr_len
    arr = ctypes.c_int * 1

    actions = arr1()
    action_len = arr()
    infos = arr1()
    infos_len = arr()
    rewards = arr1()
    rewards_len = arr()

    start_info = {"date_index": f"{start_day} - {start_day}", "skip_steps": 10000}
    ctx = expso.CreateContext(json.dumps(start_info).encode())
    # print(start_info)

    score = []
    epscore = 0
    last_score = 0

    step = 1
    while True:

        expso.GetInfo(ctx, infos, infos_len)
        expso.GetReward(ctx, rewards, rewards_len)

        epscore += rewards[0]

        score.append(rewards[0]-last_score)

        last_score = rewards[0]

        # print(infos[26])
        # print(rewards[0], rewards[3])
        # print(infos[1], infos[23])

        # info_dict = {}
        # for i in range(40):
        #     info_dict[info_names[i]] = infos[i]
        # for i in range(3):
        #     info_dict[info_names[i + 40]] = rewards[i]
        # all_data.append(info_dict)

        done = infos[0]
        if done == 1 or step == 3000:
            plt.plot(score)
            plt.show()
            print("step:", step)
            expso.ReleaseContext(ctx)
            break

        expso.Step(ctx)
        step += 1

# print(len(all_data))
# all_data_df = pd.DataFrame(all_data)
# print(all_data_df.info())
# all_data_df.to_csv("/home/shuai/day_1_no_action.csv", index=False)
print("Done!")
