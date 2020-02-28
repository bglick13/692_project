import pickle
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import pandas as pd

from draft.draft_env import CaptainModeDraft


def do_rollout(hero_ids, port, verbose=False):

    draft = CaptainModeDraft(hero_ids, port)
    state = draft.reset()
    turn = 0

    all_actions = []
    all_states = []

    while True:
        legal_moves = draft.state.get_legal_moves
        action = np.random.choice(legal_moves)
        all_states.append(state)
        all_actions.append(action)
        state, value, done = draft.step(action)

        if value == 0:  # Dire victory
            print('Dire victory')
            break
        elif value == 1:
            print('Radiant Victory')
            break
        elif done:
            print('Done but no victory')
            break
        turn += 1
    all_actions.append(action)
    all_states.append(state)

    # TODO: I'm really not confident this is right - it's worth double and triple checking
    all_values = [value] * 22
    # all_values[[0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 20]] = value
    # all_values[[1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 21]] = 1 - value
    return all_actions, all_states, all_values


if __name__ == '__main__':
    memory_size = 500000
    n_jobs = 4
    n_games = 50
    port = 13337
    verbose = True
    hero_ids = pd.read_json('../const/draft_bert_hero_ids.json', orient='records')


    start = time.time()
    for batch_of_games in range(n_games // n_jobs):
        # pool = ProcessPoolExecutor(2)
        pool = Pool(n_jobs)
        results = pool.starmap(do_rollout, [(hero_ids, port + i) for i in range(n_jobs)])
    print(f'Played {n_games} games using {n_jobs} jobs in {time.time() - start}s')
