# DEMO_LOOP_TASKS = [("assembling-kits", 3, 2), ("place-red-in-green", 8, 2), ("color-coordinated-cylinder-on-block", 3, 2)]

python gensim/robocodegen.py task=assembling-kits_eps=5 n_times=2

python gensim/robocodegen.py task=place-red-in-green max_eps=8 n_times=2

python gensim/robocodegen.py task=color-coordinated-cylinder-on-block max_eps=5 n_times=2

python gensim/robocodegen.py task=assembling-kits max_eps=1 seed=5

python gensim/robocodegen.py task=align-bottles-on-line max_eps=3 n_times=1

python gensim/robocodegen.py task=place-red-in-green max_eps=2 seed=8

python gensim/robocodegen.py task=align-rope-along-line max_eps=2

python gensim/robocodegen.py task=color-coordinated-cylinder-on-block max_eps=3 seed=12

python gensim/robocodegen.py task=sweeping-piles max_eps=1

python gensim/robocodegen.py task=place-red-in-green max_eps=2 seed=10

python gensim/robocodegen.py task=assembling-kits max_eps=1 seed=9



