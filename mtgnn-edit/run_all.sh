#!/bin/bash

# pendulum
sbatch -o logs/pendulum/pendulum_short-%j.log -c 8 --job-name=pendulum-short -p gpu -M ukko --gres=gpu:1 --mem 64G -t 0-2:00:00 --exclude ukko3-g601 experiments/pendulum_short.sh
sbatch -o logs/pendulum/pendulum_long-%j.log -c 8 --job-name=pendulum-long -p gpu -M ukko --gres=gpu:1 --mem 64G -t 0-2:00:00 --exclude ukko3-g601 experiments/pendulum_long.sh

# short weather and pp
sbatch -o logs/weather/usw-short-%j.log -c 8 --job-name=usw-short -p gpu -M ukko --gres=gpu:1 --mem 64G -t 0-12:00:00 --mail-type=END experiments/us_weather_short.sh
sbatch -o logs/powerplant/pp-short-%j.log -c 8 --job-name=pp-short -p gpu -M ukko --gres=gpu:1 --mem 64G -t 1-00:00:00 experiments/powerplant_short.sh

# long weather and pp
sbatch -o logs/weather/usw-mid-%j.log -c 8 --job-name=usw-mid -p gpu -M ukko --gres=gpu:8 --mem 256G -t 1-12:00:00 --mail-type=END  experiments/us_weather_mid.sh


# powerplant
#sbatch -o logs/powerplant/pp-mid-%j.log -c 8 --job-name=pp-mid -p gpu -M ukko --gres=gpu:8 --mem 64G -t 1-0:00:00 --exclude=dgx1-01,dgx1-02 experiments/powerplant_mid.sh
sbatch -o logs/powerplant/pp-mid-%j.log -c 8 --job-name=pp-mid -p gpu -M ukko --gres=gpu:8 --mem 450G -t 2-0:00:00 experiments/powerplant_mid.sh
