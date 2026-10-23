#!/usr/bin/env bash
# waits for the soup run, then runs the recipe sweep (one GPU job at a time)
cd "$(dirname "$0")/.."
while pgrep -f night_soup3 >/dev/null 2>&1 || ps -ef | grep -v grep | grep -q night_soup3; do sleep 60; done
bash adapter/night_recipe.sh 2>&1 | grep --line-buffered -E "^(===|==|official|rel_|p_arm|\{)" > logs/night_recipe.log
