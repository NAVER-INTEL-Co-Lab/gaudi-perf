#!/usr/bin/env bash

# Destination is in literal quotes to prevent using the local home directory.
dest='~/PyCharmProjects'

for host in "$@"
do
  rsync -rP ../gaudi-perf \
      --exclude ".git" \
      --filter=":- .gitignore" \
      "${host}:${dest}" &
done
wait
