for VARIABLE in 1 2 3 4 5
do
    rocprof --stats -o naive_FA_2028.csv python3 stableLM_profile_use.py
    python3 help_parsing.py
done
python3 help_read_log.py
