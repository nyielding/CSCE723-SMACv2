import pytest
import os
import re
import pathlib

def test_submission_dir():
    sub_folder = pathlib.Path(__file__).parent.parent.joinpath('submission')   
    for i, (root, dirs, files) in enumerate(os.walk(sub_folder)):
        # print(i)
        # print('root\n', root)
        # print('dirs\n', dirs)
        # print('files\n', files)
        if i == 0:
            assert 'YOUR-LAST-NAME' not in dirs, "Change the YOUR-LAST NAME folder to your name"
            assert not files, "Files go below YOUR-LAST-NAME folder"
        if i == 1: 
            assert any(re.match(r"checkpoint_*", item) for item in dirs), 'Did not find checkpoint folder'
            assert 'SUBMISSION.md'in files, "SUBMISSION.md got moved?"
            assert any(re.search(r".SC2Replay", item) for item in files), "Please submit a .SC2Replay file"
        if i == 2: 
            assert 'policies' in dirs, "policies folder not in right spot"
            for file in files:
                assert file in ['rllib_checkpoint.json', 'algorithm_state.pkl']
        if i == 3:
            assert 'marine' in dirs, "a policy for marine unit agents is required"
            assert 'marauder' in dirs, "a policy for marauder unit agents is required"
            assert 'medivac' in dirs, "a policy for medivac unit agents is required"
        if i == 4:
            for file in files:
                assert file in['rllib_checkpoint.json', 'policy_state.pkl']

# test_submission_dir()